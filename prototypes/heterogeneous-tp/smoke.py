"""FP32 sharded MLP correctness harness. Launch using torchrun, including on Mac.

Full CPU weights are retained for validation; this is not a model-serving engine.
GPU math is local; every Gloo collective receives CPU tensors explicitly.
"""
import argparse
from datetime import timedelta
import json
import os
import platform

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                        choices=["cpu", "mps", "cuda", "rocm"])
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    if args.device in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            parser.error("GPU was requested but is unavailable")
        if bool(torch.version.hip) != (args.device == "rocm"):
            parser.error("Requested GPU vendor does not match this PyTorch build")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda" if args.device == "rocm" else args.device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        dim, hidden = 64, 128
        if world < 2 or hidden % world:
            raise ValueError("Use >=2 ranks, with rank count dividing 128")
        width = hidden // world
        begin, end = rank * width, (rank + 1) * width
        generator = torch.Generator().manual_seed(20260917)

        def shared(shape, scale=1.0):
            value = (torch.randn(shape, generator=generator) * scale
                     if rank == 0 else torch.empty(shape))
            dist.broadcast(value, src=0)
            return value

        with torch.inference_mode():
            w1 = shared((dim, hidden), dim ** -0.5)
            b1 = shared((hidden,), 0.1)
            w2 = shared((hidden, dim), hidden ** -0.5)
            b2 = shared((dim,), 0.1)
            local_w1 = w1[:, begin:end].contiguous().to(device)
            local_b1 = b1[begin:end].contiguous().to(device)
            local_w2 = w2[begin:end, :].contiguous().to(device)
            cases = []
            for tokens in (1, 17, 128):
                x = shared((tokens, dim))
                partial = (F.gelu(x.to(device) @ local_w1 + local_b1)
                           @ local_w2).to("cpu").contiguous()
                dist.all_reduce(partial, op=dist.ReduceOp.SUM)
                # Output bias must be added once, after the sum.
                result = partial + b2
                reference = F.gelu(x @ w1 + b1) @ w2 + b2
                passed = bool(torch.allclose(result, reference,
                                            atol=2e-5, rtol=2e-4))
                cases.append({"tokens": tokens, "passed": passed,
                              "max_abs_error": (result-reference).abs().max().item()})
        report = {"rank": rank, "device": str(device), "requested": args.device,
                  "torch": torch.__version__, "hip": torch.version.hip,
                  "cuda": torch.version.cuda, "host": platform.node(),
                  "cases": cases}
        reports = [None] * world
        dist.all_gather_object(reports, report)
        success = all(c["passed"] for r in reports for c in r["cases"])
        if rank == 0:
            print(json.dumps({"passed": success, "transport": "CPU Gloo",
                              "dtype": "float32", "atol": 2e-5, "rtol": 2e-4,
                              "ranks": reports}, indent=2), flush=True)
        if not success:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
