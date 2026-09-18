"""Transport only: what does cutting a step into C frames cost, with no model?

WHY THIS EXISTS
    The pipelined GPT-2 prototype got monotonically SLOWER with more
    micro-batches, at both 124M and 774M. Two explanations fit that curve and
    they have opposite consequences:

      transport  each extra frame costs a round trip, so C frames cost C round
                 trips and no amount of overlap repays it. Architecture
                 independent — gemma would behave the same and there is nothing
                 to fix short of a different transport.
      compute    chunking makes each matmul smaller and the GPU less efficient.
                 Architecture DEPENDENT — a model with more work per token
                 could flip the sign, and gemma would be worth building.

    Petrus asked, fairly, whether GPT-2 is the wrong yardstick. This is the
    measurement that answers it without implementing a second architecture:
    the same frame pattern, the same byte counts, no model at all. Whatever
    this reproduces is transport; whatever is left over is compute.

WHAT IT DOES
    Replays the prototype's exact per-step exchange — rank 0 sends C hidden
    states and then reads C token replies, rank 1 interleaves — with zero
    arithmetic between the frames.
"""

import argparse
import json
import os
import statistics
import time

import torch

from pipeline import PipeChannel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1280, help="hidden size; 1280 = gpt2-large")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.batch % args.chunks:
        ap.error("--batch must divide by --chunks, as in the prototype")

    rank = int(os.environ["RANK"])
    chan = PipeChannel(rank, os.environ.get("MASTER_ADDR", "127.0.0.1"),
                       int(os.environ.get("MASTER_PORT", "29971")))
    size = args.batch // args.chunks
    hidden = torch.zeros(size, 1, args.dim)
    tokens = torch.zeros(size)

    # Warm the path: the first frame pays connection setup and page faults that
    # belong to neither explanation.
    for _ in range(5):
        if rank == 0:
            chan.send("h0", hidden); chan.recv("t0")
        else:
            chan.recv("h0"); chan.send("t0", tokens)

    per_step = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        if rank == 0:
            for c in range(args.chunks):
                chan.send(f"h{c}", hidden)
            for c in range(args.chunks):
                chan.recv(f"t{c}")
        else:
            for c in range(args.chunks):
                chan.recv(f"h{c}")
                chan.send(f"t{c}", tokens)
        per_step.append((time.perf_counter() - t0) * 1000.0)

    report = {
        "rank": rank, "chunks": args.chunks, "batch": args.batch, "dim": args.dim,
        "frames_per_step": 2 * args.chunks,
        "bytes_per_step": args.batch * args.dim * 4 + args.batch * 4,
        # Median, not mean: one scheduler hiccup in 200 steps should not decide
        # a result that is going to be compared across configurations.
        "median_step_ms": round(statistics.median(per_step), 4),
        "p90_step_ms": round(sorted(per_step)[int(0.9 * len(per_step))], 4),
        "min_step_ms": round(min(per_step), 4),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    chan.close()


if __name__ == "__main__":
    main()
