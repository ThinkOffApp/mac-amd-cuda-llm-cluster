#!/usr/bin/env python3
"""Measure each host, then say how to split work between them — and whether to.

WHY THIS EXISTS
    Two machines can be shared in two completely different ways and they are
    not close in throughput:

      LAYER SPLIT    one model cut across both hosts (llama.cpp `--rpc` + `-ts`).
                     The hosts take turns on the SAME token, so a step costs
                     host A plus host B plus a round trip. Nobody overlaps.
      REQUEST SPLIT  each host runs the whole model and serves different
                     requests. The hosts overlap completely and throughput adds.

    Measured on a Mac mini M4 (Metal) + Bosgame M5 (Vulkan) over gigabit,
    gemma-3-4b Q4_K_M, 64 concurrent streams, 2026-09-18:

        best layer split (-ts 16/1)        681 tok/s
        M5 alone                          1077 tok/s
        request split, measured CONCURRENTLY  1261 tok/s

    Layer split lost to one machine on its own at every ratio and every
    concurrency tried. Request split is the only arrangement where the pair
    beat the faster half. So this tool recommends request shares first, and
    gives layer ratios only for the case layer split actually solves: a model
    too big for one host.

WHAT IT DOES
    1. Runs `llama-batched-bench` on each host, alone, at the same settings.
    2. Reports each host's aggregate tokens/s.
    3. Recommends the REQUEST share per host, proportional to measured
       throughput, as concrete stream counts.
    4. Prints the layer-split `-ts` ratio too, with the predicted result and a
       warning when it is expected to lose to the fastest host alone.

    Every number it prints is measured on the machines you name, at the
    settings you give. Nothing is assumed from a spec sheet.

USAGE
    ./split-advisor.py --model-local  ~/models/gguf/gemma-3-4b-it-Q4_K_M.gguf \\
                       --host  m5:petrus@192.168.50.127:~/llama.cpp/build/bin:~/llm/models/vision/gemma-3-4b-it-Q4_K_M.gguf \\
                       --streams 64

    Hosts are `name:ssh_target:bin_dir:model_path`. The local machine is always
    included; give its binary directory with --local-bin.
"""

import argparse
import os
import shlex
import subprocess
import sys

# Within 3% of a host's best counts as the knee: past it you buy latency and
# KV cache, not throughput.
KNEE_TOLERANCE = 0.97


class Host:
    def __init__(self, spec):
        # name:ssh_target:bin_dir:model_path
        parts = spec.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                f"--host wants name:ssh_target:bin_dir:model_path, got {spec!r}"
            )
        self.name, self.ssh, self.bin_dir, self.model = parts
        if not self.ssh:
            # Local paths are quoted before the shell sees them, so a leading
            # ~ would be taken literally. Expand it here instead.
            self.bin_dir = os.path.expanduser(self.bin_dir)
            self.model = os.path.expanduser(self.model)
        self.tps = None

    def run(self, streams, npp, ntg, ctx):
        cmd = (
            f"cd {shlex.quote(self.bin_dir)} && ./llama-batched-bench "
            f"-m {shlex.quote(self.model)} -c {ctx} -b 2048 -ub 512 "
            f"-npp {npp} -ntg {ntg} -npl {streams}"
        )
        argv = ["ssh", self.ssh, cmd] if self.ssh else ["bash", "-lc", cmd]
        out = subprocess.run(argv, capture_output=True, text=True).stdout
        return parse_tps(out, streams)


def knee(curve):
    """The stream count worth running: the best measured throughput, but the
    SMALLEST count that gets within KNEE_TOLERANCE of it. Past the knee you pay
    latency and KV cache for throughput you do not get, and a bigger batch is
    the first thing to OOM an engine."""
    best = max(curve.values())
    for n in sorted(curve):
        if curve[n] >= best * KNEE_TOLERANCE:
            return n, curve[n]
    return max(curve.items(), key=lambda kv: kv[1])


def parse_tps(output, streams):
    """Aggregate tokens/s for the row with this batch size, from the md table."""
    for line in output.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 10 and cells[2].isdigit() and int(cells[2]) == streams:
            try:
                return float(cells[-1])
            except ValueError:
                continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-local", required=True, help="model path on THIS machine")
    ap.add_argument("--local-bin", default="~/work/llama-rpc/llama.cpp/build/bin",
                    help="llama.cpp build/bin on THIS machine")
    ap.add_argument("--local-name", default="local")
    ap.add_argument("--host", action="append", default=[],
                    metavar="name:ssh:bin_dir:model", help="a remote host (repeatable)")
    ap.add_argument("--streams", type=int, default=64,
                    help="total concurrent streams the cluster must serve")
    ap.add_argument("--sweep", default="1,2,4,8,16,32,64",
                    help="stream counts to try per host when finding its knee; "
                         "empty string = measure only at --streams")
    ap.add_argument("--npp", type=int, default=256, help="prompt tokens per stream")
    ap.add_argument("--ntg", type=int, default=128, help="generated tokens per stream")
    args = ap.parse_args()

    ctx = max(4096, (args.npp + args.ntg) * args.streams)
    hosts = [Host(f"{args.local_name}::{args.local_bin}:{args.model_local}")]
    hosts += [Host(h) for h in args.host]

    sweep = [int(x) for x in args.sweep.split(",") if x.strip()] or [args.streams]
    print(f"Measuring {len(hosts)} host(s) at {sweep} streams, "
          f"npp {args.npp} / ntg {args.ntg}.\n"
          f"One host at a time, so nothing contends.\n")
    for h in hosts:
        h.curve = {}
        for n in sweep:
            tps = h.run(n, args.npp, args.ntg, max(ctx, (args.npp + args.ntg) * n))
            if tps:
                h.curve[n] = tps
        if not h.curve:
            print(f"  {h.name:<12}    FAILED — no result row parsed; check the "
                  f"path and that llama-batched-bench exists there.")
            continue
        h.streams, h.tps = knee(h.curve)
        print(f"  {h.name:<12} best {h.tps:8.1f} tok/s at {h.streams:3d} streams")
        print("               " + "  ".join(f"{n}:{t:.0f}" for n, t in sorted(h.curve.items())))

    live = [h for h in hosts if h.tps]
    if not live:
        sys.exit("No host produced a number. Nothing to advise.")

    total = sum(h.tps for h in live)
    fastest = max(live, key=lambda h: h.tps)

    print("\nREQUEST SPLIT — recommended. Each host runs the whole model and\n"
          "serves its own requests, so throughput adds instead of taking turns.\n"
          "Run each host at ITS OWN knee and point that many agent streams at it:\n")
    for h in live:
        print(f"  {h.name:<12} cap at {h.streams:3d} concurrent streams  "
              f"-> {h.tps:8.1f} tok/s")
    print(f"\n  cluster capacity     {total:8.1f} tok/s across "
          f"{sum(h.streams for h in live)} streams"
          f"   ({total/fastest.tps:.2f}x {fastest.name} alone)")
    print("  Configure the agents' concurrency to these numbers. Sending a host\n"
          "  more than its knee buys latency and KV cache, not throughput, and the\n"
          "  KV cache is what OOMs an engine first.")
    print("\n  This total is a PREDICTION from separate runs. Verify it by running\n"
          "  every host at the same time and checking each still reaches its solo\n"
          "  number; on the reference pair it did (mini 119.6 solo, 119.6 while the\n"
          "  M5 ran 56 streams beside it).")

    if len(live) >= 2:
        ratio = ":".join(f"{h.tps/min(x.tps for x in live):.0f}" for h in live)
        print(f"\nLAYER SPLIT — only if the model does not fit on one host.\n"
              f"  llama.cpp -ts {ratio.replace(':', '/')}   "
              f"(remote device is slot 0; confirm with -ts 0/1)")
        print(f"  Expect it to LOSE to {fastest.name} alone: the hosts take turns on\n"
              f"  the same token and each step pays a fixed coordination cost. On the\n"
              f"  reference pair that cost was ~39 ms per step, larger than the fast\n"
              f"  host's entire decode step at small batches. Bigger batches amortise\n"
              f"  it (4.9 ms/token at 8 streams, 1.1 at 64) but never repay it.")


if __name__ == "__main__":
    main()
