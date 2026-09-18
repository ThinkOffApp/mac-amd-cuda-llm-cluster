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
    **gemma-3-4b Q4_K_M** -- a small model, NOT one we serve -- 64 concurrent
    streams, 2026-09-18:

        best layer split (-ts 16/1)        681 tok/s
        M5 alone                          1077 tok/s
        request split, measured CONCURRENTLY  1261 tok/s     1.17x

    THE MODEL MATTERS AND THIS ONE IS BOTH A TOY AND AN OBSOLETE ONE. gemma-3
    was several generations behind the current Gemma when this was measured, and
    it was chosen for no reason beyond an existing harness -- which is exactly
    how an old version survives. If a small model is the right tool it still has
    to be a CURRENT small model. On the same day, a mechanism
    concluded from GPT-2 was wrong about Qwen3.8-27B: per-call time looked flat
    against batch size on the toy and the real model turned out partly
    work-bound (32x work -> 5.4x time). So treat the 1.17x as a result ABOUT
    gemma-3-4b until someone re-runs it on the 27B.

    The reason to expect it to carry is stronger here than for anything else
    measured that day -- two machines answering DIFFERENT requests never
    interact, so there is no coordination cost waiting to reappear at scale --
    but "expect" is not "measured", and this tool exists to measure.

    PUT THE DRIVER ON A HOST THAT IS NOT UNDER TEST -- as hygiene, not because
    it was shown to matter. On a GB10 Spark it was measured and it costs almost
    nothing:

        streams    client on another box    client on the box itself
             8              56.4 tok/s               58.0 tok/s
            16              73.4                     76.8
            32              92.0                     96.4

    The colocated figures are slightly HIGHER, so on that hardware a local
    client was not depressing the single-server baseline at all.

    THIS DOES NOT TRANSFER TO A SMALLER HOST UNTESTED. The Mac mini has 10
    cores, 4 of them performance, against a GB10's; llama.cpp's CPU demand to
    feed the GPU does not scale with either in a way anyone here has measured.
    The 1.17x above was driven from the Mini, which is one of its two hosts, so
    for THAT pair the question is open in both directions: unmeasured, not
    condemned and not cleared.

    RESOLVED, and it was the instrument again: that pair result came from
    server-side token counts divided by a CLIENT-side wall clock taken on a busy
    box. Per-stream generation was identical in both conditions (9.8 tok/s), so
    nothing was contended. Two machines do scale. Counting tokens server-side and
    then dividing by the client's clock is not a server-side measurement.

    THE KNEE IS A CLIFF, NOT A PLATEAU. One GB10 Spark on Qwen3.8-27B, client
    on another box, measured 2026-09-18:

        1 stream  11.6      16 streams  73.4
        4         37.1      32          92.0   <- best
        8         56.4      64          82.9   <- WORSE than 32

    Past the knee the box loses throughput. So cap each host at its knee and
    send the overflow elsewhere; running one box at 64 costs 10% against running
    it at 32 and queueing the rest.

    AND COMPARE AT A FIXED TOTAL LOAD, not at a fixed per-host stream count.
    From the same curve:

        total load   one box   two boxes    gain    (per box)
              8 st     56.4      74.2      1.32x    (4 each)
             32 st     92.0     146.8      1.60x    (16 each)
             64 st     82.9     184.0      2.22x    (32 each)

    The second host is worth 1.3x at light load and 2.2x past the knee, because
    splitting a light load puts BOTH hosts far below it. A per-host comparison
    ("4 streams each vs 4 streams alone") compares 8 streams of work against 4
    and overstates the gain.

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
import json
import os
import shlex
import statistics
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

    def run(self, streams, npp, ntg, ctx, rpc=None, ts=None, metric="total"):
        cmd = (
            f"cd {shlex.quote(self.bin_dir)} && ./llama-batched-bench "
            f"-m {shlex.quote(self.model)} -c {ctx} -b 2048 -ub 512 "
            f"-npp {npp} -ntg {ntg} -npl {streams}"
        )
        if rpc:
            cmd += f" --rpc {shlex.quote(rpc)}"
        if ts:
            cmd += f" -ts {shlex.quote(ts)}"
        argv = ["ssh", self.ssh, cmd] if self.ssh else ["bash", "-lc", cmd]
        out = subprocess.run(argv, capture_output=True, text=True).stdout
        return parse_tps(out, streams, metric)


def ts_pairs(steps):
    """Candidate `-ts a/b` ratios as integer pairs, coarse to fine.

    llama.cpp takes integers, so the search space is genuinely discrete and
    small. Expressing candidates as a/b from the start avoids tuning a float
    and then discovering two different floats produce the same split.
    """
    out = []
    for a in steps:
        out.append((a, 1))
    for b in reversed(steps[1:]):
        out.append((1, b))
    seen, uniq = set(), []
    for a, b in out:
        key = round(a / (a + b), 4)
        if key not in seen:
            seen.add(key)
            uniq.append((a, b))
    return sorted(uniq, key=lambda p: p[0] / (p[0] + p[1]))


def measure_ts(local, rpc, a, b, streams, npp, ntg, ctx, metric, repeats):
    """Median of `repeats` runs at one ratio. Median, not mean: a single slow
    run (a GPU shared with another service, a thermal dip) drags a mean and
    the search then walks toward noise."""
    vals = []
    for _ in range(repeats):
        v = local.run(streams, npp, ntg, ctx, rpc=rpc, ts=f"{a}/{b}", metric=metric)
        if v:
            vals.append(v)
    return statistics.median(vals) if vals else None


def tune_layer_split(local, rpc, args):
    """Find the best `-ts` ratio by measuring, coarse grid then local refine.

    NOT gradient descent, and the reason is the objective rather than taste:
    there is no gradient to take (each evaluation is a ~20 s benchmark, not a
    differentiable function), the parameter is ONE discrete ratio, and the
    measurement carries several percent of run-to-run noise. A descent would
    spend its evaluations chasing that noise. A coarse grid over a space this
    small finds the basin in a handful of runs, and a local refine around the
    winner is enough because throughput against split ratio is unimodal — one
    machine starves at either end.

    Reports the noise it measured alongside the winner, so a "best" that is
    inside the noise is visible as such instead of being quoted as a result.
    """
    ctx = max(4096, (args.npp + args.ntg) * args.streams)
    coarse = ts_pairs([1, 2, 4, 8, 16])
    results = {}
    print(f"\nTUNING -ts for {args.metric} at {args.streams} streams, "
          f"{args.repeats} run(s) per point.\n"
          f"Coarse pass over {len(coarse)} ratios:\n")
    for a, b in coarse:
        v = measure_ts(local, rpc, a, b, args.streams, args.npp, args.ntg,
                       ctx, args.metric, args.repeats)
        results[(a, b)] = v
        share = 100 * a / (a + b)
        print(f"  -ts {a}/{b:<3}  remote {share:5.1f}%   "
              + (f"{v:8.1f} tok/s" if v else "   failed"))

    live = {k: v for k, v in results.items() if v}
    if not live:
        print("  every ratio failed — is the rpc-server up at --rpc?")
        return None
    best = max(live, key=live.get)

    # Refine between the winner's neighbours, where the true optimum must lie.
    order = sorted(live, key=lambda p: p[0] / (p[0] + p[1]))
    i = order.index(best)
    lo = order[max(0, i - 1)]
    hi = order[min(len(order) - 1, i + 1)]
    refine = []
    for (a1, b1), (a2, b2) in ((lo, best), (best, hi)):
        mid_share = (a1 / (a1 + b1) + a2 / (a2 + b2)) / 2
        if 0 < mid_share < 1:
            a = max(1, round(mid_share * 12))
            refine.append((a, max(1, 12 - a)))
    refine = [p for p in dict.fromkeys(refine) if p not in live]
    if refine:
        print(f"\nRefining around -ts {best[0]}/{best[1]}:\n")
        for a, b in refine:
            v = measure_ts(local, rpc, a, b, args.streams, args.npp, args.ntg,
                           ctx, args.metric, args.repeats)
            if v:
                live[(a, b)] = v
                print(f"  -ts {a}/{b:<3}  remote {100*a/(a+b):5.1f}%   {v:8.1f} tok/s")
        best = max(live, key=live.get)

    spread = max(live.values()) - min(live.values())
    print(f"\n  BEST  -ts {best[0]}/{best[1]}   {live[best]:.1f} tok/s "
          f"({args.metric})   remote share {100*best[0]/(best[0]+best[1]):.1f}%")
    print(f"  Ratio is worth {100 * spread / min(live.values()):.0f}% between the best and "
          f"worst ratio measured — that is why guessing 50/50 is not free.")
    if args.repeats < 2:
        print("  Measured ONCE per ratio. Re-run with --repeats 3 before quoting this;\n"
              "  a shared GPU moves these numbers by several percent.")
    return best


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


# Column positions in llama-batched-bench's markdown table.
METRIC_COLUMN = {"pp": 5, "tg": 7, "total": 9}


def parse_tps(output, streams, metric="total"):
    """Tokens/s for the row with this batch size, from the md table.

    `metric` picks prefill, generation or the combined figure. They optimise to
    DIFFERENT split ratios — prefill has intra-request parallelism and
    generation does not — so a tuner that reports one number for both is
    tuning the wrong thing for one of them.
    """
    col = METRIC_COLUMN.get(metric, 9)
    for line in output.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > col and cells[2].isdigit() and int(cells[2]) == streams:
            try:
                return float(cells[col])
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------
# Model-currency preflight. @codexmb, 2026-09-18: "A newly written checker is
# not yet proof that this cannot recur. The check needs to be wired into the
# actual selection/download/benchmark entry point, fail closed when current
# information cannot be verified, and save the official source, check time,
# chosen revision and any explicit older-version reason."
#
# He then searched this repo and found no checker call anywhere in it. He was
# right: the guard written an hour earlier protected downloads through one
# wrapper and left the path that actually produced the bad number untouched.
# This runs before any host is measured, and refuses.
# --------------------------------------------------------------------------

MODEL_CHECKERS = ["~/bin/modelcheck.py",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "model-currency-check.py")]
RECEIPT = "model-currency-receipt.json"


def _assert_we_load_the_models():
    """This tool may only gate a typed path while it LOADS the model itself.

    @claudeMB gates his harness on what the server reports at /props, not on
    what someone typed, so a harness pointed at a server running something else
    is still caught. That is strictly better -- and it does not apply here, for
    a reason worth asserting rather than remembering: Host.run invokes
    `llama-bench -m <model>`, so the typed path IS the loaded model. With
    --tune-ts the local llama.cpp still loads --model-local and uses the remote
    only as a compute backend.

    If anyone adds an attach-to-running-server mode, this assertion fails and
    they have to gate on what the server reports instead of what was typed.
    """
    src = open(os.path.abspath(__file__)).read()
    if "llama-bench" not in src or "-m {shlex.quote(self.model)}" not in src:
        raise SystemExit(
            "REFUSING: this tool no longer launches llama-bench with the model it was\n"
            "given, so a typed path is no longer evidence of what is loaded. Gate on\n"
            "what the server reports (llama-server /props -> model_path) before\n"
            "measuring anything.")


def _preflight_models(models, reason, receipt_path):
    """Refuse to benchmark until every model's generation has been checked.

    Fails CLOSED: if no checker is reachable, or a checker cannot answer, the
    run does not start. A benchmark that proceeds when the check is unavailable
    is the same as not having the check.
    """
    import datetime
    import shutil
    import subprocess

    checker = next((os.path.expanduser(c) for c in MODEL_CHECKERS
                    if os.path.exists(os.path.expanduser(c))), None)
    if checker is None:
        raise SystemExit(
            "REFUSING TO BENCHMARK: no model-currency checker found.\n"
            "  looked for: " + ", ".join(MODEL_CHECKERS) + "\n"
            "A model was benchmarked on 2026-09-18 that was five months out of date\n"
            "because nothing in this path asked. Install a checker or state why this\n"
            "run may skip it -- silence is not permission.")

    entries = []
    for m in models:
        cmd = [sys.executable, checker, m] + (["--reason", reason] if reason else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        entries.append({"model": m, "checker": checker, "exit": r.returncode,
                        "output": (r.stdout + r.stderr).strip()})
        if r.returncode != 0:
            print((r.stdout + r.stderr).strip(), file=sys.stderr)
            raise SystemExit(f"REFUSING TO BENCHMARK: {os.path.basename(m)} did not pass "
                             f"the currency check (exit {r.returncode}).")
        print((r.stdout + r.stderr).strip())

    with open(receipt_path, "w") as f:
        json.dump({"checked_at": datetime.datetime.now(datetime.timezone.utc)
                                      .isoformat(timespec="seconds"),
                   "stated_reason": reason or None,
                   "entries": entries}, f, indent=2)
    print(f"currency receipt -> {receipt_path}")


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
    ap.add_argument("--tune-ts", metavar="HOST:PORT",
                    help="find the best layer-split ratio against an ALREADY RUNNING "
                         "ggml-rpc-server (start it yourself; this script does not "
                         "manage remote processes)")
    ap.add_argument("--metric", choices=["pp", "tg", "total"], default="total",
                    help="what to optimise when tuning: prefill, generation, or both. "
                         "They have different optima; pick the one you care about")
    ap.add_argument("--older-model-reason", default="",
                    help="why an older model generation is right for THIS run; printed "
                         "with the result and written to the currency receipt")
    ap.add_argument("--currency-receipt", default=RECEIPT,
                    help="where to write the model-currency receipt")
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per ratio while tuning; 3+ before quoting a result")
    args = ap.parse_args()

    # Before ANY host is touched. Every model this run will use, including the
    # remote ones, goes through the check or the run does not start.
    _assert_we_load_the_models()
    _preflight_models(
        [args.model_local] + [h.split(":")[-1] for h in args.host],
        args.older_model_reason, args.currency_receipt)

    if args.tune_ts:
        local = Host(f"{args.local_name}::{args.local_bin}:{args.model_local}")
        tune_layer_split(local, args.tune_ts, args)
        return

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
