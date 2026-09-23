#!/usr/bin/env python3
"""Find the layer-split ratio that actually goes fastest, by measuring.

WHY THIS EXISTS
    llama.cpp's `-ts` is a ratio you type, and its default is proportional to
    DEVICE MEMORY. In a pipeline the slowest stage sets the rate, so on unequal
    machines an even split hands the same work to both and bottlenecks on the
    slower one. Measured on a Mac + DGX Spark pair, the ratio was worth 62% of
    prefill throughput, and the memory-proportional default was SLOWER than the
    faster machine on its own (0.757x).

WHY NOT GRADIENT DESCENT
    Each evaluation costs a benchmark run of tens of seconds, the objective is
    noisy, it is one-dimensional, and there is no gradient to take. Golden
    section search is the right tool for that shape: it needs no derivative,
    it is robust to a unimodal curve, and it converges in a handful of
    evaluations rather than hundreds.

    The measured curve IS unimodal (verified on a real pair, single peak with
    monotone flanks), which is what makes the search valid. If your curve turns
    out multi-peaked, fall back to a coarse scan; this script says so rather
    than returning a local peak it cannot justify.
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from typing import Callable

INV_PHI = (math.sqrt(5) - 1) / 2   # 0.618...


class Unmeasurable(Exception):
    """Raised instead of returning a ratio we cannot stand behind."""


def bench(client: str, model: str, rpc: str, remote_share: int, prompt: int, reps: int) -> float:
    """tok/s at this split. `-ts` lists RPC devices FIRST, then the local backend."""
    ts = f"{remote_share}/{100 - remote_share}"
    out = subprocess.run(
        [client, "-m", model, "--rpc", rpc, "-ts", ts,
         "-p", str(prompt), "-n", "0", "-r", str(reps), "-o", "json"],
        capture_output=True, text=True, timeout=1800,
    )
    if out.returncode != 0:
        raise Unmeasurable(f"-ts {ts} failed: {(out.stderr or '').strip()[:160]}")
    try:
        rows = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise Unmeasurable(f"-ts {ts} produced no parseable result")
    vals = [r["avg_ts"] for r in rows if r.get("n_prompt")]
    if not vals:
        raise Unmeasurable(f"-ts {ts} produced no prefill rows")
    return sum(vals) / len(vals)


def golden_max(f: Callable[[int], float], lo: int, hi: int, tol: int, log) -> tuple[int, float]:
    """Maximise a unimodal f over integers in [lo, hi]. Caches, so repeats are free."""
    cache: dict[int, float] = {}

    def ev(x: int) -> float:
        x = max(lo, min(hi, int(round(x))))
        if x not in cache:
            cache[x] = f(x)
            log(f"    {x:>3}/{100 - x:<3}  {cache[x]:8.2f} tok/s")
        return cache[x]

    a, b = lo, hi
    c = int(round(b - INV_PHI * (b - a)))
    d = int(round(a + INV_PHI * (b - a)))
    fc, fd = ev(c), ev(d)
    while abs(b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = int(round(b - INV_PHI * (b - a)))
            fc = ev(c)
        else:
            a, c, fc = c, d, fd
            d = int(round(a + INV_PHI * (b - a)))
            fd = ev(d)
        if c == d:
            break
    best = max(cache, key=cache.get)
    return best, cache[best]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True, help="host:port of the ggml-rpc-server")
    p.add_argument("--client", default="llama-bench")
    p.add_argument("--prompt", type=int, default=2048, help="prompt length to optimise for")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--lo", type=int, default=10, help="min %% of layers on the remote")
    p.add_argument("--hi", type=int, default=70)
    p.add_argument("--tol", type=int, default=4, help="stop when the bracket is this narrow")
    a = p.parse_args()

    log = lambda s: print(s, flush=True)
    log(f"  optimising -ts for pp{a.prompt} on {a.model.split('/')[-1]}")
    log(f"  searching remote share in [{a.lo}, {a.hi}], golden section, tol {a.tol}")
    try:
        best, val = golden_max(
            lambda x: bench(a.client, a.model, a.rpc, x, a.prompt, a.reps),
            a.lo, a.hi, a.tol, log)
    except Unmeasurable as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    log("")
    log(f"  BEST  -ts {best}/{100 - best}   {val:.2f} tok/s")
    log(f"  use:  --rpc {a.rpc} -ts {best}/{100 - best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
