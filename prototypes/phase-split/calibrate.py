"""Decide whether to split THIS pair, from 30 seconds of calibration.

The rule is only worth shipping if it answers before the expensive run. Summing a
run's own measured parts and recovering that run's total proves nothing -- it is an
accounting identity. What has to be shown is that the two extrapolated terms are
accurate from cheap measurements:

    slow-machine prefill at the target length   fitted from two short prefills
    KV bytes at the target length               const + per_token * n

Checked against the Mini at 2107 tokens, calibrating on 512 and 1024 only:

    slow prefill    predicted 38,242 ms   actual 38,153 ms    0.23%
    KV bytes        predicted 295,002,313  actual 295,038,132  -0.01%

Both the constant and the per-token term are model-specific and must be measured
per model. The numbers above are Qwen3.8-27B-UD-Q4_K_XL.
"""
import argparse


def fit_linear(points):
    (n0, t0), (n1, t1) = points
    slope = (t1 - t0) / (n1 - n0)
    return slope, t0 - n0 * slope


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, required=True, help="target prompt length")
    ap.add_argument("--slow-cal", nargs=4, type=float, required=True,
                    metavar=("N0", "MS0", "N1", "MS1"),
                    help="two short prefills on the SLOW machine")
    ap.add_argument("--fast-cal", nargs=4, type=float, required=True,
                    metavar=("N0", "MS0", "N1", "MS1"),
                    help="two short prefills on the FAST machine")
    ap.add_argument("--link-mbps", type=float, required=True,
                    help="MEASURED MB/s from one timed copy, not the nominal line rate")
    # @claudeMB published a 52,440-token crossover fitted from data ending at 3,072,
    # on servers that could not have run a prompt that long. A linear fit evaluated
    # 17x outside its range is not a result, and a tool that hands it over without
    # comment is how it becomes one. Refuse by default.
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="evaluate beyond the longest calibrated length anyway; "
                         "the answer is then a model, not a measurement")
    # No defaults on any of these. Today's recurring failure was a constant
    # carried across a boundary where it did not hold: a per-token KV rate that
    # hid a fixed block, and a ceiling from one pair quoted at another. A default
    # here would be my machine's number silently becoming someone else's.
    ap.add_argument("--kv-const-mb", type=float, required=True,
                    help="fixed part of the slot file, MB -- PER MODEL")
    ap.add_argument("--kv-per-token", type=float, required=True,
                    help="marginal bytes per token -- PER MODEL")
    ap.add_argument("--save-ms", type=float, required=True)
    ap.add_argument("--restore-ms", type=float, required=True)
    ap.add_argument("--token-n-ms", type=float, required=True,
                    help="decoder's prefill of token N -- the N-1 boundary. "
                         "Constant across links, NOT across machines.")
    a = ap.parse_args()

    n = a.tokens
    calibrated_to = min(max(a.slow_cal[0], a.slow_cal[2]),
                        max(a.fast_cal[0], a.fast_cal[2]))
    if n > calibrated_to and not a.allow_extrapolation:
        raise SystemExit(
            f"REFUSING: asked for {n:,} tokens, calibrated only to {calibrated_to:,.0f}.\n"
            f"That is a {n/calibrated_to:.1f}x extrapolation of a linear fit.\n"
            "Prefill is not linear -- attention is quadratic -- so the prefill GAP grows\n"
            "faster than this model assumes while KV grows exactly linearly. A crossover\n"
            "computed out here is an UPPER BOUND at best, and the run may not even be\n"
            "possible at the configured context.\n"
            "Calibrate at a length near the target, or pass --allow-extrapolation and\n"
            "report the answer as a model rather than a measurement.")
    s_slope, s_icept = fit_linear([(a.slow_cal[0], a.slow_cal[1]), (a.slow_cal[2], a.slow_cal[3])])
    f_slope, f_icept = fit_linear([(a.fast_cal[0], a.fast_cal[1]), (a.fast_cal[2], a.fast_cal[3])])
    slow = s_slope * n + s_icept
    fast = f_slope * n + f_icept
    kv = a.kv_const_mb * 1e6 + a.kv_per_token * n
    transfer = kv / 1e6 / a.link_mbps * 1000
    fixed = fast + a.save_ms + a.restore_ms + a.token_n_ms
    split = fixed + transfer

    print(f"target {n} tokens (calibrated to {calibrated_to:,.0f})"
          + (f"  ** {n/calibrated_to:.1f}x EXTRAPOLATION -- this is a model, not a measurement **"
             if n > calibrated_to else "") + "\n")
    print(f"  slow machine prefill      {slow:>10,.0f} ms   ({n/(slow/1000):.0f} tok/s)")
    print(f"  fast machine prefill      {fast:>10,.0f} ms   ({n/(fast/1000):.0f} tok/s)")
    print(f"  prefill ratio             {slow/fast:>10.2f}x")
    print(f"  KV bytes                  {kv:>10,.0f} B")
    print(f"  transfer at {a.link_mbps:.0f} MB/s     {transfer:>10,.0f} ms")
    print(f"  save+restore+token N      {a.save_ms+a.restore_ms+a.token_n_ms:>10,.0f} ms")
    print(f"  ---")
    print(f"  split TTFT                {split:>10,.0f} ms")
    print(f"  slow machine alone        {slow:>10,.0f} ms")
    print(f"  speedup                   {slow/split:>10.2f}x")

    budget = slow - fixed
    if budget <= 0:
        print("\n  DO NOT SPLIT. The fixed cost alone exceeds the slow machine's prefill;")
        print("  no link speed can make this pair pay.")
    else:
        need = kv / 1e6 / (budget / 1000)
        ceiling = slow / fixed
        print(f"\n  break-even link           {need:>10,.0f} MB/s = {need*8/1000:.1f} Gbit/s")
        print(f"  ceiling on an infinite link {ceiling:>8.2f}x")
        print("\n  " + ("SPLIT." if slow / split > 1.0 else "DO NOT SPLIT.")
              + " This is a property of THIS PAIR, not of the technique --"
              "\n  a ceiling computed here does not transfer to different hardware.")


if __name__ == "__main__":
    main()
