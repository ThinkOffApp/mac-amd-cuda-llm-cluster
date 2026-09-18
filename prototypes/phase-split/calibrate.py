"""Decide whether to split THIS pair, from 30 seconds of calibration.

The rule is only worth shipping if it answers before the expensive run. Summing a
run's own measured parts and recovering that run's total proves nothing -- it is an
accounting identity. What has to be shown is that the two extrapolated terms are
accurate from cheap measurements:

    slow-machine prefill at the target length   fitted from two short prefills
    KV bytes at the target length               const + per_token * n

These are bounded observations on ONE pair, one model, one build, one -c. They are
not validation of a universal install-time rule, and the full-vocabulary correctness
gate is still outstanding.

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
    # Bandwidth is not a constant of the link. Measured on @grok's pair, the same
    # copy path gave 439, 500 and 525 MB/s for 295, 358 and 694 MB -- startup cost
    # amortising. Taking the figure from the smallest file and applying it to a
    # larger one over-charges the transfer and makes the split look worse than it
    # is, by 0.26 s at 8k tokens.
    ap.add_argument("--link-measured-at-mb", type=float, required=True,
                    help="size of the file the --link-mbps timing was taken on, in MB")
    ap.add_argument("--link-tolerance", type=float, default=2.0,
                    help="how far from that size the figure may be applied (default 2x)")
    # @claudeMB published a 52,440-token crossover fitted from data ending at 3,072,
    # on servers that could not have run a prompt that long. A linear fit evaluated
    # 17x outside its range is not a result, and a tool that hands it over without
    # comment is how it becomes one. Refuse by default.
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="evaluate beyond the longest calibrated length anyway; "
                         "the answer is then a model, not a measurement")
    # Length is not the only range a calibration has. @codexmb: refuse outside the
    # calibrated model, build and context as well -- a curve measured on one build
    # at one -c is not evidence about another.
    ap.add_argument("--calibrated-on", required=True,
                    help="identity the calibration was taken under, e.g. "
                         "'sha3f227079/build434ddbbc/c4096'")
    ap.add_argument("--target-on", required=True,
                    help="identity this prediction is for; must match --calibrated-on")
    # Where the prefill timings came from is part of what they mean. @grok posted
    # llama-bench and llama-server numbers for the same pair at the same length:
    # each machine's ABSOLUTE prefill agreed to within 7%, and their DIFFERENCE
    # disagreed by 2.07x, because the difference is small and the errors point
    # opposite ways. This rule uses only the difference, so it inherits the worst
    # of both -- and on that pair the two sources fall on opposite sides of the
    # split/do-not-split line.
    # The honest default the room agreed on: a handoff between different backends
    # is not numerically equivalent to the un-split path. @codexmb's full-vocabulary
    # gate fails across CUDA->Metal (max 9.46, 179,629 logits out of tolerance) while
    # his same-host control is exactly zero. Nobody has shown this changes a
    # user-visible answer -- and nobody has shown it does not -- so the tool refuses
    # and says what would settle it, rather than quietly recommending a split whose
    # output differs from not splitting.
    ap.add_argument("--producer-backend", required=True,
                    help="backend the producer runs, e.g. CUDA, Metal, Vulkan")
    ap.add_argument("--consumer-backend", required=True,
                    help="backend the consumer runs; a mismatch is refused")
    ap.add_argument("--accept-numerical-divergence", action="store_true",
                    help="proceed across a backend boundary anyway. A measured flip rate "
                         "is an INPUT to that decision, not an authorization: it does not "
                         "override the failed numerical gate")
    ap.add_argument("--timing-source", required=True, choices=["llama-server", "llama-bench"],
                    help="where the prefill timings came from. llama-bench is REFUSED: "
                         "it is not the path being split")
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

    if (a.producer_backend.strip().lower() != a.consumer_backend.strip().lower()
            and not a.accept_numerical_divergence):
        raise SystemExit(
            f"REFUSING: producer is {a.producer_backend} and consumer is "
            f"{a.consumer_backend}.\n"
            "Across a backend boundary the handoff is NOT numerically equivalent to the\n"
            "un-split path: a full-vocabulary check across CUDA->Metal fails at\n"
            "atol 0.1 / rtol 0.01, max abs error 9.46 with 179,629 logits out of\n"
            "tolerance at one step, while the same-host control is exactly 0.000000.\n"
            "Sixteen greedy tokens still matched, so this may never change an answer --\n"
            "but nobody has measured that. What would settle it is a FLIP RATE: a few\n"
            "hundred generations under your real sampling settings, split against\n"
            "un-split, counting how often the emitted text differs.\n"
            "A low flip rate does NOT by itself authorize this and does not override the\n"
            "failed gate -- it is one input to a decision that stays a human's.")

    if a.timing_source != "llama-server":
        raise SystemExit(
            "REFUSING: timings from llama-bench.\n"
            "Measured on one real pair, llama-bench and llama-server agreed within 7% on\n"
            "each machine's absolute prefill and disagreed by 2.07x on their DIFFERENCE\n"
            "(0.702 s against 0.339 s at 2,111 tokens). This rule uses only the\n"
            "difference. At 2,111 tokens BOTH still say do-not-split (-0.205 s against\n"
            "-0.568 s), so they agreed on the verdict there -- but a 2x error in the only\n"
            "term that matters moves the crossover, and llama-bench puts it just ~17%\n"
            "above the length actually tested.\n"
            "Time the prefill with llama-server on the real prompt -- that is the path\n"
            "being split.")

    if a.calibrated_on != a.target_on:
        raise SystemExit(
            f"REFUSING: calibrated on '{a.calibrated_on}' but asked about "
            f"'{a.target_on}'.\n"
            "Model weights, llama.cpp build and -c each change the curves this tool\n"
            "fits. A calibration is evidence about the configuration it was taken\n"
            "under and about no other. Re-calibrate on the target.")

    n = a.tokens
    kv_mb = (a.kv_const_mb * 1e6 + a.kv_per_token * n) / 1e6
    lo, hi = a.link_measured_at_mb / a.link_tolerance, a.link_measured_at_mb * a.link_tolerance
    if not lo <= kv_mb <= hi:
        raise SystemExit(
            f"REFUSING: --link-mbps was timed on {a.link_measured_at_mb:,.0f} MB but this "
            f"prompt moves {kv_mb:,.0f} MB.\n"
            "Effective bandwidth is not a constant of the link -- on one measured pair the "
            "same\npath gave 439, 500 and 525 MB/s for 295, 358 and 694 MB as startup cost "
            "amortised.\nApplying a small-file figure to a large transfer over-charges it "
            "and biases the\nanswer toward do-not-split. Re-time the copy at roughly the "
            "size you will move.")

    calibrated_to = min(max(a.slow_cal[0], a.slow_cal[2]),
                        max(a.fast_cal[0], a.fast_cal[2]))
    if n > calibrated_to and not a.allow_extrapolation:
        raise SystemExit(
            f"REFUSING: asked for {n:,} tokens, calibrated only to {calibrated_to:,.0f}.\n"
            f"That is a {n/calibrated_to:.1f}x extrapolation of a linear fit.\n"
            "Out here the fit supplies NEITHER A VALUE NOR A BOUND, not even a sign.\n"
            "Each machine's prefill curve has a quadratic term, but the DIFFERENCE of two\n"
            "such curves need not grow superlinearly and need not stay positive; kernels,\n"
            "hybrid attention, context settings and memory pressure move either curve\n"
            "independently. A near-zero denominator likewise does not prove 'never pays'.\n"
            "The run may also be impossible at the configured context.\n"
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

    print(f"[{a.target_on}]")
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
