"""How often does splitting the prefill actually change the answer a user sees?

THIS IS THE EXPERIMENT THAT DECIDES DEPLOYABILITY, and nothing run on 2026-09-18
answers it. What is known:

    same host, chunk boundaries matched      0.000000 logit error exactly
    across CUDA -> Metal, boundaries matched full-vocab gate FAILS, max 9.46,
                                             179,629 logits out of tolerance
    ...yet 16 greedy tokens matched anyway

A failing logit gate is not a user-visible defect and a passing greedy check is not
safety. The question a product owner actually has is: **under our real sampling
settings, how often does the split path emit different text than the un-split path?**
That is a rate, it has a confidence interval, and it is measurable.

AND THERE IS NO CHEAP SUBSTITUTE FOR IT. Total variation distance was proposed as a
ceiling this rate could be checked against; that was wrong. TV is the MINIMUM
disagreement over couplings, achieved only by the optimal one, so it is a LOWER bound.
This harness uses the same seed on both arms -- the shared-uniform coupling -- where
disagreement can run many times TV. A rate well above TV is what a CORRECT harness
produces here, and an earlier version of this docstring said the opposite.

WHAT THIS REFUSES TO DO
    Report a rate without controls. A zero flip rate is exactly what a broken harness
    produces -- comparing a string with itself, or silently re-prefilling so both
    sides run identical code. Three guards, all of which must hold:

      cache hit asserted      cache_n == N-1 and prompt_n == 1 on every split run,
                              so the restored state is what answered
      negative control        split vs split with the same seed must agree; if it
                              does not, the harness cannot detect agreement
      POSITIVE control        the comparison must be shown able to report a
                              difference. @codexmb: an altered PROMPT is not
                              guaranteed to change the output, so using one as the
                              positive control tests the model, not the harness, and
                              fails spuriously. The difference is injected into the
                              compared values directly instead.

    Today a control that returned a confident 0.000000 while comparing two failed
    requests survived long enough to be published. The positive control is the guard
    against that, and it is the one people leave out.

VALIDATE IT ON A KNOWN-GOOD PAIR FIRST
    @claudeMB measured bit-identical full-vocabulary log-probs for CUDA -> CUDA across
    two hosts with the chunk plan matched. Bit-identical log-probs means bit-identical
    probabilities, and every sampler transform -- temperature, top_k, top_p, penalties
    -- is a function of those. So that pair MUST return exactly 0/N flips at any
    settings.

    That is a positive control at the experiment level rather than the function level:
    if this harness reports a single flip there, the harness is broken and not the
    split. Run it there before pointing it at a pair whose answer is unknown, where a
    wrong number cannot be recognised as wrong.

    It also extends the depth cheaply. That measurement was four steps; --n-predict 64
    on the same pair is 64, and the failure mode for a product is a long answer.

USAGE
    Needs both servers up, the producer's slot file already copied to the consumer,
    and the same tokenised prompt on both sides.
"""

import argparse
import json
import math
import random
import subprocess
import sys

MARK = "\n__S__:"


class Failed(Exception):
    pass


def post(base, path, payload, timeout=900):
    o = subprocess.run(["curl", "-sS", "--max-time", str(timeout), "-X", "POST", base + path,
                        "-H", "Content-Type: application/json",
                        "-w", MARK + "%{http_code}", "-d", json.dumps(payload)],
                       capture_output=True, text=True)
    if o.returncode:
        raise Failed(f"{path}: curl exit {o.returncode}: {o.stderr.strip()[:200]}")
    body, _, st = o.stdout.rpartition(MARK)
    if not st.strip().isdigit():
        raise Failed(f"{path}: no status code")
    if not 200 <= int(st) < 300:
        raise Failed(f"{path}: HTTP {st.strip()}: {body.strip()[:300]}")
    d = json.loads(body)
    if isinstance(d, dict) and "error" in d:
        raise Failed(f"{path}: {d['error']}")
    return d


def sample(base, prompt, seed, args, slotfile=None, expect_cached=None):
    """One generation. With slotfile, restores first and proves the cache answered."""
    if slotfile:
        r = post(base, "/slots/0?action=restore", {"filename": slotfile})
        if r.get("n_restored") != expect_cached:
            raise Failed(f"{slotfile}: restored {r.get('n_restored')}, want {expect_cached}")
    else:
        post(base, "/slots/0?action=erase", {})
    c = post(base, "/completion", {
        "prompt": prompt, "n_predict": args.n_predict, "cache_prompt": bool(slotfile),
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "seed": seed})
    if slotfile:
        t = c.get("timings", {})
        if t.get("cache_n") != expect_cached or t.get("prompt_n") != 1:
            raise Failed(
                f"{slotfile}: cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')}, "
                f"want {expect_cached} and 1. The restore was not used -- both sides "
                "would be running identical un-split code and every flip rate below "
                "would be a zero produced by the harness.")
    text = c.get("content")
    if not text:
        raise Failed("empty content: the server is not really generating")
    return text


def differs(x, y):
    """The single comparison every flip decision goes through, so the positive
    control can exercise exactly the code the measurement uses."""
    return x != y


def wilson(k, n, z=1.96):
    """Interval that stays meaningful at k=0, where the normal approximation gives 0±0."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consumer", required=True, help="base URL of the decoding server")
    ap.add_argument("--request", required=True, nargs="+",
                    help="one or more JSON files, each holding a token array. Several "
                         "REPRESENTATIVE prompts, not one prompt with many seeds: "
                         "varying only the seed measures that prompt, not a workload")
    ap.add_argument("--slot", required=True, help="producer's slot file, on the consumer")
    ap.add_argument("--cached-tokens", type=int, required=True, help="N-1")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--temperature", type=float, required=True,
                    help="YOUR production value. A flip rate at temperature 0 does not "
                         "describe a product that samples")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    # A FLIP here means: the whole generation differs in any token. That is the
    # user-visible event. A per-token divergence rate is a different quantity and is
    # not what this reports.
    N = args.cached_tokens + 1
    prompts = []
    for f in args.request:
        toks = json.load(open(f))["prompt"]
        if len(toks) < N:
            sys.exit(f"{f}: {len(toks)} tokens, need at least {N}")
        prompts.append((f, toks[:N]))
    if len(prompts) == 1:
        print("WARNING: one prompt. The interval below describes THIS prompt under "
              "these settings,\n         not a workload. Pass several representative "
              "prompts.\n")
    prompt = prompts[0][1]

    print("controls first -- a rate without these is not evidence\n")
    try:
        a = sample(args.consumer, prompt, 12345, args, args.slot, args.cached_tokens)
        b = sample(args.consumer, prompt, 12345, args, args.slot, args.cached_tokens)
        if a != b:
            raise Failed("split vs split at one seed disagreed; the harness cannot "
                         "detect agreement and no flip rate below would mean anything")
        print(f"  negative control  same seed, both split      AGREE      ok")

        # Inject the difference into the compared values, not into the prompt.
        # An altered prompt may legitimately produce identical text, so it cannot
        # distinguish "the harness cannot see differences" from "the model agreed".
        if not differs(a, a + "\u2400"):
            raise Failed("POSITIVE CONTROL FAILED: the comparison reported two "
                         "provably different strings as equal. A zero flip rate "
                         "below would be an artefact of the comparison.")
        print(f"  positive control  injected difference         DETECTED   ok\n")
    except Failed as e:
        sys.exit(f"CONTROL FAILED -- no flip rate reported: {e}")

    flips = 0
    rng = random.Random(20260918)
    for i in range(args.runs):
        seed = rng.randrange(1, 2**31 - 1)
        prompt = prompts[i % len(prompts)][1]      # rotate prompts, not just seeds
        try:
            split = sample(args.consumer, prompt, seed, args, args.slot, args.cached_tokens)
            solo = sample(args.consumer, prompt, seed, args)
        except Failed as e:
            sys.exit(f"run {i}: {e}")
        if differs(split, solo):
            flips += 1
        if (i + 1) % 25 == 0:
            lo, hi = wilson(flips, i + 1)
            print(f"  {i+1:>4} runs   {flips:>4} flips   {flips/(i+1)*100:>6.2f}%   "
                  f"95% CI [{lo*100:.2f}, {hi*100:.2f}]")

    lo, hi = wilson(flips, args.runs)
    print(f"\nFLIP RATE  {flips}/{args.runs} = {flips/args.runs*100:.2f}%   "
          f"95% CI [{lo*100:.2f}%, {hi*100:.2f}%]")
    print(f"A FLIP = the whole generation differs in any token.")
    print(f"Interval applies to THIS sampled workload and THESE settings only: "
          f"temperature {args.temperature}, top_p {args.top_p}, {args.n_predict} "
          f"tokens,\n{len(prompts)} prompt(s) of {args.cached_tokens+1} tokens.")
    print("A low rate does NOT authorize deployment and does not override the failed "
          "numerical gate;\nit is one input to that decision.")
    if flips == 0:
        print(f"\nZero flips is not zero risk: with {args.runs} runs the interval still "
              f"reaches {hi*100:.2f}%.\nQuote the interval, never the point estimate.")


if __name__ == "__main__":
    main()
