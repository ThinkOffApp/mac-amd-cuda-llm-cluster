"""Which comparison actually answers "does the split change the answer"?

Four quantities were argued over on 2026-09-18 and three of them proxy for the
question rather than answering it. This demonstrates why, with no model and no
hardware, so the reasoning can be checked in a second.

    full-vocab RAW LOGITS   softmax is shift-invariant, so a uniform shift across
                            the vocabulary is a large raw-logit difference that
                            changes NOTHING. The gating number, 9.46, is of this
                            kind and nobody has decomposed it.

    full-vocab LOG-PROBS    the right space -- sampling depends on these -- but a
                            MAXIMUM over ~152k entries is set by whichever entry
                            moved most, and the tail moves most while mattering
                            least. A token at p=1e-17 can fail an atol=0.1 gate.

    TOTAL VARIATION         0.5 * sum |p - q|. The MINIMUM probability that a
                            single sampled token differs, over all couplings --
                            a LOWER bound, achieved only by the optimal coupling.
                            An earlier version of this file called it the maximum
                            and said "whatever the sampler does". @claudeMB
                            caught it; it was wrong in the direction that
                            licenses a deployment. Measured below.

    observed FLIP RATE      the only thing that answers the question. No cheap
                            bound stands in for it.

AND TV MUST BE COMPUTED ON THE DISTRIBUTION ACTUALLY SAMPLED. @codexmb: a small TV
before truncation need not stay small after temperature, top-k/top-p and separate
renormalisation. Measured below -- it amplifies by 1.46x when the two distributions
disagree about which tokens make the cut, and shrinks when they agree. It is a
rank-boundary effect near k, so raw TV is not even a floor for post-truncation TV.

TV ALSO BOUNDS ONE NEXT-TOKEN DRAW AT A SHARED HISTORY. Once two sequences diverge
their histories differ, so per-step TV does not compose into an answer-level bound.
flip_rate.py measures whole generations empirically, which sidesteps that rather than
solving it.

TV IS STILL WORTH COMPUTING. It is a floor, so a large TV settles the question
immediately and against us. TV = 0 exactly means p = q, and identical
distributions give identical samples under a shared seed, so a zero is
conclusive. It is the small-but-nonzero range where it says much less than it
looks like it says.
"""
import math
import random

VOCAB = 151_936


def logprobs(z):
    m = max(z)
    lse = m + math.log(sum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


def probs(z):
    return [math.exp(v) for v in logprobs(z)]


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def tv(p, q):
    """Total variation: the MINIMUM disagreement probability over couplings."""
    return 0.5 * sum(abs(x - y) for x, y in zip(p, q))


def cdf_sample(dist, u):
    c = 0.0
    for i, x in enumerate(dist):
        c += x
        if u <= c:
            return i
    return len(dist) - 1


def coupling_demo(n=200_000, vocab=2000, sd=0.05, seed=7):
    """Show that TV is a floor, not a ceiling, by sampling three couplings."""
    rng = random.Random(seed)
    base = [rng.gauss(0, 3) for _ in range(vocab)]
    noisy = [v + rng.gauss(0, sd) for v in base]
    p, q = probs(base), probs(noisy)
    t = tv(p, q)

    r1 = random.Random(99)
    shared = sum(1 for _ in range(n)
                 for u in (r1.random(),)
                 if cdf_sample(p, u) != cdf_sample(q, u))
    r2 = random.Random(1234)
    indep = sum(1 for _ in range(n)
                if cdf_sample(p, r2.random()) != cdf_sample(q, r2.random()))

    print(f"\n{'TV(p,q)':<36}{t*100:>9.3f}%")
    print(f"{'optimal coupling (theory)':<36}{t*100:>9.3f}%   <- TV IS THIS")
    print(f"{'shared uniform, same token order':<36}{shared/n*100:>9.3f}%"
          f"   <- {shared/n/t:.1f}x TV")
    print(f"{'independent randomness':<36}{indep/n*100:>9.3f}%"
          f"   <- {indep/n/t:.0f}x TV")
    print("\nTV is the MINIMUM over couplings. It does not cap anything.")
    print("Caveat on this demo: inverse-CDF over an untruncated 2,000-token")
    print("distribution. Real sampling truncates to top_k first, where orderings")
    print("agree far more often, so the multiple above is illustrative. Only the")
    print("DIRECTION of the inequality is certain.")


def row(label, base, other):
    print(f"{label:<34}{maxdiff(base, other):>14.6f}"
          f"{maxdiff(logprobs(base), logprobs(other)):>14.6f}"
          f"{tv(probs(base), probs(other)):>14.3e}")


def topk_renorm(p, k, temp=1.0):
    """What the sampler actually draws from. Each distribution is truncated
    SEPARATELY, so the two need not keep the same k tokens -- which is where the
    amplification comes from."""
    if temp != 1.0:
        z = [math.log(max(x, 1e-300)) / temp for x in p]
        m = max(z)
        lse = m + math.log(sum(math.exp(v - m) for v in z))
        p = [math.exp(v - lse) for v in z]
    idx = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:k]
    total = sum(p[i] for i in idx)
    out = [0.0] * len(p)
    for i in idx:
        out[i] = p[i] / total
    return out


def truncation_demo(vocab=4000, seed=11):
    print("\nTV on the RAW distribution vs TV on what is actually sampled:\n")
    rng = random.Random(seed)
    print(f"{'settings':<28}{'TV raw':>10}{'TV sampled':>12}{'kept sets differ':>18}")
    for sd, temp, k in [(0.05, 1.0, 40), (0.05, 0.7, 40), (0.05, 1.0, 5), (0.02, 1.0, 40)]:
        base = [rng.gauss(0, 3) for _ in range(vocab)]
        noisy = [v + rng.gauss(0, sd) for v in base]
        p, q = probs(base), probs(noisy)
        pt, qt = topk_renorm(p, k, temp), topk_renorm(q, k, temp)
        kp = {i for i, x in enumerate(pt) if x > 0}
        kq = {i for i, x in enumerate(qt) if x > 0}
        print(f"{f'sd={sd} temp={temp} k={k}':<28}{tv(p,q)*100:>9.4f}%"
              f"{tv(pt,qt)*100:>11.4f}%{len(kp ^ kq):>12} of {k}")
    print("\nAmplified exactly when the kept sets differ; reduced when they agree.")
    print("So raw TV is not even a floor for the TV that governs sampling.")


def main():
    random.seed(20260918)
    base = [random.gauss(0, 6) for _ in range(VOCAB)]

    print(f"vocabulary {VOCAB:,}\n")
    print(f"{'':<34}{'max RAW LOGIT':>14}{'max LOG-PROB':>14}{'TV distance':>14}")
    row("uniform shift of 9.46", base, [v + 9.46 for v in base])
    row("non-uniform noise, sd 0.05", base,
        [v + random.gauss(0, 0.05) for v in base])
    tail = list(base)
    tail[-1] -= 3.0
    row("one p=1e-17 token moved 3.0", base, tail)

    print("\nRow 1: a raw-logit gate FAILS, nothing can change.")
    print("Row 3: a log-prob gate FAILS at atol=0.1, nothing can change.")
    print("Only TV is small in both -- but TV is a FLOOR, not a ceiling:")
    coupling_demo()
    truncation_demo()


if __name__ == "__main__":
    main()
