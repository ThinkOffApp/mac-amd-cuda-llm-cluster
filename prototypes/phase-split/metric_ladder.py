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

    TOTAL VARIATION         0.5 * sum |p - q|. Exactly the maximum probability
                            that a single sampled token differs. A BOUND, not an
                            estimate, and free of any assumption about top_k.

    observed FLIP RATE      what actually happens under real settings. TV is the
                            ceiling; the flip rate is the realisation. A measured
                            flip rate above TV means the harness is wrong.
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
    """Total variation. Bounds the probability that one sampled token differs."""
    return 0.5 * sum(abs(x - y) for x, y in zip(p, q))


def row(label, base, other):
    print(f"{label:<34}{maxdiff(base, other):>14.6f}"
          f"{maxdiff(logprobs(base), logprobs(other)):>14.6f}"
          f"{tv(probs(base), probs(other)):>14.3e}")


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
    print("Only TV is small in both, and only TV bounds what a sampler can do.")


if __name__ == "__main__":
    main()
