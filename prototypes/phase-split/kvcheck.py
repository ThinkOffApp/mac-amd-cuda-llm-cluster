"""Compare next-token distributions from two saved KV caches, and refuse to
produce a number unless every step of the comparison is known to have worked.

WHY THIS REPLACES THE FIRST VERSION
    @codexmb reproduced, with no network at all, that the original printed the
    headline result -- "same text True, worst gap 0.000000" -- for a pair of
    FAILED requests. Three independent holes lined up:

        curl -sS exits 0 on an HTTP error, so a 404 looked like success
        run() never read the restore response, so a failed restore was invisible
        gap([], []) returns 0.0, so an empty distribution list scored perfectly

    A zero produced by a dead server is indistinguishable from a zero produced
    by a genuine control, and I had published that zero as the evidence the
    cross-backend gap was real. Every check below exists because of that.

WHAT IT REFUSES TO DO
    Return a number when: the HTTP status is not 2xx; the body is an error
    object; a restore reports a token count other than the one expected; any
    step has an empty distribution; any probability is missing or non-finite;
    or the two runs disagree on how many steps they produced.

    Run --self-test first. It drives the whole path against a server that is not
    there and asserts the script FAILS. A control that cannot fail proves
    nothing, so the failing case is the thing that has to be demonstrated.

WHAT A PASSING RESULT STILL DOES NOT SHOW
    The shared top-10 is not a full-vocabulary gate. Repeating a restore of one
    fixture excludes run-to-run variation in that fixture; it does not isolate
    backend differences from differing prefill chunk boundaries, builds or
    weights. Agreement on a handful of greedy tokens is a narrow positive
    observation and nothing more.
"""

import argparse
import json
import math
import subprocess
import sys

BANNER = (
    "NOT THE ACCEPTANCE GATE. This reports a TOP-K LOG-PROBABILITY difference.\n"
    "The gate is a FULL-VOCABULARY RAW-LOGIT comparison at atol=0.1 AND rtol=0.01.\n"
    "log p = logit - logsumexp, so a uniform shift across the vocabulary vanishes\n"
    "here and is fully visible there; these numbers cannot be compared with gate\n"
    "results or put in the same table. A quiet top-k is consistent with a loud\n"
    "vocabulary: one measured run had the top-10 move 0.67 while 179,629 logits\n"
    "were out of tolerance.\n"
)



class CheckFailed(Exception):
    """Anything that makes the printed number untrustworthy."""


def request(base, path, payload, timeout=600):
    """POST and return parsed JSON, or raise. The status code is captured
    separately because curl's exit status does not carry it."""
    marker = "\n__HTTP_STATUS__:"
    out = subprocess.run(
        ["curl", "-sS", "--max-time", str(timeout), "-X", "POST", base + path,
         "-H", "Content-Type: application/json",
         "-w", marker + "%{http_code}",
         "-d", json.dumps(payload)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise CheckFailed(f"{path}: curl exit {out.returncode}: {out.stderr.strip()[:200]}")
    body, _, status = out.stdout.rpartition(marker)
    if not status.strip().isdigit():
        raise CheckFailed(f"{path}: no status code in response: {out.stdout[:200]}")
    code = int(status)
    if not 200 <= code < 300:
        raise CheckFailed(f"{path}: HTTP {code}: {body.strip()[:300]}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        raise CheckFailed(f"{path}: HTTP {code} but body is not JSON: {body[:200]}")
    if isinstance(parsed, dict) and "error" in parsed:
        raise CheckFailed(f"{path}: error object with HTTP {code}: {parsed['error']}")
    return parsed


def assert_cache_hit(completion, expect_cached, slotfile):
    """The restored cache must be the thing that answered.

    @codexmb found the failure this prevents: sending back the SAME prompt that was
    saved triggers llama.cpp's rewind path, the server re-prefills every token, and
    the comparison silently becomes "two fresh native prefills" -- which of course
    agree exactly. The restore is then decorative and the zero means nothing.
    """
    t = completion.get("timings", {})
    cache_n, prompt_n = t.get("cache_n"), t.get("prompt_n")
    if cache_n is None or prompt_n is None:
        raise CheckFailed(f"{slotfile}: timings lack cache_n/prompt_n; cannot prove a cache hit")
    if cache_n != expect_cached:
        raise CheckFailed(
            f"{slotfile}: cache_n={cache_n}, expected {expect_cached}. The restored "
            "cache was NOT used -- this is the re-prefill path and the comparison is void.")
    if prompt_n != 1:
        raise CheckFailed(
            f"{slotfile}: prompt_n={prompt_n}, expected 1. More than the held-back "
            "token was evaluated, so the saved history is not what produced this output.")


def restore(base, slotfile, expect_tokens):
    r = request(base, "/slots/0?action=restore", {"filename": slotfile})
    n = r.get("n_restored")
    if n is None:
        raise CheckFailed(f"restore {slotfile}: no n_restored in {json.dumps(r)[:200]}")
    if n != expect_tokens:
        raise CheckFailed(
            f"restore {slotfile}: restored {n} tokens, expected {expect_tokens}. "
            "A short restore silently re-prefills and the comparison is meaningless.")
    return r


def steps_of(completion, slotfile):
    probs = completion.get("completion_probabilities")
    if not probs:
        raise CheckFailed(
            f"{slotfile}: no completion_probabilities. Either this build does not "
            "support n_probs or the request was rejected -- either way there is "
            "nothing to compare, and an empty list must not score as agreement.")
    out = []
    for i, step in enumerate(probs):
        entries = step.get("top_logprobs") or step.get("probs")
        if not entries:
            raise CheckFailed(f"{slotfile}: step {i} has an empty distribution")
        # Key by token ID, never by decoded text. Distinct IDs can decode to the
        # same string, so a text-keyed dict silently collapses them -- which is why
        # a request for the top 10 was reporting 9 shared entries.
        d = {}
        for e in entries:
            tid, lp = e.get("id"), e.get("logprob", e.get("prob"))
            if tid is None:
                raise CheckFailed(
                    f"{slotfile}: step {i} entry has no token id; this build cannot "
                    "support an ID-keyed comparison and text keys collapse duplicates")
            if lp is None:
                raise CheckFailed(f"{slotfile}: step {i} entry missing logprob: {e}")
            if not math.isfinite(lp):
                raise CheckFailed(f"{slotfile}: step {i} non-finite logprob for id {tid}")
            if tid in d:
                raise CheckFailed(f"{slotfile}: step {i} duplicate token id {tid}")
            d[tid] = (lp, e.get("token"))
        if len(d) != len(entries):
            raise CheckFailed(f"{slotfile}: step {i} lost entries when keyed by id")
        out.append(d)
    return out


def run(base, slotfile, req, expect_tokens, n_predict, n_probs):
    restore(base, slotfile, expect_tokens)
    body = dict(req)
    body.update(n_predict=n_predict, cache_prompt=True, temperature=0.0,
                top_k=0, top_p=1.0, n_probs=n_probs)
    c = request(base, "/completion", body)
    assert_cache_hit(c, expect_tokens, slotfile)
    return c.get("content"), steps_of(c, slotfile)


def compare(a_steps, b_steps, label_a, label_b):
    if len(a_steps) != len(b_steps):
        raise CheckFailed(f"{label_a} produced {len(a_steps)} steps, "
                          f"{label_b} produced {len(b_steps)} -- not comparable")
    worst, rows = 0.0, []
    for i, (da, db) in enumerate(zip(a_steps, b_steps)):
        shared = set(da) & set(db)
        if not shared:
            raise CheckFailed(f"step {i}: the two top-10 sets share no token at all")
        d = max(abs(da[k][0] - db[k][0]) for k in shared)
        worst = max(worst, d)
        top_a = max(da, key=lambda k: da[k][0])
        top_b = max(db, key=lambda k: db[k][0])
        rows.append((i, f"{da[top_a][1]!r}#{top_a}", f"{db[top_b][1]!r}#{top_b}",
                     d, len(shared), len(da)))
    return worst, rows


def self_test(args):
    """Drive the real code path against a server that is not there, and require
    a failure. This is the case the old script passed."""
    dead = "http://127.0.0.1:9"  # discard port: connects nowhere
    cases = [
        ("unreachable server", lambda: request(dead, "/completion", {})),
        ("HTTP error on restore",
         lambda: restore(args.base, "definitely-not-a-slot-file.bin", 1)),
        ("empty probability list",
         lambda: steps_of({"completion_probabilities": []}, "fixture")),
        ("step with no entries",
         lambda: steps_of({"completion_probabilities": [{"top_logprobs": []}]}, "fixture")),
        ("non-finite logprob",
         lambda: steps_of({"completion_probabilities":
                           [{"top_logprobs": [{"token": "x", "logprob": float("-inf")}]}]},
                          "fixture")),
        ("mismatched step counts",
         lambda: compare([{1: (0.0, "a")}], [{1: (0.0, "a")}, {2: (0.0, "b")}], "x", "y")),
        ("disjoint top-10 sets",
         lambda: compare([{1: (0.0, "a")}], [{2: (0.0, "b")}], "x", "y")),
        ("entry with no token id",
         lambda: steps_of({"completion_probabilities":
                           [{"top_logprobs": [{"token": "x", "logprob": -1.0}]}]}, "fixture")),
        ("restore not actually used (re-prefill path)",
         lambda: assert_cache_hit({"timings": {"cache_n": 0, "prompt_n": 2107}}, 2107, "fixture")),
        ("more than the held-back token evaluated",
         lambda: assert_cache_hit({"timings": {"cache_n": 2107, "prompt_n": 5}}, 2107, "fixture")),
        ("timings without cache_n",
         lambda: assert_cache_hit({"timings": {"prompt_n": 1}}, 2107, "fixture")),
    ]
    bad = 0
    for name, fn in cases:
        try:
            fn()
        except CheckFailed as e:
            print(f"  FAILS as required  {name:<24} {str(e)[:70]}")
            continue
        except Exception as e:  # noqa: BLE001 - any other escape is also a hole
            print(f"  FAILS as required  {name:<24} ({type(e).__name__})")
            continue
        print(f"  *** DID NOT FAIL    {name}  <- this is the bug class, fix it")
        bad += 1
    print(f"\nself-test: {len(cases) - bad}/{len(cases)} cases fail as they must")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8091")
    ap.add_argument("--request", default="/tmp/q_full.json",
                    help="JSON body carrying the tokenised prompt")
    ap.add_argument("--expect-tokens", type=int, required=False,
                    help="tokens each restore must report; required unless --self-test")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="A:B slot-file pairs; A:A is the same-file control")
    ap.add_argument("--n-predict", type=int, default=4)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--verbose", action="store_true", help="per-step table")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test(args))
    if args.expect_tokens is None or not args.pairs:
        ap.error("--expect-tokens and at least one --pairs A:B are required")

    req = json.load(open(args.request))
    print(BANNER)
    print(f"{'comparison':<26} {'same text':<10} {'steps':<6} worst |dlogprob|")
    failures = 0
    for pair in args.pairs:
        a, _, b = pair.partition(":")
        try:
            ca, sa = run(args.base, a, req, args.expect_tokens, args.n_predict, args.n_probs)
            cb, sb = run(args.base, b, req, args.expect_tokens, args.n_predict, args.n_probs)
            worst, rows = compare(sa, sb, a, b)
        except CheckFailed as e:
            print(f"{pair:<26} CHECK FAILED -- no number: {e}")
            failures += 1
            continue
        kind = "SAME FILE twice (control)" if a == b else "cross-file"
        print(f"{pair:<26} {str(ca == cb):<10} {len(rows):<6} {worst:.6f}   <- {kind}")
        if args.verbose:
            for i, ta, tb, d, ns, nt in rows:
                print(f"    step {i}  {str(ta)[:14]:<16} {str(tb)[:14]:<16} "
                      f"{d:.6f}  (shared {ns}/{nt})")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
