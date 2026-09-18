"""SUPERSEDED by @claudeMB's ~/bin/modelcheck.py -- do not use this one.

    Two tools for one job is worse than one tested tool, and his is the better
    of the two: it queries the registry live so it cannot go stale, exits 2 when
    it cannot check at all so it can never pass silently, and it ships with a
    regression set.

    Decisively, it already fixes a bug THIS file still has. The parser below
    reads a SIZE as a GENERATION:

        gemma-7b        -> family gemma, "generation" 7     WRONG
        CodeLlama-70b   -> family codellama, "generation" 70 WRONG

    So this file would wave through a gemma-7b as newer than gemma 4. He found
    that class in his own tool and locked it with tests before trusting it; I
    found it in mine only after reading his post.

    Kept, not deleted, for the one idea worth porting into his: the --reason a
    caller gives for an older generation is PRINTED WITH THE RESULT, so it
    travels with the number instead of living in someone's memory. Petrus's
    words were "for no stated reason", and a stated reason has to be attached
    to the output to mean anything.

Original docstring follows.

Refuse to benchmark a model whose generation nobody checked.

WHY THIS EXISTS
    On 2026-09-18 a request-split benchmark was run on gemma-3-4b. Gemma 4 had
    been out since 2 April 2026 -- five months. The model file was downloaded
    that same afternoon, so it was not inherited; it was chosen because a
    harness already existed for it.

    Petrus, twice: "Using ancient for no stated reason is not ok" and "there's
    zero excuse for this, and especially frustrating to AGAIN have to REPEAT
    this BASIC COMMON SENSE." It is a recurring fault across ThinkOff's history,
    not a one-off.

    "My training data is old" is not available as a reason. The rule to check
    current versions rather than rely on training data already existed and was
    skipped. So this makes skipping it fail loudly instead of quietly.

WHAT IT DOES
    Reads a registry of what the current generation is, keyed by family, with
    the date each entry was verified and by whom. A model whose family has no
    entry, or whose entry is stale, or which is not the current generation, is
    REFUSED unless a reason is given on the command line -- and the reason is
    printed with the result so it travels with the number.

    It does not go online. A check nobody recorded is not a check, and a tool
    that silently fetches an answer hides the very step that was skipped.
"""

import argparse
import datetime as dt
import json
import os
import re
import sys

REGISTRY = os.path.join(os.path.dirname(__file__), "model-currency.json")
STALE_DAYS = 60


def load():
    try:
        with open(REGISTRY) as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"REFUSING: no registry at {REGISTRY}. Record what is current first.")


def family_of(path):
    """Best-effort family + generation from a GGUF filename."""
    name = os.path.basename(path).lower()
    m = re.match(r"([a-z]+)[-_.]?(\d+(?:\.\d+)?)", name)
    return (m.group(1), m.group(2)) if m else (name.split("-")[0], None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path or name of the model about to be used")
    ap.add_argument("--reason", default="",
                    help="why an older generation is the right choice HERE; printed "
                         "with the result so it travels with the number")
    a = ap.parse_args()

    reg = load()
    fam, gen = family_of(a.model)
    entry = reg.get("families", {}).get(fam)

    if entry is None:
        sys.exit(f"REFUSING: '{fam}' has no entry in {os.path.basename(REGISTRY)}.\n"
                 f"Find out what the current {fam} generation is, add it with the date\n"
                 f"and who verified it, then run again. An unchecked family is how the\n"
                 f"old version stays.")

    checked = dt.date.fromisoformat(entry["verified"])
    age = (dt.date.today() - checked).days
    if age > STALE_DAYS:
        sys.exit(f"REFUSING: the '{fam}' entry was verified {age} days ago "
                 f"({entry['verified']}).\nRe-verify before relying on it.")

    current = str(entry["current_generation"])
    if gen is not None and gen != current:
        if not a.reason:
            sys.exit(f"REFUSING: {os.path.basename(a.model)} is {fam} {gen}; "
                     f"current is {fam} {current}\n"
                     f"({entry['current_example']}, verified {entry['verified']} "
                     f"by {entry['verified_by']}).\n"
                     f"Pass --reason if an older generation is genuinely right here. "
                     f"'It was already\non disk' is not a reason -- it is the "
                     f"mechanism by which the old one stays.")
        print(f"OLDER GENERATION IN USE: {fam} {gen}, current is {fam} {current}")
        print(f"  stated reason: {a.reason}")
        print(f"  QUOTE THIS WITH ANY NUMBER FROM THIS RUN.")
        return 0

    print(f"ok: {fam} {gen or '?'} is current "
          f"(verified {entry['verified']} by {entry['verified_by']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
