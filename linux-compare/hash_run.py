#!/usr/bin/env python3
"""Served token hashes for one TensorFold server, to compare machines bit for bit.

Same request shape as TensorFold's tools/exact_restarts.py: greedy, seed 1, ignore_eos, and the reply's token ids
(`tensorfold.token_ids`), not the text. Each prompt of bench_prompts.json (TensorFold's fixed set, copied here) is
asked twice; both runs must agree, and the sha256 of the ids is what you compare with the other machine.

    python3 hash_run.py --base http://127.0.0.1:8766 --label mac-5090 --out mac-5090.json
    python3 hash_run.py --compare mac-5090.json colab-rtxpro6000.json
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ids_for(base, item, tokens, timeout=900):
    body = {"model": "served", "max_tokens": tokens, "temperature": 0, "seed": 1,
            "ignore_eos": True, "return_token_ids": True}
    if item["kind"] == "chat":
        url, body["messages"] = base + "/v1/chat/completions", [{"role": "user", "content": item["prompt"]}]
    else:
        url, body["prompt"] = base + "/v1/completions", item["prompt"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        reply = json.load(resp)
    ids = reply.get("tensorfold", {}).get("token_ids")
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"{item['name']}: the reply carried no tensorfold.token_ids")
    return ids


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "no nvidia-smi (macOS route)"


def run(args):
    prompts = [p for p in json.loads(Path(args.prompts).read_text()) if isinstance(p.get("name"), str)]
    rows, ok = [], True
    for item in prompts:
        a = ids_for(args.base, item, args.tokens)
        b = ids_for(args.base, item, args.tokens)
        same = a == b
        ok &= same
        rows.append({"name": item["name"], "kind": item["kind"], "tokens": len(a), "sha256": digest(a),
                     "repeat_identical": same, "first_ids": a[:12]})
        print(f"{item['name']:28s} {len(a):4d} tokens  {digest(a)[:16]}  {'repeat ok' if same else 'REPEAT DIFFERS'}",
              flush=True)
    all_hash = hashlib.sha256("".join(r["sha256"] for r in rows).encode()).hexdigest()
    out = {"label": args.label, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu_name(),
           "host": f"{platform.system()} {platform.machine()}", "tokens": args.tokens, "all_sha256": all_hash,
           "all_repeat_identical": ok, "prompts": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nall prompts: {all_hash}\nrepeats identical: {ok}\nwritten: {args.out}")


def compare(a_path, b_path):
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    print(f"{a['label']} ({a['gpu']}) vs {b['label']} ({b['gpu']})")
    problems = []
    if a["tokens"] != b["tokens"]:
        problems.append(f"token budgets differ: {a['tokens']} vs {b['tokens']}")
    names_a, names_b = [r["name"] for r in a["prompts"]], [r["name"] for r in b["prompts"]]
    if names_a != names_b:
        problems.append(f"prompt lists differ ({len(names_a)} vs {len(names_b)} prompts, or another order)")
    for side in (a, b):
        bad = [r["name"] for r in side["prompts"] if not r["repeat_identical"]]
        if bad or not side.get("all_repeat_identical", False):
            problems.append(f"{side['label']}: repeats not identical on {', '.join(bad) or 'some prompt'}")
    for p in problems:
        print("  NOT COMPARABLE: " + p)
    if problems:
        return False
    same_n = 0
    for ra, rb in zip(a["prompts"], b["prompts"]):
        same = ra["sha256"] == rb["sha256"] and ra["tokens"] == rb["tokens"]
        same_n += same
        note = "" if same else next((f" first difference at token {i}" for i, (x, y) in enumerate(zip(ra["first_ids"], rb["first_ids"])) if x != y), " (beyond the first 12 tokens)")
        print(f"  {ra['name']:28s} {'IDENTICAL' if same else 'DIFFERS' + note}")
    print(f"\n{same_n} of {len(a['prompts'])} prompts bit-identical at {a['tokens']} tokens; all-prompt hash "
          f"{'IDENTICAL' if a['all_sha256'] == b['all_sha256'] else 'DIFFERS'}")
    return same_n == len(a["prompts"]) and a["all_sha256"] == b["all_sha256"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8766")
    ap.add_argument("--prompts", default=str(HERE / "bench_prompts.json"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="hashes.json")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    run(args)
