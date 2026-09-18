"""Download a pinned GPT-2 family member and hash every file."""
import hashlib, json, os, platform, sys
from huggingface_hub import snapshot_download
REPOS = {
 "gpt2":        ("openai-community/gpt2",        "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
 "gpt2-medium": ("openai-community/gpt2-medium", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
 "gpt2-large":  ("openai-community/gpt2-large",  "32b71b12589c2f8d625668d2335a01cac3249519"),
}
FILES = ["config.json","generation_config.json","model.safetensors",
         "merges.txt","vocab.json","tokenizer.json","tokenizer_config.json"]
key = sys.argv[1]; target = sys.argv[2]
repo, rev = REPOS[key]
path = snapshot_download(repo_id=repo, revision=rev, local_dir=target, allow_patterns=FILES)
out = {"host": platform.node(), "repo": repo, "revision": rev, "files": {}}
for n in sorted(os.listdir(path)):
    f = os.path.join(path, n)
    if not os.path.isfile(f): continue
    h = hashlib.sha256()
    with open(f,'rb') as fh:
        for c in iter(lambda: fh.read(1<<20), b''): h.update(c)
    out["files"][n] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(f)}
print(json.dumps(out, indent=2))
