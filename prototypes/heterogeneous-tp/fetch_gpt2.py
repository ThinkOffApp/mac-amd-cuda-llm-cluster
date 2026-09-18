"""Download openai-community/gpt2 at a PINNED revision and hash every file.

No remote custom code is executed: this only downloads config, weights and
tokenizer files by name.
"""
import hashlib, json, os, platform, sys
from huggingface_hub import snapshot_download

REPO = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FILES = ["config.json", "generation_config.json", "model.safetensors",
         "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json"]

target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/het-tp/gpt2"
path = snapshot_download(repo_id=REPO, revision=REVISION, local_dir=target,
                         allow_patterns=FILES)
out = {"host": platform.node(), "repo": REPO, "revision": REVISION,
       "path": path, "files": {}}
for name in sorted(os.listdir(path)):
    f = os.path.join(path, name)
    if not os.path.isfile(f):
        continue
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    out["files"][name] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(f)}
print(json.dumps(out, indent=2))
