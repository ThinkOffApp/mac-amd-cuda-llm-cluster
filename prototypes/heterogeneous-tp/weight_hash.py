"""DIAGNOSTIC ONLY: hashes the RNG DRAW STREAM, not any run's actual weights.

Caveat raised by @codexmb and it is correct: this uses scale 0.0 for the norm
draws and hashes values BEFORE the +1 transform, so it describes the draw stream
of the SUPERSEDED harness, not the weights any current run uses.

The authoritative per-layer hashes of the actual post-transform tensors are
emitted by model.py itself, under report key "weight_sha256_post_transform".
They live in the run that uses them so they cannot drift from it.

Kept because the cross-build comparison it supports is still valid FOR THE DRAW
STREAM: two builds given the same seed produce different draws.
"""
import hashlib, json, platform, torch

heads, head_dim, hidden = 8, 16, 256
dim = heads * head_dim
g = torch.Generator().manual_seed(20260918)
draws = [((dim,), 0.0), ((dim,), 0.0), ((dim, dim), dim ** -0.5),
         ((dim, dim), dim ** -0.5), ((dim, dim), dim ** -0.5),
         ((dim, dim), dim ** -0.5), ((dim,), 0.1),
         ((dim, hidden), dim ** -0.5), ((hidden,), 0.1),
         ((hidden, dim), hidden ** -0.5), ((dim,), 0.1)]
h = hashlib.sha256()
per = []
for shape, scale in draws:
    t = torch.randn(shape, generator=g) * scale
    b = t.contiguous().numpy().astype('<f4', copy=False).tobytes()
    h.update(b)
    per.append(hashlib.sha256(b).hexdigest()[:16])
print(json.dumps({"host": platform.node(), "torch": torch.__version__,
                  "hip": torch.version.hip, "combined_sha256": h.hexdigest(),
                  "per_tensor_sha256_prefix": per}, indent=2))
