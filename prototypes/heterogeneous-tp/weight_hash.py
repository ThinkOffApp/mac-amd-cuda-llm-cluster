"""Hash the rank-0 weight draw, to test whether torch version changes it.

Replicates decode.py's generator draw ORDER exactly (no collectives, rank 0 path).
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
