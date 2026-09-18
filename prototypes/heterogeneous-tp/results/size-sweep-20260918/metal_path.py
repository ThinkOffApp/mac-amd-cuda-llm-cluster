"""Is the 0.15 ms per copy PyTorch's cost, or Metal's?

PyTorch MPS device->host costs ~0.15 ms for 3 KB regardless of size, sync or
allocation. MLX is Apple's own framework on the same unified memory and the same
GPU. If MLX moves the same bytes far cheaper, the cost is PyTorch's MPS copy
path and is therefore fixable software, not a property of the hardware.
"""
import json, statistics as st, time
import numpy as np, torch, mlx.core as mx

REPS = 300
FLOATS = 768                                  # one GPT-2 collective payload, 3 KB
out = {}

def med(fn, sync, reps=REPS):
    for _ in range(30): fn()
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t0)
    return st.median(ts) * 1000

# PyTorch MPS
t = torch.randn(FLOATS, device="mps")
out["torch_mps_to_cpu_ms"] = med(lambda: t.to("cpu").contiguous(), torch.mps.synchronize)

# MLX: force evaluation on GPU, then read to host via numpy
a = mx.random.normal((FLOATS,)); mx.eval(a)
def mlx_to_host():
    b = mx.add(a, 0.0)      # a real GPU op so we are not timing a no-op
    mx.eval(b)
    return np.array(b, copy=True)
out["mlx_gpu_op_plus_to_host_ms"] = med(mlx_to_host, lambda: mx.eval(a))

# MLX host read of an already-evaluated array (pure transfer, no op)
out["mlx_to_host_only_ms"] = med(lambda: np.array(a, copy=True), lambda: mx.eval(a))

# control: a same-size host-to-host numpy copy, the floor of "just moving bytes"
h = np.random.randn(FLOATS).astype(np.float32); dstn = np.empty(FLOATS, dtype=np.float32)
out["numpy_host_to_host_ms"] = med(lambda: np.copyto(dstn, h), lambda: None)

print(json.dumps(out, indent=1))
