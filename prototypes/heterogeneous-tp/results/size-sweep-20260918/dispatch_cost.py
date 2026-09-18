"""Decisive control: is the fixed cost the COPY, or the GPU round trip?

If a trivial GPU op followed by a wait costs about the same as op+copy, then the
copy contributes almost nothing and what we have been calling "staging cost" is
really the cost of forcing the GPU pipeline to drain 24 times per token.
"""
import json, statistics as st, time, torch
dev = "mps"; torch.set_num_threads(1)
REPS = 300
x = torch.randn(768, device=dev)
out = {}

def med(fn, reps=REPS):
    for _ in range(30): fn()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts) * 1000

out["gpu_op_only_ms"]        = med(lambda: x.add(1.0))            # op, stays on GPU
out["gpu_op_then_copy_ms"]   = med(lambda: x.add(1.0).to("cpu"))  # op + copy to host
out["copy_only_ms"]          = med(lambda: x.to("cpu"))           # copy of a settled tensor
out["nothing_ms"]            = med(lambda: None)                  # loop + sync overhead
print(json.dumps(out, indent=1))
