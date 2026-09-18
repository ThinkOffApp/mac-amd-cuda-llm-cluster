# Where the time goes — first decomposition, 2026-09-18

**Mini alone, Metal, 128-token prompt, 64 decode steps. No M5, no network.**
The staged column makes exactly the same device-to-CPU round trips tensor
parallel makes; only the collective is a no-op.

```
                        ms/token    share
GPU-resident
  compute                  9.083     100%

Staged through CPU
  compute                  9.374    51.5%
  device -> CPU            4.520    24.9%
  collective (no-op)       0.016     0.1%
  CPU -> device            4.274    23.5%
  total                   18.184
```

**The device-to-host round trip costs 8.8 ms per token — about as much as all
the GPU arithmetic in the model.** Even with an instantaneous network, staging
alone nearly doubles per-token time.

That redirects the optimisation target named in the goal ("reduce time spent in
traffic"): **the first cost is the copies, not the wire.**

## How to read it

`--profile` takes a GPU sync around every stage, which **inflates the absolute
numbers**. Read the shares. Profiled runs deliberately return no `timings` at
all, so a profiled number can never be quoted as throughput.

## Bounds

One machine. No network. GPT-2 124M. This is the staged *control*, not a
decomposition of the real TP cost, which additionally pays wire time,
synchronisation and waiting for the peer. That measurement needs the M5 GPU and
has not been taken.
