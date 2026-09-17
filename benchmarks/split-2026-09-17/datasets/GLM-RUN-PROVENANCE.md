# GLM-5.3-Flash Q4_K_XL split run — provenance

**Model**: `/Volumes/t705/ModelArchive/GLM-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf`
(6 shards; shard 1 is a 9.4 MB metadata+vocab shard by design, not a placeholder)
llama-bench reports: `glm5next 312B.A17B Q4_K - Medium`, **185.98 GiB, 320.76 B params**

**Client**: MacBook Pro M5 Max, `~/llama-glm5/build/bin/llama-bench`, build **1d0c76f3c (753)**, Metal
**Server**: GX10 `gx10-6678`, `~/llama.cpp/build-rpc/bin/ggml-rpc-server -H 10.10.10.2 -p 50052 -d CUDA0`,
build **434ddbbc0 (10884)**, CUDA 13 — NOTE: the two ends are DIFFERENT builds
**Link**: direct 10 GbE, Mac `en12` 10.10.10.1 ↔ 10.10.10.2

**Command** (17 Sep 04:17, one pass, NOT interleaved, NOT order-rotated):

    llama-bench -m <shard1> --rpc 10.10.10.2:50052 -ts 50/50 -p 512,2048 -n 32 -r 2

| test | t/s |
|---|---:|
| pp512 | 312.22 ± 10.04 |
| pp2048 | 424.18 ± 3.12 |
| tg32 | 13.69 ± 0.23 |

(± here is llama-bench's own within-run stddev over r=2, NOT an across-pass sample s.d. like the
interleaved datasets. These numbers are a single run and are weaker evidence than anything in
`interleaved3.jsonl` / `gen.jsonl`.)

**`-ts 40/60` FAILED**: `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Allocator breakdown from the
successful 50/50 run: MTL0 92,164 MiB model + 76 ctx + 365 compute, 17,493 MiB free;
RPC0 93,272 MiB model + 86 ctx + 221 compute.

**Correctness**: with `--jinja` applying the GGUF's chat template the split answered a technical question
correctly at temp 0; an untemplated completion chain was also correct. See
`../glm-gate-artifacts/gate5-raw.log` (unfiltered). This is a functional smoke test, **not** a
matched-reference equivalence check across the two builds.

Raw logs: `glm-split.log`, `glm-gate-templated.log`.

---

## UPDATE 17 Sep 09:20Z — repeated to n=3

The single pass above was the weakest evidence in the whole set. Passes 2 and 3 were run with the
**identical command** on the same two builds, machine otherwise quiet, and recorded per pass in
`glm-split.jsonl`. Raw log: `../glm-gate-artifacts/glm-repeat-pass2-3-raw.log`.

| test | pass 1 (04:17) | pass 2 (09:06Z) | pass 3 (09:13Z) | **mean ± s.d. (n=3)** | CV |
|---|---:|---:|---:|---:|---:|
| pp512 | 312.22 | 316.03 | 318.01 | **315.42 ± 2.94** | 0.9 % |
| pp2048 | 424.18 | 419.17 | 430.21 | **424.52 ± 5.53** | 1.3 % |
| tg32 | 13.69 | 13.84 | 14.55 | **14.03 ± 0.46** | 3.3 % |

The ± in this table is an **across-pass sample s.d.**, the same quantity as in `interleaved3.jsonl` and
`gen.jsonl`, so these figures are now directly comparable with the rest of the datasets. The per-pass
`sd_within` values in the JSONL remain llama-bench's own r=2 spread and are a different quantity.

**Caveat that the ± hides: tg32 rises monotonically across the three passes** (13.69, 13.84, 14.55).
With n=3 a monotone trend cannot be distinguished from noise, but it is the pattern drift would make,
so the generation figure carries more uncertainty than its 3.3 % CV suggests. pp512 and pp2048 show no
such ordering. If the generation number matters for a published claim, run it interleaved and
order-rotated the way `gen.jsonl` was, rather than as three sequential passes.

**Still true and unchanged:** this model cannot run solo on either box (185.98 GiB against a Mac Metal
working set of 115,448 MB and a GB10 with 124,544 MiB), so there is **no single-machine baseline** for
it and no speedup ratio can be quoted. The numbers describe the only configuration in which the model
runs at all.
