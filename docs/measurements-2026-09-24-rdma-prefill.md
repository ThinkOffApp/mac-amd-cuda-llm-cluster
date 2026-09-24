# Prefill on the GX10, decode on the Mac, over RDMA (23-24 September 2026)

The first model run over our own RDMA link, and to our knowledge the first hardware run of the
remote prefill path in [jundot/omlx#3870](https://github.com/jundot/omlx/pull/3870). The GX10
prefills the prompt, the KV cache comes back to the Mac over MCDMA, and the Mac decodes. This splits
the work by **phase**, not by layer, so it avoids the pipeline stall that sank every layer split in
[measurements-mac-spark-split.md](measurements-mac-spark-split.md).

Full report, with build identities and every caveat: our contribution
[ashhart/MCDMA#6](https://github.com/ashhart/MCDMA/pull/6). Hardware report on the oMLX PR:
[comment on #3870](https://github.com/jundot/omlx/pull/3870#issuecomment-5806895559).

## Setup

- **Mac:** MacBook Pro M5 Max, 128 GB, macOS 27 build 26A428, oMLX 0.7.0.dev4 at #3870's head `c48c1787`.
- **Peer:** ASUS Ascent GX10 (GB10), vLLM 0.27.1 in `nvcr.io/nvidia/vllm:26.08-py3`, with MCDMA's
  `integrations/vllm/mcdma_kv` connector.
- **Link:** MCDMA 0.1.18 (Ash Hart's macOS RDMA driver), ConnectX-5 Ex in an OWC Helios 5S to the
  GX10's ConnectX-7, 100GBASE-CR4, one `mcdma-rpcd` link with 4 + 64 MiB mailboxes. Transport
  numbers for this link are in [interconnect.md](interconnect.md).
- **Model:** `Qwen/Qwen3-4B-Instruct-2507`. BF16 on the GX10; BF16 or MLX MXFP4 (group 32, 4.25 bits
  per weight) on the Mac.
- **Method:** needle-in-a-haystack prompts with fresh random filler (no prefix cache can help),
  greedy decoding, replies forced to the 128-token cap, streamed and timed end to end by the client,
  medians of three. A split request counted only if oMLX logged a completed handoff for it.

## Results, MXFP4 on the Mac (best setup)

Prefill tok/s = prompt tokens ÷ time to first token (for the split this includes the RDMA transfer).

| Prompt tokens | Prefill tok/s: Mac only / split | Decode tok/s (Mac) | 128-token reply, s: Mac only / split | Split saves |
|---:|---|---:|---|---:|
| 3,831 | **4,893** / 4,362 | 155.1 | **1.60** / 1.69 | -6% |
| 7,600 | 4,353 / **4,811** | 133.3 | 2.70 / **2.54** | 6% |
| 15,140 | 3,409 / **4,454** | 106.1 | 5.64 / **4.64** | 18% |
| 28,270 | 2,652 / **3,735** | 76.5 | 12.90 / **9.24** | **28%** |

The split pays off on long prompts, and the gain grows with length. Decode speed is the Mac's either
way, so longer replies shrink the percentage. 28 of 30 answers were correct. The two misses
reproduced in both configurations with the cache cleared, so they are the 4-bit model, not the
handoff.

## Against Ash Hart's Studio run (his numbers, not ours)

Ash's [disaggregated-inference note](https://github.com/ashhart/MCDMA/blob/main/docs/disaggregated-inference.md)
ran the same model on an M3 Ultra Mac Studio and a DGX Spark, MXFP4 on both sides, with reply times
summed from separately measured stages. It is not like for like: our GX10 prefilled in BF16, and our
times are end to end.

| Prompt, ours / his | Decode tok/s: our M5 Max / his M3 Ultra | Whole-reply output tok/s: our split / his split |
|---|---|---|
| 3,831 / 3,852 | **155.1** / 147 | **75.8** / 69.6 |
| 7,600 / 7,702 | **133.3** / 131 | **50.5** / 43.1 |
| 15,140 / 15,402 | 106.1 / **109** | **27.6** / 23.1 |
| 28,270 / 28,852 | 76.5 / **83** | **13.9** / 11.4 |

At the same quantisation the M5 Max decodes within about 8% of the M3 Ultra. It prefills about
twice as fast alone (28k tokens: 10.7 s against his 22.3 s), which is why our split saves less in
percent (28% against his 53%). In absolute time our split is faster: 9.24 s against 11.21 s at the
longest prompt.

## What made it faster

1. **The producer checksum was the bottleneck.** The connector's `zlib.crc32` ran at 6.4 GB/s on the
   GB10 and held checked KV transfers to 16-19 Gbit/s. python-isal computes the identical CRC at
   18.9 GB/s, which raised them to 26-30 Gbit/s. Fix submitted upstream in
   [ashhart/MCDMA#5](https://github.com/ashhart/MCDMA/pull/5).
2. **Bigger prefill chunks on vLLM** (`--max-num-batched-tokens 16384`, default 2048): the 28k prefill
   dropped from about 6.4 to 6.0 s.
3. **MXFP4 on the Mac:** decode 1.8-2.9x faster than BF16, prefill almost unchanged.

Tried and dropped: checksums off (no further gain after step 2), and online FP8 on vLLM (doubled the
GX10's own decode, but made long prefill slower).

With BF16 on the Mac the split saved 21% at 28k before tuning and 24% after (10.54 s against
13.90 s).

## Not covered

One small model, one link, one request at a time, three runs per point. No tensor-parallel
producer. The next speedup is code rather than a setting: streaming the cache layer by layer while
prefill is still running.
