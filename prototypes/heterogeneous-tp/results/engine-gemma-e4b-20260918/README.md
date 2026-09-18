# Gemma 4 E4B quantized tensor-split gate, 18 September 2026

Model: `unsloth/gemma-4-E4B-it-GGUF`, revision
`bfc15c382204943c3a8fff0c750b94ae2364d7a3`, `gemma-4-E4B-it-Q4_K_M.gguf`.
Downloaded size 4,977,171,584 bytes and full SHA-256
`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`
match the repository's published LFS metadata. The model name E4B should not be
read as a claim of exactly four billion total stored parameters.

Engine: llama.cpp `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`, using the same
local empty-view RPC patch documented in `../engine-27b-20260918/README.md`.
No architecture gate was bypassed. This run does not establish whether Gemma
requires that patch; it tests the same experimental engine build.

The helper `engine_correctness.cpp` compared unpartitioned Metal against a
50/50 tensor split between local Metal and a local Metal RPC peer. Both use
one physical Mac GPU. Three prompts, up to 16 output choices each (14 before EOS, then 16 and 16):
**46/46 token choices match**, all logits finite, all full-vocabulary logits within the
predeclared atol 0.1 / rtol 0.01 criterion. Per-prompt maximum logit errors are
0.011820, 0.006348, and 0.013723. Reports preserve the raw token IDs, minimum
reference top-two margins, and source/patch hashes.

This is a real quantized billion-parameter model and a local RPC tensor-split
correctness gate. It is not a physical Mac/PC performance result, a comparison
with GPT-2 speed, or a universal output-quality guarantee. No profiling or
throughput claim follows from these cold-load correctness runs. A matched
hardware-pair comparison and communication measurements remain outstanding.

Use the reproduction procedure in the 27B results directory with this pinned
GGUF. Full reference logits remain in local output files; the repository keeps
compact reports. Temporary loopback RPC processes were terminated after each
run. No Mia or Flash-Next service was interrupted.
