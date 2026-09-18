# Working slot-state handoff on current GGUF models

A stock `llama-server` can reuse restored state for the tested Qwen3.8-27B and Gemma 4
E4B models. Replaying the original prompt after saving a later generation state is a
different operation: it can require a rewind that the retained recurrent/sliding-window
state cannot support. A failed rewind does not establish that forward handoff is broken.

This is a **same-host mechanism and correctness test**, not a network speedup result.
Producer and fresh consumer ran sequentially on the same Apple M5 Max GPU. Both used
engine `bdcbaaf6e7520b68c8c60ff724c67409970d70e1` and the identical GGUF checkpoint.
No server code change was required. The existing private engine patch is archived;
it has no slot-handler modification. Production services were not touched.

## Recipe that passed

1. Use identical GGUF weights, tokenizer, build, context/KV types and relevant memory
   settings on producer and consumer. Explicitly select and serialize access to a slot.
2. Tokenize the full prompt once. Keep its final prompt token for the consumer.
3. Producer `/completion`: send the prefix token IDs, `cache_prompt: true`,
   `id_slot: 0`, `n_predict: 1`, `temperature: 0`, `return_tokens: true`.
   Discard the sampled token. It exists to finish the producer's request; it is not
   appended to the prompt. Verify the slot's saved prefix length.
4. `POST /slots/0?action=save` with a unique filename. Transfer that file unchanged
   into the consumer's configured `--slot-save-path`; verify its hash and size.
5. On the fresh consumer, `POST /slots/0?action=restore` with the filename.
6. Consumer `/completion`: send the **original full prompt token IDs**, including
   the held-back final prompt token. Select the restored slot and enable prompt caching.
   This evaluates the real final prompt token and obtains consumer-side continuation
   logits, rather than relying on producer logits that the slot file does not preserve.
7. Check the active completion's `timings.cache_n` and `timings.prompt_n`, not an
   absent field on an idle `/slots` response. Check continued outputs against native
   prefill before using the result in a performance table.

For **Gemma 4 E4B**, the tested recipe additionally requires `--swa-full` on both ends.
Without it, the server's sliding-window/checkpoint guard reprocessed the prompt even
at this forward boundary. Full SWA retains more cache and changes memory requirements.
For **Qwen3.8-27B**, default hybrid/recurrent memory worked at the forward boundary;
replaying the old prompt from the post-generation save did not.

Both tested servers used `-c 4096 -np 1 -b 512 -ub 512 -t 4 -ngl 999`,
`--cache-ram 0`, and loopback-only ports 19083/19084. Disabling the separate server
RAM cache isolates slot-file reuse. The archived scripts contain exact commands,
requests and responses. Slot files remain local, with byte counts and SHA256 hashes
in the committed manifest rather than hundreds of megabytes of binary state in Git.

## Receipts and limits

Synthetic prompt: 2,112 tokens; transferred prefix: 2,111 tokens. A fresh server
restored a copied state file, then generated 16 tokens. We also extended the prompt
with those returned tokens and new input and compared that next turn to cold native
prefill. Gemma ended the next turn after one EOS token; Qwen generated all 16.

| Model | State bytes | Cached / evaluated prompt tokens | Fresh continuation IDs | Next-turn IDs |
|---|---:|---:|---|---|
| Gemma 4 E4B Q4_K_M, full SWA | 55,598,700 | 2,111 / 1 | 16/16 match | EOS-aware match |
| Qwen3.8-27B UD-Q4_K_XL | 295,300,388 | 2,111 / 1 | 16/16 match | 16/16 match |

Warm next turns reused 2,127 tokens and evaluated six additional tokens in both cases.
The warm native comparator matters: avoiding repeated cold prefill is not a benefit
unique to a distributed implementation. Cold native timings varied substantially;
no speedup ratio is claimed from these local tests.

### Full-logit check: preserve the failed control too

`gguf_state_check.cpp` restores the HTTP-produced slot file in another executable and
compares all vocabulary logits for 16 continuation steps against native inference.
The predeclared tolerance is unchanged: `atol=0.1`, `rtol=0.01`.

The initial uniform-chunk native prefill **failed** that gate despite identical token
IDs: max difference 0.129621 for Gemma and 1.036802 for Qwen. It did not use the same
prefill graph shapes as the producer. Matching the producer's chunk boundaries and
its separate final prompt token changed the result:

| Model | Native prefill chunk end positions | Max logit difference | Gate |
|---|---|---:|---|
| Gemma 4 | 512, 1024, 1536, 2048, 2111, 2112 | 0.00498199 | Pass |
| Qwen 27B | 512, 1024, 1536, 1599, 2107, 2111, 2112 | 0 | Pass |

The Qwen boundaries come from the server's checkpoint-aware prefill schedule. These
matched tests isolate state handoff more closely; they do **not** turn the failed
uniform-prefill comparison into a pass or establish equivalence for arbitrary
chunking/backends. All failed and passing receipts are retained under
`results/slot-handoff-20260918`, alongside the initial checker source and exact runners.

## Remaining physical test

Use matching llama.cpp and GGUF on both hosts for the first CUDA-to-Metal or
Vulkan-to-Metal probe. A vLLM NVFP4 cache is **not** this file format and cannot be
restored by this recipe without a separate converter. Verify cross-backend continued
logits/tokens before performance, then measure producer prefill, serialization,
transfer, restore, consumer tail evaluation, TTFT and complete response latency.
Count all phases and compare against both cold and warm native controls with the
same token workload. Include memory headroom and the transferred recurrent state.
There is no verified physical-pair handoff or end-to-end speedup in these receipts.
