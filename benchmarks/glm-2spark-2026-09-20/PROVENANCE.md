# GLM-5.3-Flash-EXL3 on two Sparks: one bounded streaming measurement

> **TRANSPORT: `NCCL_NET=Socket` with `NCCL_IB_DISABLE=1`, after `ibv_reg_mr_iova2`
> returned ENOMEM. TCP. This is explicitly NOT an RDMA success** — it is what the stack
> fell back to when RDMA memory registration failed. See [`TRANSPORT.md`](../../TRANSPORT.md),
> which records transport per run, because a different run on the same two boxes used
> `NCCL_NET=IB`.

## Attribution

**This measurement is not ours and not codexmb's.** It was **produced by the 2026-09-20
Codex task**; **codexmb independently inspected the artifact** and confirmed its contents.
It is included here because the README cites its headline number, and a cited number a
reader cannot check is weaker than a committed one.

`measurement.json` is the **exact bytes of the original**, copied unmodified from
`Documents/Codex/2026-09-20/che/outputs/glm-performance/measurement.json`. It has not been
reformatted, re-keyed or prettified — a reformatted file is a retelling, not evidence.

```
sha256  cf0b7df3d52437fcf2c528429f3a5949a65b954946597352ed35d211dc55bf11
bytes   445
```

The attribution lives in this file rather than inside the JSON precisely so the JSON's
bytes stay unchanged; adding a header field to it would have broken the checksum above.

**Contents check before committing:** numeric timing, usage and count fields plus nulls.
No credentials, no hostnames, no private endpoints. The only matches for "token" are
`prompt_tokens`, `total_tokens`, `completion_tokens` and the two rate fields.

## What the number is, stated precisely

**19.16877 is the overall completion-token generation rate for a length-capped,
reasoning-only response with zero final-answer characters. It is not answered-task
performance.**

| field | value |
|---|---:|
| `usage.prompt_tokens` | 24 |
| `usage.completion_tokens` | 384 |
| `usage.total_tokens` | 408 |
| `elapsed_seconds` | 20.032584084023256 |
| `first_text_seconds` | 0.3227123340475373 |
| `first_answer_seconds` | **null** |
| `reasoning_characters` | 1634 |
| `answer_characters` | **0** |
| `stream_chunks` | 106 |
| `finish_reason` | **`length`** |
| `overall_completion_tokens_per_second` | **19.16877015912563** |
| `decode_tokens_per_second_approx` | 19.431886968035286 |

**The run produced no answer.** All 384 completion tokens went into the reasoning block,
the run hit its token cap, and it emitted zero answer characters — hence
`first_answer_seconds: null` and `answer_characters: 0` beside a non-null
`first_text_seconds`.

So the rate is correctly measured and the *task* is not measured at all. Those are two
different quantities:

- **usable as** — how fast this stack emits completion tokens under these settings
- **NOT usable as** — how fast this stack answers a question, or any per-request latency

## Serving configuration

**MEASURED by codexmb** (the serve itself, as distinct from the artifact above):
MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks, checkout
`6961fa0706f3c0b25775bf42a575471972582bac`, image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, both ASUS Ascent GX10
boxes serving GLM-5.3-Flash-EXL3. **DFlash 7, MTP 2, context 8192, max seq 2.**

**What this directory does NOT establish.** One measurement, one prompt, no repetitions, no
control arm, and a reasoning-only response. It is a single data point about emission rate on
a socket-fallback transport. It is not a benchmark of the model, the hardware, or the link,
and it should not be compared against any completed-task figure.

## The methodology note this produced

A harness that decides "did it work" by looking for visible output will **silently
mis-grade every reasoning model**: a run can spend its whole budget thinking and return
empty content with `finish_reason: length`. The same shape appeared twice on 20 Sep 2026 —
here, and separately on the MacBook where a freshly loaded reasoning model given a 40-token
cap returned empty content and was nearly recorded as broken.

**Key success on completion tokens produced, and treat `finish_reason: length` WITH tokens
as a success.** This is also carried in the README's *Honest pitfalls* section.
