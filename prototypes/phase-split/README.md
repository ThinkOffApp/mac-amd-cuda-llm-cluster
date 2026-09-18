# Cross-machine prefill/decode split

Stock `llama.cpp`, no patch: run `llama-server --slot-save-path DIR` on both ends,
`POST /slots/N?action=save` on the producer, copy the file, `action=restore` on the
decoder. The producer must prefill the first **N-1** prompt tokens and the decoder
receive all N, or the decoder re-prefills from scratch.

**SWA models (gemma-3 and friends) force a full re-prefill** unless both ends run
`--swa-full`. An earlier conclusion here that llama.cpp could not do this at all was
wrong: every test behind it used gemma-3, so five tests shared one hidden assumption
and were really one test.

## Measured, 2026-09-18

Mac mini (M4, Metal) + Bosgame M5 (ROCm), gigabit ethernet.
Qwen3.8-27B-UD-Q4_K_XL, 2107-token prompt, **both ends `-c 4096 -ngl 99 -np 1`**.

| step | time |
|---|---|
| M5 prefill, 2107 tokens | 7082.8 ms |
| save slot (281.4 MB) | 92.2 ms |
| transfer M5 to Mini | 2.70 s = 104 MB/s |
| restore on Mini | 91-107 ms |
| Mini finishes (1 prefill + 4 decode) | 0.97 s |
| **TTFT** | **~10.2 s** |
| Mini alone, same prompt | 38.15 s prefill / 38.78 s total |

**3.7x.** Same-machine save/restore: 38,152.7 ms cold to 271.0 ms restored = **141x**,
so the restore itself is nearly free and the wire sets the break-even. KV for this
model is **140 KB/token**.

## Correctness: matching tokens is not equivalence

The transfer is lossless (`sha256` of the slot file is identical on both machines),
but the two backends do not produce the same KV for the same prompt, so the files
differ. `logit_check.py` compares the next-token distributions; `logit_control.py`
adds the control without which the number cannot be read at all, because Metal
reductions need not repeat exactly either.

```
comparison              same text   worst |dlogprob| over shared top-10
pd.bin  vs pd.bin       True        0.000000    <- SAME FILE twice (control)
q27.bin vs q27.bin      True        0.000000    <- SAME FILE twice (control)
pd.bin  vs q27.bin      True        0.670641    <- ROCm-KV vs Metal-KV
```

The same-file control is exactly zero, so the 0.67 is cross-backend and not run
noise. The top-10 sets agreed on only 8-9 of 10 entries. Greedy decoding matched for
four tokens and **that is the whole claim** — a 0.67 logprob gap will diverge under
sampling or over a long generation. Do not quote this result as "token-identical".

Still owed: a long greedy run and a sampled run, to find where the two actually part.

## Running it

Both servers up with `--slot-save-path`, the prompt saved as a token array in a
request body, then:

```bash
python3 logit_control.py   # run this first; a gap is meaningless without it
python3 logit_check.py     # per-step top-10 comparison
```
