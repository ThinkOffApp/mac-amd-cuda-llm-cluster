# Drafter exactness on the RTX 5090: what was actually compared, 27 November 2026

Sections C and D of
[measurements-2026-11-27-rtx5090-egpu-tensorfold.md](measurements-2026-11-27-rtx5090-egpu-tensorfold.md)
say the drafted text was identical to the undrafted text. This page records how far that comparison
went, so the statement is not read as wider than it is. Same day, same card, same Thunderbolt dock,
same MacBook, same driver route as that page.

## What was compared

One stream, greedy decoding, 200 generated tokens per request, three prompts: the 36-token prose
prompt, the code prompt, and the document-reproduction ("copy") prompt.

| Kernel build | Prompt | Compared | Settings | Result |
|---|---|---|---|---|
| nvcc 13.0.88, without `-Dnative-fp-decode` | prose | plain against DFlash2, and plain against DSpark | 16-bit, Mia's exact settings | byte-identical, 4 of 4 each |
| nvcc 13.0.88, without `-Dnative-fp-decode` | code | plain against DFlash2 | 16-bit | byte-identical, 4 of 4 |
| nvcc 13.0.88, without `-Dnative-fp-decode` | copy | plain against DFlash2 | 16-bit | byte-identical, 4 of 4 |
| nvcc 13.3.73, `-Dnative-fp-decode=true` (rebuilt) | prose | plain against DFlash2 | 16-bit | byte-identical, 4 of 4 |

Each "4 of 4" is the warm-up plus three timed runs, compared file by file with `cmp`.

"Byte-identical" means the generated text of the two runs matched exactly. Both drafted and undrafted
requests came from the same server build on the same card; the only change between a pair was whether
a drafter was loaded.

DFlash2 is incoai's separate draft model. DSpark is the drafting head that ships inside the model
checkpoint, so the DSpark rows needed no extra download.

## What was not compared

- **8-bit settings.** No plain (undrafted) run at 8-bit was saved, so the 8-bit drafted runs have no
  baseline. The one 8-bit drafted output kept on file (`out-nfpdef`) differs from the 16-bit plain run,
  which is what changing the cache and attention precision should do; it says nothing about the drafter.
- **DSpark beyond prose, and the rebuilt kernels beyond prose.** DSpark was compared on the prose prompt only,
  on the nvcc 13.0.88 build. On the rebuilt build only prose has a saved plain run; its code and copy runs were
  drafted only.
- **Linux.** Nothing here compares the Mac route with a Linux sm_120 run of the same prompts on the
  same card. See "Not done" on the 27 November page.

## How to read the claim

Speed numbers on the 27 November page, and any PR text built from them, are "on this setup": this
card, this dock, this Mac driver route, these kernels, 16-bit settings, these three prompts, 200
tokens. Three prompts is a small sample. The comparison is run against run on the same card, not
against Linux, and not against upstream TensorFold main, which differs from Mia's base on long prompts
(27 November page, section H).

To repeat it: run one prompt undrafted and with `--drafter DFlash2` against the same server build, and
compare the generated text, not the token counts or the speed.
