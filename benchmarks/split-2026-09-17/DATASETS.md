# Paired datasets for the split figure (17 Sep 2026)

All JSONL is one measurement per line with `pass`, `cfg`, a timestamp, and the value.
`cfg` values: `mac` = MacBook solo, `gb10` = GX10 solo, `50/50` / `25/75` / `15/85` = llama.cpp RPC
layer split, **RPC device first**, so `15/85` is 15 % GB10 / 85 % Mac.

| file | what | model | passes | conditions |
|---|---|---|---|---|
| `datasets/interleaved3.jsonl` | **rotated PREFILL**, the chart's left panel | Qwen3.8-27B UD-Q4_K_XL (dense) | 6, order rotated 3 ways | under concurrent downloads |
| `datasets/dense.jsonl` | **dense crossover sweep**, 640-1280 | same | 4, order rotated 2 ways | under concurrent downloads |
| `datasets/gen.jsonl` | **rotated GENERATION** | same | 4, order rotated 4 ways | **quiet machine, downloads finished** |
| `datasets/interleaved.jsonl` | first interleaved run, fixed order | same | 3 | under downloads; superseded |
| `datasets/interleaved2-FIXEDORDER-partial.jsonl` | aborted fixed-order run, kept deliberately | same | 1 partial | superseded, NOT deleted |
| `datasets/sweep-*.json` | llama-bench JSON from the long ladder that produced the 753 outlier | same | r=3 | disturbed window, see PROVENANCE.md |
| `datasets/glm-split.jsonl` | **GLM-5.3-Flash 320B split, n=3** | GLM-5.3-Flash UD-Q4_K_XL, 185.98 GiB | 3 sequential, NOT rotated | quiet machine; no solo baseline exists, model fits neither box |

Runners in `runners/` produced them; `runners/chart.py` draws `split-scaling-2026-09-17.png`.

## Conditions that differ between panels, and must be stated on any figure

- **PREFILL was measured with two large downloads running**; GENERATION was measured after they finished.
  Mac solo generation reads 27.05 on the quiet machine against 25.10 and 21.46 earlier. In-pass ratios are
  the comparable quantity; absolute rates between the two sets are not.
- **Both sets are dense Qwen3.8-27B, NOT Flash-Next.** Flash-Next has only been measured solo per box
  (Mac beats GB10 1.236x prefill / 1.401x generation); its split run is separate and later.
- Nothing here measures an end-to-end request. pp and tg are separate steady-state rates, so no
  total-request speedup can be read off this data.
- Builds differ per end: Mac `1d0c76f3c` (Metal), GB10 `434ddbbc0` (CUDA 13), same ggml 0.23.0.
