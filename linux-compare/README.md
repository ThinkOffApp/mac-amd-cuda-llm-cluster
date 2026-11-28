# Linux comparison: served token hashes, Mac RTX 5090 against a Linux sm_120 card

Ash asked (TensorFold RFC, 27 Nov 2026) for served token hashes against Linux with the same checkpoint and settings,
bit-identical. This folder runs the same request set on both machines and compares the token ids.

- `hash_run.py`: asks a TensorFold server for TensorFold's 16 fixed prompts (`bench_prompts.json`, copied from
  `tools/` in TensorFold), greedy, seed 1, 128 tokens, `ignore_eos`, and records the sha256 of the reply token ids.
  Every prompt is asked twice and both runs must agree. Same request shape as TensorFold's `tools/exact_restarts.py`.
- `colab_linux_hashes.ipynb`: the Linux side on Google Colab with an RTX PRO 6000 (sm_120). It builds TensorFold at
  `ab268f923` (ThinkOffApp/TensorFold `local/macos-cuda-pr`), checks its 166 kernel images against
  `fatbins-sm120.sha256` (the Mac's images, nvcc 13.3.1), serves Inkling 2 Mini and writes `colab-hashes.json`.
  Open it in Colab: https://colab.research.google.com/github/ThinkOffApp/mac-amd-cuda-llm-cluster/blob/linux-compare/linux-compare/colab_linux_hashes.ipynb
- Settings on both sides: no drafts (`--no-drafts`), 16-bit (no `--kv-cache fp8`, no `TF_INKLING2_*` switches),
  context 8192, the model weights with the SHA-256 listed in the recipe.

The Mac side, with the server built from the same commit and the same images:
```
DYLD_LIBRARY_PATH=<tfcuda>/out TENSORFOLD_DEVICE_CHECK=0 \
  tensorfold-native serve thinkingmachines/Inkling-2-Mini --host 127.0.0.1 --port 8766 \
  --name inkling-2-mini --context 8192 --no-drafts --no-update-check
python3 hash_run.py --base http://127.0.0.1:8766 --label mac-rtx5090 --out mac-hashes.json
python3 hash_run.py --compare mac-hashes.json colab-hashes.json
```

Caveat for reading the result: the RTX PRO 6000 has 188 SMs and the RTX 5090 170, and TensorFold chooses some tilings
by SM count, so a kernel may sum in a different order on each. A full match is strong evidence; a difference is not by
itself a bug, and would call for the 5090 itself in a Linux machine.
