# Recipe: Inkling 2 Mini on an RTX 5090 driven from a Mac

Run Thinking Machines' Inkling 2 Mini on an NVIDIA RTX 5090 in a Thunderbolt 5 dock, served by TensorFold from
macOS, with Mia's speed patches and incoai's DFlash2 draft model. Measured on 27 November 2026.

## What runs where

| On the Mac | On the RTX 5090 |
|---|---|
| TensorFold's server process: HTTP, tokenizer, scheduling, sampling, the drafter's control logic | Every model kernel: all layers of Inkling 2 Mini, the DFlash2 draft model, attention, experts, logits |
| [tfcuda](../../tfcuda/), our translation library, turning TensorFold's CUDA driver calls into macuda calls | The model weights and the KV cache, in the card's 32 GB |
| [macuda](https://github.com/Davinchy/macuda)'s user-space driver, which builds command buffers and submits them over Thunderbolt | |

Nothing of the model runs on the Mac's own GPU or CPU. The Mac drives the card; every token is computed on the 5090.

## One stream: exact and memory-saving modes on three prompts

All with the DFlash2 drafter, one stream, greedy, 200 tokens, three timed runs after one warm-up. Kernels rebuilt
with nvcc 13.3.73 and `-Dnative-fp-decode=true`. The prompts are in [`ask.py`](ask.py). Prose and code
rows include a second, clean-room build made from a fresh clone by following the steps below (27 Nov 2026).

| Prompt | Exact (16-bit) tok/s | Memory-saving (8-bit) tok/s |
|---|---|---|
| prose (the eGPU question) | 211.7 to 214.7 | 199.3 to 207.4 |
| code (parse_csv with tests) | 352.3 to 361.6 | 293.5 to 297.5 |
| document reproduction (repeat a given text) | 386.6 to 387.7 | 164.3 to 169.5 |

- Exact: Mia's 16-bit settings: bf16 KV cache, the three `TF_INKLING2_*` switches below unset.
- Memory-saving: Mia's 8-bit defaults: `--kv-cache fp8`, `TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1 TF_INKLING2_ATTN_FP8_TC=1`.
- Naming: Mia calls these defaults "fast"; this recipe uses "memory-saving" because the smaller cache fits more streams, while single-stream generation with the drafter can be slower.
- What is counted: Inkling 2 Mini thinks before it answers, and in these 200-token runs all 200 tokens are still
  thinking (the final answer field is empty). So tok/s here is the speed of the model's reasoning text on each prompt.
  "Code" means a coding prompt, not a finished code answer; the reasoning on it is more predictable, which is why the
  drafter helps more there.

Speed with a drafter depends strongly on how predictable the text is: code and copying go fastest. Without a drafter
the exact mode does 122 tok/s on the prose prompt.

Why the memory-saving mode generates slower with the drafter: the 8-bit settings shift the big model's numbers slightly, so it
rejects more of the draft model's guesses. Drafted tokens accepted, one request each, same prompts, DFlash2:

| Prompt | 16-bit | 8-bit |
|---|---|---|
| prose | 115/217 (53%) | 117/265 (44%) |
| code | 157/241 (65%) | 147/280 (52%) |
| document reproduction | 160/197 (81%) | 102/282 (36%) |

Without a drafter on the prose prompt: 16-bit 121 tok/s, 8-bit attention alone 132, 8-bit cache alone 116. Mia's
own notes state about +22% for 8-bit attention at one stream on her card; we measured about +9% here.

So the memory-saving (8-bit) settings save memory (20 streams fit instead of 11) and read long prompts faster; for one stream
with the drafter, the 16-bit settings generate faster on this setup.

Prompt reading (prefill), tok/s:

| Prompt length | Exact (16-bit) | Memory-saving (8-bit) |
|---|---|---|
| 5,246 tokens | 10,660 to 10,687 | 13,673 to 13,734 |
| 20,046 tokens | | 14,205 to 14,237 |
| 100,046 tokens | | 11,602 to 11,617 |

"Exact" here means Mia's bf16 settings. On the 5090, drafted text was byte-identical to undrafted text. Separately,
on a Linux GB10 box (not the 5090), her 28 patches gave the same tokens as the branch they are built on, on three
tested prompts. That branch itself differs from upstream TensorFold main on long prompts, so this is not a claim of
identity with upstream.

About the 227 tok/s figure that was quoted elsewhere: that was the exact mode on a different, conversational prompt
typed into a chat page. Speed with a drafter depends on the text; use the fixed prompts above to compare.

## Several streams

Exact mode settings, no drafter unless stated, 200 tokens per stream, three rounds, all streams running together,
`TENSORFOLD_MEMORY_PRESSURE=0`. Server flags `--context 2048 --parallel N`; for 20 streams also `--kv-cache fp8` and
`TF_INKLING2_ATTN_FP8=1`.

| Streams | Total tok/s |
|---|---|
| 1 | 115 to 122 |
| 8 | 516 to 565 |
| 8, with DFlash2 | 523 to 619 |
| 11 (most that fit at 16-bit) | 667 to 692 |
| 20, 8-bit cache and attention (most that fit) | 891 to 909 |

TensorFold watches macOS memory pressure and, when macOS reports pressure, runs one stream at a time; for a
throughput test we set `TENSORFOLD_MEMORY_PRESSURE=0`.

## You need

- A Mac with Apple silicon and a Thunderbolt 5 or 4 port, an RTX 5090 in an eGPU dock (we used a Minisforum DEG2).
- macuda installed and working, including its TinyGPU system extension: follow macuda's README until its own checks
  pass, and build its driver (`sh setup.sh shim` in the macuda checkout), which tfcuda links against. This is the
  only part that touches the system.
- Zig 0.17.0 for macOS (TensorFold's build refuses any other version; check with `zig version`), Homebrew `zstd`,
  Xcode or the Command Line Tools.
- Once, a Linux machine (x86-64 or ARM64) with Docker and Zig 0.17.0 for Linux on that machine's architecture, to
  compile the CUDA kernels. The Mac cannot run nvcc. No GPU is needed on that machine.
- About 20 GB for the model, 1 GB for the draft model.

## Steps

1. **Driver library.** In this repository's [tfcuda](../../tfcuda/) folder:
   ```
   MACUDA=<macuda checkout> sh build.sh
   ```
   The last line printed is `exported cu* symbols: 78`. You get `out/libcuda.so.1`.

2. **TensorFold with Mia's patches.** Commit `dc2ca8a1` is the head of a TensorFold pull request, not on any branch of
   the fork, so a plain clone does not contain it: fetch it by hash. Mia's patches are plain diffs (`git am` refuses
   them), so apply each with `git apply` and commit it:
   ```
   git clone https://github.com/jschmied/TensorFold.git
   git clone https://github.com/MiaAI-Lab/Inkling-2-Mini-RTX-5090-TensorFold mia-recipe
   cd TensorFold
   git fetch origin dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568
   git switch -c inkling-mac dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568
   for p in ../mia-recipe/patches/00*.patch; do
     git apply "$p" && git add -A && git commit -q -m "$(basename "$p")" || { echo "failed: $p"; break; }
   done
   ```
   That applies `patches/0001` to `0028` in order (tested with Mia's repository at commit `86f1d69`). Patch 0016
   prints a harmless warning about a blank line at the end of a file. `git commit` needs a git name and email set.

3. **Three small changes for macOS**, in the patched tree:
   ```
   git apply <this recipe folder>/macos.patch
   ```
   [`macos.patch`](macos.patch) changes three lines:
   - `zig/build/product.zig`, the CUDA OS gate: allow `.macos` next to `.linux` and `.windows`.
   - `zig/build/product.zig`, the prebuilt-image profile check: check against the native-fp-decode profile,
     i.e. `catalog.profile(catalog.nativeFp(b.allocator, k, true), target_sm)` instead of
     `catalog.profile(k, target_sm)`, because the kernels in step 4 are built with `-Dnative-fp-decode=true`.
   - `zig/src/cuda/qmmf.zig`: `.clusters = capability_major >= 1000` instead of `>= 9`. macuda has no
     thread-block cluster launch, so this selects the path TensorFold uses before sm_90.

4. **CUDA kernels, on the Linux machine.** Copy the patched tree there (for example with `rsync -a`), then, in it:
   ```
   mkdir -p out-fatbins
   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/src/.zig-global \
     -v "$PWD":/src -v <zig-0.17.0-for-linux>:/opt/zig:ro -v "$PWD/out-fatbins":/out -w /src \
     nvidia/cuda:13.3.1-devel-ubuntu24.04 \
     /opt/zig/zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dnative-fp-decode=true --prefix /out
   ```
   `<zig-0.17.0-for-linux>` is the unpacked Zig folder, the one that holds the `zig` binary. Create `out-fatbins`
   first: otherwise Docker creates it owned by root and the build, running as you, cannot write into it.
   The images land in `out-fatbins/fatbin/`: 140 `.fatbin` files, about 47 MB. Copy that folder's `.fatbin` files
   to the Mac. nvcc 13.2 or later is required: with an older nvcc or without
   `-Dnative-fp-decode=true`, the 8-bit prefill switch stops the card.

5. **Build the server on the Mac**, in the patched tree:
   ```
   zig build -Dgpu=sm_120 -Dfatbins=<folder with the .fatbin files>
   ```
   `zig` must be 0.17.0 (Homebrew's `zig` may be older; put the 0.17.0 folder first on `PATH` or call its `zig` by
   path). `-Dfatbins` names the folder that directly holds the `.fatbin` files, the copy of `out-fatbins/fatbin/`.
   You get `zig-out/native/bin/tensorfold-native`. Building does not touch the GPU.

6. **Model and draft model.** `thinkingmachines/Inkling-2-Mini` (Mia's recipe pins revision
   `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`; the weight files' SHA-256 are what matter:
   `model-00001-of-00001.safetensors` `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`,
   `mtp.safetensors` `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`), in the Hugging Face cache,
   where the server looks it up by name. `incoai/Inkling-2-Mini-DFlash2` at revision
   `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, into a folder `DFlash2` (any path; `--drafter` takes it).

## Run

Exact mode:
```
DYLD_LIBRARY_PATH=<tfcuda>/out TENSORFOLD_DEVICE_CHECK=0 TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  zig-out/native/bin/tensorfold-native serve thinkingmachines/Inkling-2-Mini \
  --host 127.0.0.1 --port 8766 --name inkling-2-mini --context 8192 --drafter DFlash2
```
Memory-saving mode: the same command with `--kv-cache fp8` added, and `TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1
TF_INKLING2_ATTN_FP8_TC=1` in front of it.

- `TENSORFOLD_DEVICE_CHECK=0`: TensorFold's device check looks at the Mac's own GPU, not the card.
- `TENSORFOLD_MEMORY_RESERVE_GIB=3`: without it the first drafted request ran out of card memory.

### Repeating the tables

One stream: start the server in the mode of the column, wait until it answers, then run
`python3 ask.py 8766 inkling-2-mini 200 <prompt>` four times; the first is warm-up. `<prompt>` is the row:

| Row | `ask.py` argument |
|---|---|
| prose | `prose` (the default) |
| code | `code` |
| document reproduction | `copy` |

Without a drafter: leave out `--drafter DFlash2`. For the acceptance figures, use the same prompts and read the
server's log of drafted and accepted tokens.

Several streams: start the server with `TENSORFOLD_MEMORY_PRESSURE=0` in front of the command, `--context 2048
--parallel N` instead of `--context 8192`, and without `--drafter DFlash2` unless the row says DFlash2. Then run
`python3 ask_many.py 8766 inkling-2-mini N` three times; it sends N different prompts at once, 200 tokens each, and
prints the total tok/s.

| Row | Server | `ask_many.py` |
|---|---|---|
| 1 | exact mode, `--parallel 1` | `1` |
| 8 | exact mode, `--parallel 8` | `8` |
| 8, with DFlash2 | exact mode, `--parallel 8 --drafter DFlash2` | `8` |
| 11 | exact mode, `--parallel 11` | `11` |
| 20 | `TF_INKLING2_ATTN_FP8=1` in front, `--parallel 20 --kv-cache fp8` | `20` |

## Known limits

- No thread-block cluster launch on this route (step 3, third change).
- TensorFold's full kernel contract has not completed on this route; a filtered part passed.
- Tested on one MacBook Pro M5 Max (macOS 27), one RTX 5090, one dock, on one day.
- Sleep and unplugging the dock while serving are untested.

## Credits

- Mia, [@MiaAI-Lab](https://github.com/MiaAI-Lab): the 28 TensorFold patches that give the speed.
- Inco AI, [@incoai](https://github.com/incoai): the DFlash2 draft model.
- Ash Hart, [@ashhart](https://github.com/ashhart): [TensorFold](https://github.com/ashhart/TensorFold).
- Jürgen Schmied, [@jschmied](https://github.com/jschmied): the TensorFold branch Mia's patches are built on.
- Antonio Torres, [@Davinchy](https://github.com/Davinchy): [macuda](https://github.com/Davinchy/macuda), the
  user-space NVIDIA driver for macOS this route runs on.
- Thinking Machines: Inkling 2 Mini.

All measurements and every table:
[docs/measurements-2026-11-27-rtx5090-egpu-tensorfold.md](../../docs/measurements-2026-11-27-rtx5090-egpu-tensorfold.md).
