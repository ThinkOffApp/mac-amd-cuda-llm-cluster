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

## Exact and fast mode on the same prompt

One stream, greedy, 200 tokens, the prompt in [`ask.py`](ask.py):
`Explain in plain words what a Thunderbolt eGPU dock does and name two limits of using one.`

| Mode | Settings | Generation (tok/s) | Prompt reading, 5,246-token prompt (tok/s) |
|---|---|---|---|
| Exact: Mia's 16-bit settings | bf16 KV cache, the three `TF_INKLING2_*` switches below unset | 211.7 to 214.7 | 10,660 to 10,687 |
| Fast: Mia's 8-bit defaults | `--kv-cache fp8`, `TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1 TF_INKLING2_ATTN_FP8_TC=1` | 199.5 to 206.7 | 13,673 to 13,734 |

Both with the DFlash2 drafter, three timed runs after one warm-up. Without a drafter the exact mode does 122 tok/s.
On this route the fast mode reads long prompts faster but generates this prompt slightly slower.

"Exact" here means Mia's bf16 settings: drafted text was byte-identical to undrafted text, and on three tested
prompts her 28 patches gave the same tokens as the branch they are built on. That branch itself differs from
upstream TensorFold main on long prompts, so this is not a claim of identity with upstream.

About the 227 tok/s figure that was quoted elsewhere: that was the exact mode on a different, conversational prompt
typed into a chat page. Speed with a drafter depends on the text; use the fixed prompt above to compare.

## You need

- A Mac with Apple silicon and a Thunderbolt 5 or 4 port, an RTX 5090 in an eGPU dock (we used a Minisforum DEG2).
- macuda installed and working, including its TinyGPU system extension: follow macuda's README until its own checks
  pass. This is the only part that touches the system.
- Zig 0.17.0 for macOS, Homebrew `zstd`.
- Once, a Linux machine (x86-64 or ARM64) with Docker, to compile the CUDA kernels. The Mac cannot run nvcc.
- About 20 GB for the model, 1 GB for the draft model.

## Steps

1. **Driver library.** Build [tfcuda](../../tfcuda/) as its README says (`MACUDA=<macuda checkout> sh build.sh`).
   You get `out/libcuda.so.1`.

2. **TensorFold with Mia's patches.** Clone `https://github.com/jschmied/TensorFold.git`, check out
   `dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568`, and apply the 28 patches from
   [MiaAI-Lab/Inkling-2-Mini-RTX-5090-TensorFold](https://github.com/MiaAI-Lab/Inkling-2-Mini-RTX-5090-TensorFold)
   (`patches/0001` to `0028`, in order). Then three small local changes for macOS:
   - `zig/build/product.zig`, the CUDA OS gate: allow `.macos` next to `.linux` and `.windows`.
   - `zig/build/product.zig`, the prebuilt-image profile check: check against the native-fp-decode profile,
     i.e. `catalog.profile(catalog.nativeFp(b.allocator, k, true), target_sm)` instead of
     `catalog.profile(k, target_sm)`, because the kernels in step 3 are built with `-Dnative-fp-decode=true`.
   - `zig/src/cuda/qmmf.zig`: `.clusters = capability_major >= 1000` instead of `>= 9`. macuda has no
     thread-block cluster launch, so this selects the path TensorFold uses before sm_90.

3. **CUDA kernels, on the Linux machine.** In the patched source tree:
   ```
   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/src/.zig-global \
     -v "$PWD":/src -v <zig-0.17.0-for-linux>:/opt/zig:ro -v "$PWD/out-fatbins":/out -w /src \
     nvidia/cuda:13.3.1-devel-ubuntu24.04 \
     /opt/zig/zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dnative-fp-decode=true --prefix /out
   ```
   Copy the `.fatbin` files to the Mac. nvcc 13.2 or later is required: with an older nvcc or without
   `-Dnative-fp-decode=true`, the fast mode's prefill switch stops the card.

4. **Build the server on the Mac**, in the patched tree:
   `zig build -Dgpu=sm_120 -Dfatbins=<folder with the .fatbin files>`

5. **Model and draft model.** `thinkingmachines/Inkling-2-Mini` (Mia's recipe pins revision
   `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`) and `incoai/Inkling-2-Mini-DFlash2` at revision
   `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, into a folder `DFlash2`.

## Run

Exact mode:
```
DYLD_LIBRARY_PATH=<tfcuda>/out TENSORFOLD_DEVICE_CHECK=0 TENSORFOLD_MEMORY_RESERVE_GIB=3 \
  zig-out/native/bin/tensorfold-native serve thinkingmachines/Inkling-2-Mini \
  --host 127.0.0.1 --port 8766 --name inkling-2-mini --context 8192 --drafter DFlash2
```
Fast mode: the same command with `--kv-cache fp8` added, and `TF_INKLING2_ATTN_FP8=1 TF_INKLING2_PREFILL_TC=1
TF_INKLING2_ATTN_FP8_TC=1` in front of it.

Then, once the server answers, run `python3 ask.py 8766 inkling-2-mini 200` four times; the first is warm-up.

- `TENSORFOLD_DEVICE_CHECK=0`: TensorFold's device check looks at the Mac's own GPU, not the card.
- `TENSORFOLD_MEMORY_RESERVE_GIB=3`: without it the first drafted request ran out of card memory.
- Several streams: add `--parallel N`. TensorFold also watches macOS memory pressure and, when macOS reports
  pressure, runs one stream at a time; for a throughput test we set `TENSORFOLD_MEMORY_PRESSURE=0`. With that,
  8 streams gave 516 to 565 tok/s in total and 20 streams (8-bit cache) 891 to 909.

## Known limits

- No thread-block cluster launch on this route (step 2, third change).
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
