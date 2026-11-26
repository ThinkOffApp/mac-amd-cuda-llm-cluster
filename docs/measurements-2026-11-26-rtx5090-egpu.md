# An RTX 5090 in a Thunderbolt 5 dock: first evening, 26 November 2026

One card, one dock, two hosts, one evening. This page records what happened, in
order, including what did not work. Nothing here is tuned, and nothing has been
repeated on a second day.

## Hardware

- **Card:** MSI GeForce RTX 5090, 32 GB (PCI `10de:2b85`, subsystem `1462:5306`).
- **Dock:** Minisforum DEG2, Thunderbolt 5 and OCuLink, powered by a separate
  1200 W ATX supply. It reports itself as `TBGAA` from Micro Computer (HK) Tech.
  Only the Thunderbolt/USB4 side was used. No OCuLink cable was at hand.
- **Host 1:** Bosgame HX 470 mini PC (AMD Ryzen AI 9 HX 470, Radeon 890M, one
  front USB4 port, 27 GiB RAM visible), Ubuntu 26.04.1, kernels 7.0.0-43 and
  7.0.0-44, Secure Boot on.
- **Host 2:** MacBook Pro, Apple M5 Max, 128 GB, macOS 27.0.1.

## Summary

| Host | Link the host reported | Result |
|---|---|---|
| HX 470, USB4 | USB4, 40 Gb/s (2 lanes of 20) | Card enumerates and `nvidia-smi` works. The host then hard-freezes within minutes, also when the card is idle. No model was ever loaded. |
| MacBook, Thunderbolt 5, tinygrad 0.14.0 | USB4 v2, 80 Gb/s; PCIe x4 at 16 GT/s | Card works through tinygrad's TinyGPU driver, once per power cycle of the dock. A second start fails at card initialisation. |
| MacBook, Thunderbolt 5, macuda | same | llama.cpp's CUDA backend runs on the card: Llama 3.1 8B Q8 decodes at 158 tok/s, on a cold start and on two warm starts after it. |

## Host 1: Linux over USB4

Driver: `nvidia-headless-no-dkms-615-open` 615.71.09 with Canonical's prebuilt
signed modules for both kernels, from the Ubuntu archive. Compute parts only.

What worked:

- The dock was authorised by `bolt` on first contact, and the card appeared on
  PCI behind the USB4 bridge.
- `nvidia-smi` reported `NVIDIA GeForce RTX 5090, 0 MiB / 32607 MiB`.
- With hot-plug, the kernel could not place the larger prefetchable bridge
  windows ("can't assign; no space"). BAR 1 stayed at 256 MB. That was enough for
  the driver to load.

What went wrong, in order (UTC):

1. **18:10** The USB4 link dropped for about a second, 16 seconds after the card
   first appeared. The driver logged `Xid 79, GPU has fallen off the bus`. The
   display part of the driver (`nvidia_modeset`, `nvidia_drm`) then looped on
   "Error while waiting for GPU progress" and blocked PCI hot-plug handling. A
   clean reboot hung on the same loop and needed the power button.
   After that boot the two display modules were blocked from loading
   (`install nvidia_drm /bin/false`, same for `nvidia_modeset`), since the card
   is used for compute only.
2. **18:51** The whole dock vanished from the host at once (USB hub, USB4 link,
   card) while idle. This time the driver cleaned up and the host kept running.
   The dock came back only after the cable was reseated at 19:27.
3. **About 19:35** The host froze hard with a solid green screen. The last kernel
   log line on disk is from 19:27.
4. **19:41** After a power cycle with the dock attached, the host froze again
   about two minutes after boot. A live log stream to a second machine shows the
   card idle at 0 MiB and 46 C in the last sample, two seconds before the freeze,
   and no kernel message at all. The soft and hard lock-up detectors had been
   set to panic and did not fire. No crash dump was written.

What this does and does not show: the freezes happen with the card attached over
USB4 and the driver loaded, without any load on the card. The cause is not
known. The cable, the dock firmware, the host's USB4 controller and the driver
are all still candidates. OCuLink on the same host is untested.

## Host 2: macOS over Thunderbolt 5

macOS has no NVIDIA driver. The card was reached with tinygrad's TinyGPU, a
signed and notarised DriverKit extension (`org.tinygrad.tinygpu.driver2`,
1.0.0/3), with tinygrad v0.14.0. The NVIDIA compiler ran in a container
(`cuda-nvcc 12.8`, linux/arm64, under Colima).

tinygrad's master branch removed the macOS remote path on 5 September 2026
("remove hcq1 remote for now"), so v0.14.0 is the last release that has it. Its
install script on master no longer matches the code.

Measured, one process, 8192 x 8192 float32 matrix multiply, best of five after
one warm-up, tinygrad's generic kernel with no kernel search:

| Device | Best | Runs (ms) | GFLOP/s |
|---|---|---|---|
| RTX 5090 through TinyGPU (`DEV=NV`) | 80.4 ms | 80.6, 90.8, 80.6, 80.4, 80.6 | about 13,700 |
| M5 Max through Metal (`DEV=METAL`) | 76.4 ms | 76.6, 102.4, 76.4, 76.8, 76.4 | about 14,400 |

The two are level on this test. That measures tinygrad's untuned matmul on both
devices, not what either device can do. No language model was run through tinygrad.

The limit found: the first start after the dock is powered on works. Every later
start fails in the driver's card initialisation with

```
AssertionError: Must be table pt=0x0, pt.lv=0 pte_idx=0 pt.entry(pte_idx)=0xffffffffffffffff
```

while the Thunderbolt link stays up at 80 Gb/s. Seen on two separate power
cycles. Forcing the driver's own full-reset path before initialisation (the
branch it takes when it finds the card already booted) did not change the
outcome. Power cycling the dock restores one more start.

## Host 2, second driver: macuda

[macuda](https://github.com/Davinchy/macuda) (commit `e797e31`) is a C user-space driver for the same TinyGPU
extension, with a CUDA runtime layer and llama.cpp's CUDA backend built against it. The dock was plugged directly
into the MacBook. Built with the same `cuda-nvcc 12.8` container; one local change was needed under Colima, which
shares only the home folder with the container (the kernel build's output folder had to move out of `/tmp`).

Model: `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`, 8,540,775,840 bytes, all layers on the card. Command:
`sh tools/nv_shim_step.sh A bench <model> 128`, which runs `llama-bench -ngl 99 -p 256 -n 128 -r 3`.

| Start | Decode (tok/s) | Prompt (tok/s) |
|---|---|---|
| Cold, after a dock power cycle | 158.6 | about 9,300 |
| Warm, second start | 158.0 | about 9,300 |
| Warm, third start | 158.3 | about 9,300 |

Plain greedy generation of 64 tokens (`llama-simple`) ran at 135.8 tok/s. macuda's README lists 161.0 ± 1.0 tok/s for the same model file on its
author's setup. Its health check (`tools/preflight.sh`) reported OK after every run. Unlike
tinygrad 0.14.0, repeated starts worked without power cycling the dock.

## A TensorFold kernel on the card

TensorFold's CUDA backend talks to the CUDA driver API (77 `cu*` functions), which macuda does not export; it
exports the `cuda*` runtime functions llama.cpp uses. A small translation library on top of macuda's driver
(about 330 lines of C, a Mach-O library under the file name `libcuda.so.1`) now exports all 77. 58 do real work;
19 refuse by name (15 graph functions, 3 for mapping pinned host memory, 1 for event timing).

Through it, the `tf_l2_rows` kernel from the TensorFold 1.3.5 Linux release image (unchanged bytes, the sm_120
entry of its five-architecture `norms` image) ran on the 5090: 10 shapes, 367,936 bf16 output values, all equal
bit for bit to a host model that repeats the kernel's fp32 operation order. This is a focused test program, not
TensorFold's kernel contract and not TensorFold itself, and it says nothing about speed.

## Not done

- No TensorFold model has run on the 5090. TensorFold has no macOS CUDA build yet.
- No model has run on the 5090 from the HX 470.
- OCuLink is untested on the HX 470.
- The macuda figures are one evening, one model, one cable. None of this has been repeated on another day.
