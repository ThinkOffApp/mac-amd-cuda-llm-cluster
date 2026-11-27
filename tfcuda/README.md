# tfcuda

A CUDA driver API library for macOS, written for the TensorFold inference engine.

`tfcuda.c` implements the 77 `cu*` driver functions that TensorFold resolves, on top of `libtinynv`, the user-space
NVIDIA driver from [macuda](https://github.com/Davinchy/macuda). It is built as a Mach-O dylib whose file name is
`libcuda.so.1`, because that is the name TensorFold opens on every non-Windows OS, so TensorFold's existing loader
finds it through `DYLD_LIBRARY_PATH`. The library also exports one extra function, `cuFuncGetName`, which TensorFold
does not resolve; the build therefore reports 78 exported `cu*` symbols.

Rule followed throughout: a call the library cannot do truthfully is refused by name (`CUDA_ERROR_NOT_SUPPORTED` and a
line on stderr starting `[tfcuda] refused`). It never answers with a made-up success.

## Status

Works with TensorFold's CUDA server built for macOS (see "Use with TensorFold").

Of the 77 functions, 73 do real work. Graphs are implemented by record and replay: a captured or explicitly built
graph is a list of operations replayed in dependency order, so results equal eager execution, without the launch
overhead saving of a real CUDA graph.

These 4 always refuse with `CUDA_ERROR_NOT_SUPPORTED`:

| Function | Reason |
|---|---|
| `cuMemHostRegister_v2` | host pages cannot be pinned or mapped through libtinynv |
| `cuMemHostUnregister` | nothing can have been registered |
| `cuMemHostGetDevicePointer_v2` | host pages are not mapped into the card's address space |
| `cuEventElapsedTime` | libtinynv takes no timestamp at an event's record point |

Partial support inside working functions (each refused case returns `CUDA_ERROR_NOT_SUPPORTED`):

- `cuLaunchKernelEx`: a cluster dimension larger than one block is refused (libtinynv has no cluster launch); a
  cooperative launch is refused. A cluster of one block runs as an ordinary launch.
- `cuLaunchKernel` and graph kernel nodes: the `extra` argument form is refused.
- `cuDeviceGetAttribute`: answers attributes 1, 10, 16, 18, 75, 76, 81, 97, 102, 110 and refuses the rest. Attribute
  16 (SM count) is libtinynv's upper bound, not the enabled count. 102 and 110 answer 0.
- `cuDeviceGetUuid_v2`: returns the fixed value `tfcuda-macuda-00`, not the card's UUID.
- `cuFuncGetAttribute`: attributes 0 to 6, 8 (once set) and 14; some only when the kernel image reports them.
  `cuFuncSetAttribute`: attributes 8 and 14 only.
- `cuMemHostAlloc`: `CU_MEMHOSTALLOC_DEVICEMAP` is refused. The memory is page aligned but not DMA pinned.
- `cuMemsetD32_v2` / `cuMemsetD32Async`: only patterns whose four bytes are equal.
- `cuMemcpyDtoHAsync_v2` and `cuMemcpy2DAsync_v2` inside a stream capture are refused; `cuMemcpy2DAsync_v2` handles
  host and device memory types only.
- `cuGraphLaunch` inside a stream capture (child graphs) is refused. `cuGraphExecKernelNodeSetParams_v2` refuses a
  change of function.
- `cuModuleLoadData` / `cuModuleLoadDataEx`: an nvcc fatbin (the entry matching the card's SM, zstd-compressed
  entries included) or a bare ELF cubin. PTX is not compiled. `cuModuleLoadDataEx` ignores its options.
- `cuStreamCreateWithPriority`: priority ignored; `cuCtxGetStreamPriorityRange` reports 0 to 0.
- `cuFuncSetCacheConfig` and `cuGraphUpload` are accepted and do nothing.
- `cuOccupancyMaxActiveBlocksPerMultiprocessor` uses macuda's conservative model (registers, threads, shared memory),
  not NVIDIA's tables.
- Only device 0, one primary context. `cuDriverGetVersion` reports 13000 (a subset of the CUDA 13.0 driver API).
- Async host-to-device copies stage the host data before returning, and device-to-host copies wait for themselves,
  so both are stricter than CUDA's async semantics.

Limits:

- No cluster launch (see above).
- One process per card.
- All streams share one in-order queue: no concurrency between streams, no priorities.
- Dynamic shared memory: the opt-in ceiling is the per-SM shared memory less 1 KiB, near 99 KiB per block on the card
  tested.
- No PTX JIT: kernel images must contain SASS for the card.

Tested only on: one MacBook Pro M5 Max with one RTX 5090 in a Thunderbolt 5 dock, macOS 27.0.1, macuda at commit
`e797e31`.

## Build

Requirements: Xcode or the Command Line Tools, zstd (`brew install zstd`), and a macuda checkout whose driver is
built (`sh setup.sh shim` inside macuda; see macuda's README). Building does not touch the GPU.

```sh
sh build.sh                         # macuda expected at ../macuda
MACUDA=/path/to/macuda sh build.sh  # or name it
ZSTD=/path/to/zstd/prefix sh build.sh   # if zstd is not where `brew --prefix zstd` says
```

Output in `out/`: `libcuda.so.1` (the library), `l2_harness`, `graph_harness` (tests). The script uses macuda's
`cuda-shim/build/sdk.sh` to pick the macOS SDK, and links macuda's `libtinynv.a`, its `copy1d_cubin.o` and
`keepalive_cubin.o`, and `libzstd.a` statically. Only `cu*` symbols are exported (`-Wl,-exported_symbol,_cu*`).
The last line printed is `exported cu* symbols: N` (78 for this source).

## Use with TensorFold

1. TensorFold's build refuses CUDA servers for macOS targets. One line in `zig/build/product.zig` (the CUDA branch of
   `refusal`) must also accept `.macos`:

   ```zig
   if (os != .linux and os != .windows and os != .macos) return b.fmt("-Dgpu={s}: CUDA servers build for Linux, Windows and macOS targets", .{name});
   ```

2. Build the CUDA server with `-Dgpu=sm_120 -Dfatbins=DIR`. The kernel images in `DIR` must be produced by NVIDIA's
   `nvcc` on Linux (macOS has no nvcc, and this library compiles no PTX).
3. Run it with this library first on the loader path, the device-profile check off, and macuda's driver pointed at
   the card (`TINYNV_SOCKET`, see macuda):

   ```sh
   DYLD_LIBRARY_PATH=/path/to/tfcuda/out TENSORFOLD_DEVICE_CHECK=0 ./tensorfold ...
   ```

   Without `TINYNV_SOCKET`, libtinynv runs on its null device; `cuInit` detects that and returns
   `CUDA_ERROR_NO_DEVICE` rather than pretend.

## Environment variables

Read by this library (`getenv` in `tfcuda.c`). All are off unless set; any value counts as set.

| Variable | Effect |
|---|---|
| `TFCUDA_STATS` | Count each kind of call (launches, copies, fills, syncs, event records, queries, waits, graph launches) with bytes, and print the counts to stderr at exit. |
| `TFCUDA_TRACE` | Print one stderr line per cluster launch request (grid, block, cluster shape, shared memory, registers). |
| `TFCUDA_NO_COPY_KERNEL` | Do not lend macuda's copy kernel to the driver; device-to-device copies then stay on the copy engine. For measuring the difference. |
| `TFCUDA_NO_KEEPALIVE_KERNEL` | Do not register macuda's keep-alive kernel. |
| `TFCUDA_GRAPH_RESIDENT` | Experimental. Together with libtinynv's `TINYNV_GRAPH_RESIDENT=1`, an executable graph made only of non-cluster kernel launches is recorded once into a chain on the card and later launches run that chain as one batch. |

Read by macuda's libtinynv, not by this library, and relevant here: `TINYNV_SOCKET` (which TinyGPU server to use),
`TINYNV_KEEPALIVE` (launch the registered keep-alive kernel after a synchronisation), `TINYNV_DOWNLOAD_VIA_COMPUTE`
(serve downloads with the registered copy kernel), `TINYNV_GRAPH_RESIDENT`. See macuda's documentation.

## Measured

Inkling 2 Mini, text, one stream, greedy decoding, 36-token prompt, 200 generated tokens, on the machine listed under
Status: 75 tok/s with TensorFold 1.3.6 and with TensorFold main.

## Known problems

- TensorFold's kernel contract run stopped on an MMU fault right after cluster-launch refusals. The cause is not
  known; under investigation.
- TensorFold's host memory-pressure governor can serialise streams on a busy Mac. For testing,
  `TENSORFOLD_MEMORY_PRESSURE=0` turns the governor off.
- `tests/graph_harness.c` starts with a copy of the header comment of `l2_harness.c`; see `tests/README.md` for what
  it actually does.

## Credits

- [macuda](https://github.com/Davinchy/macuda) by Antonio Torres ([@Davinchy](https://github.com/Davinchy)): the user-space NVIDIA driver `libtinynv`, its copy
  and keep-alive kernels, SDK selection and the shared-memory and occupancy rules this library follows. MIT licence.
- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([@ashhart](https://github.com/ashhart)): the inference engine whose driver API surface this
  library implements, and whose kernels the tests use.

## Licence

MIT, see `LICENSE`. Third-party notices for binary distributions are in `NOTICE`.
