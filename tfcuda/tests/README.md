# Tests

Both harnesses need the GPU (an NVIDIA card reached through macuda's driver, `TINYNV_SOCKET` set). Run only one GPU
job at a time. `sh build.sh` builds them into `out/`.

Both take the library path and a TensorFold norms kernel image (fatbin built for the card's SM by nvcc on Linux):

```sh
out/l2_harness    out/libcuda.so.1 /path/to/norms.fatbin
out/graph_harness out/libcuda.so.1 /path/to/norms.fatbin
```

- `l2_harness.c`: opens the library the way TensorFold does (by path, resolving all 77 names, refusing if one is
  missing), runs TensorFold's `tf_l2_rows` kernel in TensorFold's call order, and compares every output bf16 bit for
  bit with a host model of the kernel's fp32 operation order. It is a focused probe, not TensorFold's own kernel
  contract gate.
- `graph_harness.c`: exercises graphs with `tf_l2_rows` and `tf_slerp_rows`: stream capture, explicitly built nodes,
  executable updates and the refusals. Its first comment lines are a copy of `l2_harness.c`'s header. It does not
  check that the kernel image file exists before reading it.

Both end with `RESULT: PASS` or `RESULT: FAIL (n failures)`. Exit status is 0 on success, 1 on any failure, 2 when the
library or kernel image cannot be loaded.
