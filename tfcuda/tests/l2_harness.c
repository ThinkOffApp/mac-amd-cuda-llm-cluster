// l2_harness.c: TensorFold's tf_l2_rows kernel through the driver library, checked bit for bit against a host model.
// It opens the library the way TensorFold does (by path, resolving all 77 names, refusing if one is missing), follows
// TensorFold's call order for the l2 path, and compares every output bf16 with a host computation that repeats the
// kernel's exact fp32 operation order (zig/kernels/cuda/norms_l2.cuh at v1.3.5). This is NOT TensorFold's own
// kernel-contract gate; it is a focused probe of the library with TensorFold's released kernel bytes.
//   l2_harness <path to libcuda.so.1> <norms fatbin>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *NAMES[77] = {
  "cuInit","cuDriverGetVersion","cuDeviceGet","cuDeviceGetCount","cuDeviceGetUuid_v2","cuDeviceGetName","cuDeviceGetAttribute",
  "cuDeviceTotalMem_v2","cuDevicePrimaryCtxRetain","cuDevicePrimaryCtxSetFlags_v2","cuDevicePrimaryCtxRelease_v2","cuCtxSetCurrent",
  "cuCtxGetCurrent","cuCtxGetDevice","cuCtxSynchronize","cuMemGetInfo_v2","cuMemAlloc_v2","cuMemFree_v2","cuMemHostAlloc",
  "cuMemHostGetDevicePointer_v2","cuMemFreeHost","cuMemHostRegister_v2","cuMemHostUnregister","cuMemcpyHtoD_v2","cuMemcpyDtoH_v2",
  "cuMemcpyDtoD_v2","cuMemcpyHtoDAsync_v2","cuMemcpyDtoHAsync_v2","cuMemcpyDtoDAsync_v2","cuMemcpy2DAsync_v2","cuMemsetD8_v2",
  "cuMemsetD32_v2","cuMemsetD8Async","cuMemsetD32Async","cuStreamCreate","cuStreamCreateWithPriority","cuCtxGetStreamPriorityRange",
  "cuStreamDestroy_v2","cuStreamSynchronize","cuStreamWaitEvent","cuStreamQuery","cuEventCreate","cuEventDestroy_v2","cuEventRecord",
  "cuEventRecordWithFlags","cuEventSynchronize","cuEventQuery","cuEventElapsedTime","cuModuleLoadData","cuModuleLoadDataEx",
  "cuModuleUnload","cuModuleGetFunction","cuModuleGetGlobal_v2","cuFuncGetAttribute","cuFuncSetAttribute","cuFuncSetCacheConfig",
  "cuOccupancyMaxActiveBlocksPerMultiprocessor","cuLaunchKernel","cuLaunchKernelEx","cuStreamBeginCapture_v2","cuStreamEndCapture",
  "cuStreamIsCapturing","cuGraphCreate","cuGraphDestroy","cuGraphAddKernelNode_v2","cuGraphKernelNodeGetParams_v2",
  "cuGraphKernelNodeSetParams_v2","cuGraphAddDependencies","cuGraphGetNodes","cuGraphInstantiateWithFlags","cuGraphUpload",
  "cuGraphLaunch","cuGraphExecDestroy","cuGraphExecUpdate_v2","cuGraphExecKernelNodeSetParams_v2","cuGetErrorName","cuGetErrorString" };

static void *lib;
#define FN(ret, name, ...) ((ret (*)(__VA_ARGS__))dlsym(lib, name))
static int fails;
static int ck(int r, const char *what) {
  if (r == 0) return 0;
  const char *n = NULL, *s = NULL;
  FN(int, "cuGetErrorName", int, const char **)(r, &n); FN(int, "cuGetErrorString", int, const char **)(r, &s);
  printf("FAIL %s: %d %s (%s)\n", what, r, n ? n : "?", s ? s : "?"); fails++; return r;
}

// bf16 <-> fp32, round to nearest even (finite inputs only)
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bfr(float f) { return bf2f(f2bf(f)); }

// One row exactly as the kernel computes it: 256 threads, thread t owns 8-value chunks t, t+256, t+512, t+768.
static void ref_row(const uint16_t *x, const uint16_t *r, uint16_t *y, int width, float eps) {
  static float v[8192]; const int chunks = width / 8; float part[256];
  for (int i = 0; i < width; i++) { float a = bf2f(x[i]); v[i] = r ? bfr(a + bf2f(r[i])) : a; }
  for (int t = 0; t < 256; t++) {
    float p = 0.f;
    for (int j = 0; j < 4; j++) { int c = t + j * 256; for (int i = 0; i < 8; i++) { float e = c < chunks ? v[c * 8 + i] : 0.f; p = fmaf(e, e, p); } }
    part[t] = p;
  }
  float warps[8];
  for (int w = 0; w < 8; w++) {
    float a[32], b[32]; memcpy(a, part + w * 32, sizeof a);
    for (int m = 16; m > 0; m >>= 1) { for (int l = 0; l < 32; l++) b[l] = a[l] + a[l ^ m]; memcpy(a, b, sizeof a); }
    warps[w] = a[0];
  }
  float a[32] = {0}, b[32]; memcpy(a, warps, sizeof warps);
  for (int m = 4; m > 0; m >>= 1) { for (int l = 0; l < 32; l++) b[l] = a[l] + a[l ^ m]; memcpy(a, b, sizeof a); }
  const float inv = 1.f / sqrtf(a[0] + eps);
  for (int i = 0; i < width; i++) y[i] = f2bf(v[i] * inv);
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }
static uint16_t rnd_bf(float scale) { float u = (float)(next() >> 8) / 8388608.f - 1.f; return f2bf(u * scale); }

int main(int argc, char **argv) {
  if (argc < 3) { printf("usage: %s <libcuda.so.1 path> <norms fatbin>\n", argv[0]); return 2; }
  lib = dlopen(argv[1], RTLD_NOW); if (!lib) { printf("cannot open %s: %s\n", argv[1], dlerror()); return 2; }
  int missing = 0; for (int i = 0; i < 77; i++) if (!dlsym(lib, NAMES[i])) { printf("missing symbol %s\n", NAMES[i]); missing++; }
  printf("driver symbols resolved: %d of 77\n", 77 - missing); if (missing) return 2;

  FILE *fp = fopen(argv[2], "rb"); if (!fp) { printf("cannot read %s\n", argv[2]); return 2; }
  fseek(fp, 0, SEEK_END); long isz = ftell(fp); fseek(fp, 0, SEEK_SET);
  void *image; if (posix_memalign(&image, 16, (size_t)isz)) return 2;
  if (fread(image, 1, (size_t)isz, fp) != (size_t)isz) return 2; fclose(fp);

  if (ck(FN(int, "cuInit", unsigned)(0), "cuInit")) return 1;
  int ver = 0, dev = -1, maj = 0, min = 0, integ = -1; char name[128] = ""; void *ctx = NULL, *stream = NULL, *mod = NULL, *fn = NULL;
  ck(FN(int, "cuDriverGetVersion", int *)(&ver), "cuDriverGetVersion");
  ck(FN(int, "cuDeviceGet", int *, int)(&dev, 0), "cuDeviceGet");
  ck(FN(int, "cuDeviceGetAttribute", int *, int, int)(&maj, 75, dev), "attr 75");
  ck(FN(int, "cuDeviceGetAttribute", int *, int, int)(&min, 76, dev), "attr 76");
  ck(FN(int, "cuDeviceGetAttribute", int *, int, int)(&integ, 18, dev), "attr 18");
  ck(FN(int, "cuDeviceGetName", char *, int, int)(name, sizeof name, dev), "cuDeviceGetName");
  printf("device: %s, sm_%d%d, driver version %d, integrated %d\n", name, maj, min, ver, integ);
  ck(FN(int, "cuDevicePrimaryCtxRetain", void **, int)(&ctx, dev), "cuDevicePrimaryCtxRetain");
  ck(FN(int, "cuCtxSetCurrent", void *)(ctx), "cuCtxSetCurrent");
  ck(FN(int, "cuStreamCreate", void **, unsigned)(&stream, 0), "cuStreamCreate");
  if (ck(FN(int, "cuModuleLoadData", void **, const void *)(&mod, image), "cuModuleLoadData")) goto down;
  // every kernel name cuda/norms.zig resolves from this image has to be there, as in Norms.load
  static const char *all[] = { "tf_l2_rows", "tf_slerp_rows", "tf_add_rms_rows_bf16h", "tf_add_rms_rows_bf16h_f32w", "tf_gated_rms_rows_64",
                               "tf_gated_rms_rows_128", "tf_gated_rms_rows_256", "tf_add_rms_rows_bf16h_tail", "tf_add_rms_rows_bf16h_tail_round" };
  for (unsigned i = 0; i < sizeof all / sizeof *all; i++) { void *f = NULL; ck(FN(int, "cuModuleGetFunction", void **, void *, const char *)(&f, mod, all[i]), all[i]); if (i == 0) fn = f; }
  if (!fn) goto down;
  { int a = 0;
    static const int attrs[] = { 0, 1, 4, 6 }; static const char *an[] = { "max threads", "static shared bytes", "registers", "binary version" };
    for (int i = 0; i < 4; i++) if (!ck(FN(int, "cuFuncGetAttribute", int *, int, void *)(&a, attrs[i], fn), an[i])) printf("tf_l2_rows %s: %d\n", an[i], a); }

  static const struct { int rows, width, add; float scale; } cases[] = {
    { 16, 2560, 0, 1.f }, { 3, 2560, 0, 1.f }, { 16, 2560, 1, 1.f }, { 3, 2560, 1, 1.f },   // the served shapes
    { 1, 64, 0, 1.f }, { 5, 64, 1, 3.f }, { 7, 8192, 0, 0.02f }, { 2, 8192, 1, 40.f }, { 64, 2048, 0, 1e-3f }, { 33, 1984, 1, 1.f } };
  const float eps = 1e-6f; const size_t guard = 256;
  long total = 0, wrong = 0;
  for (unsigned c = 0; c < sizeof cases / sizeof *cases; c++) {
    const int rows = cases[c].rows, width = cases[c].width; const size_t n = (size_t)rows * width, bytes = n * 2;
    uint16_t *x = malloc(bytes), *r = malloc(bytes), *want = malloc(bytes), *got = malloc(bytes + guard);
    for (size_t i = 0; i < n; i++) { x[i] = rnd_bf(cases[c].scale); r[i] = rnd_bf(cases[c].scale); }
    for (int i = 0; i < rows; i++) ref_row(x + (size_t)i * width, cases[c].add ? r + (size_t)i * width : NULL, want + (size_t)i * width, width, eps);
    uint64_t dx = 0, dr = 0, dy = 0; memset(got, 0xA5, bytes + guard);
    int bad = 0;
    bad |= ck(FN(int, "cuMemAlloc_v2", uint64_t *, size_t)(&dx, bytes), "alloc x");
    bad |= ck(FN(int, "cuMemAlloc_v2", uint64_t *, size_t)(&dr, bytes), "alloc r");
    bad |= ck(FN(int, "cuMemAlloc_v2", uint64_t *, size_t)(&dy, bytes + guard), "alloc y");
    if (!bad) {
      bad |= ck(FN(int, "cuMemcpyHtoD_v2", uint64_t, const void *, size_t)(dx, x, bytes), "upload x");
      bad |= ck(FN(int, "cuMemcpyHtoD_v2", uint64_t, const void *, size_t)(dr, r, bytes), "upload r");
      bad |= ck(FN(int, "cuMemcpyHtoD_v2", uint64_t, const void *, size_t)(dy, got, bytes + guard), "upload y fill");
      bad |= ck(FN(int, "cuStreamSynchronize", void *)(NULL), "sync default");
      uint64_t pr = cases[c].add ? dr : 0; int w = width; float e = eps; void *params[5] = { &dx, &pr, &dy, &w, &e };
      if (!bad) bad |= ck(FN(int, "cuLaunchKernel", void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **)
                          (fn, (unsigned)rows, 1, 1, 256, 1, 1, 0, stream, params, NULL), "cuLaunchKernel");
      if (!bad) bad |= ck(FN(int, "cuStreamSynchronize", void *)(stream), "sync stream");
      memset(got, 0, bytes + guard);
      if (!bad) bad |= ck(FN(int, "cuMemcpyDtoH_v2", void *, uint64_t, size_t)(got, dy, bytes + guard), "download y");
    }
    long w0 = 0, first = -1; int guard_ok = 1;
    if (!bad) {
      for (size_t i = 0; i < n; i++) if (got[i] != want[i]) { if (first < 0) first = (long)i; w0++; }
      for (size_t i = 0; i < guard; i++) if (((unsigned char *)got)[bytes + i] != 0xA5) guard_ok = 0;
      total += (long)n; wrong += w0; if (w0 || !guard_ok) fails++;
    }
    printf("case %u: rows %d width %d %s scale %g: %s", c, rows, width, cases[c].add ? "with add" : "plain", cases[c].scale,
           bad ? "DID NOT RUN" : w0 ? "MISMATCH" : "bit-exact");
    if (!bad) { printf(" (%ld of %zu values differ, guard %s", w0, n, guard_ok ? "intact" : "OVERWRITTEN");
                if (first >= 0) printf(", first at %ld: got %#06x want %#06x", first, got[first], want[first]); printf(")"); }
    printf("\n");
    if (dx) ck(FN(int, "cuMemFree_v2", uint64_t)(dx), "free x");
    if (dr) ck(FN(int, "cuMemFree_v2", uint64_t)(dr), "free r");
    if (dy) ck(FN(int, "cuMemFree_v2", uint64_t)(dy), "free y");
    free(x); free(r); free(want); free(got);
  }
  printf("values compared: %ld, differing: %ld\n", total, wrong);
down:
  if (mod) ck(FN(int, "cuModuleUnload", void *)(mod), "cuModuleUnload");
  if (stream) { ck(FN(int, "cuStreamSynchronize", void *)(stream), "final sync"); ck(FN(int, "cuStreamDestroy_v2", void *)(stream), "cuStreamDestroy"); }
  ck(FN(int, "cuCtxSynchronize", void)(), "cuCtxSynchronize");
  ck(FN(int, "cuDevicePrimaryCtxRelease_v2", int)(dev), "cuDevicePrimaryCtxRelease");
  printf("RESULT: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
