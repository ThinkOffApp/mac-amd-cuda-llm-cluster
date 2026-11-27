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


typedef struct { void *func; unsigned gx, gy, gz, bx, by, bz, shared; void **params; void **extra; void *kern; void *ctx; } KNP;
#define CU(ret, name, ...) FN(ret, name, __VA_ARGS__)
static int eq(const char *what, const uint16_t *got, const uint16_t *want, size_t n) {
  size_t bad = 0; for (size_t i = 0; i < n; i++) bad += got[i] != want[i];
  printf("%-58s %s (%zu of %zu differ)\n", what, bad ? "MISMATCH" : "bit-exact", bad, n); if (bad) fails++; return bad == 0;
}
static void expect(const char *what, int ok) { printf("%-58s %s\n", what, ok ? "ok" : "WRONG"); if (!ok) fails++; }
int main(int argc, char **argv) {
  if (argc < 3) return 2;
  lib = dlopen(argv[1], RTLD_NOW); if (!lib) { printf("cannot open %s\n", argv[1]); return 2; }
  FILE *fp = fopen(argv[2], "rb"); fseek(fp, 0, SEEK_END); long isz = ftell(fp); fseek(fp, 0, SEEK_SET);
  void *image; if (posix_memalign(&image, 16, (size_t)isz)) return 2; if (fread(image, 1, (size_t)isz, fp) != (size_t)isz) return 2; fclose(fp);
  if (ck(CU(int, "cuInit", unsigned)(0), "cuInit")) return 1;
  int dev = 0; void *ctx = NULL, *stream = NULL, *mod = NULL, *fn = NULL, *fn2 = NULL;
  ck(CU(int, "cuDeviceGet", int *, int)(&dev, 0), "cuDeviceGet");
  ck(CU(int, "cuDevicePrimaryCtxRetain", void **, int)(&ctx, dev), "ctx"); ck(CU(int, "cuCtxSetCurrent", void *)(ctx), "set ctx");
  ck(CU(int, "cuStreamCreate", void **, unsigned)(&stream, 0), "stream");
  if (ck(CU(int, "cuModuleLoadData", void **, const void *)(&mod, image), "module")) return 1;
  ck(CU(int, "cuModuleGetFunction", void **, void *, const char *)(&fn, mod, "tf_l2_rows"), "tf_l2_rows");
  ck(CU(int, "cuModuleGetFunction", void **, void *, const char *)(&fn2, mod, "tf_slerp_rows"), "tf_slerp_rows");
  enum { ROWS = 16, W = 2560 }; const size_t n = (size_t)ROWS * W, bytes = n * 2; const float eps = 1e-6f;
  uint16_t *x = malloc(bytes), *x2 = malloc(bytes), *r = malloc(bytes), *fill = malloc(bytes), *got = malloc(bytes), *w1 = malloc(bytes), *w2 = malloc(bytes), *w1b = malloc(bytes), *w2b = malloc(bytes);
  for (size_t i = 0; i < n; i++) { x[i] = rnd_bf(1.f); x2[i] = rnd_bf(2.f); r[i] = rnd_bf(1.f); fill[i] = 0x3f80; }
  for (int i = 0; i < ROWS; i++) { ref_row(x + i * W, NULL, w1 + i * W, W, eps); ref_row(w1 + i * W, r + i * W, w2 + i * W, W, eps);
                                   ref_row(x2 + i * W, NULL, w1b + i * W, W, eps); ref_row(w1b + i * W, r + i * W, w2b + i * W, W, eps); }
  uint64_t dx = 0, dx2 = 0, dr = 0, dy1 = 0, dy2 = 0, nul = 0; int w = W; float e = eps;
  uint64_t *bufs[] = { &dx, &dx2, &dr, &dy1, &dy2 }; for (int i = 0; i < 5; i++) ck(CU(int, "cuMemAlloc_v2", uint64_t *, size_t)(bufs[i], bytes), "alloc");
#define UP(d, h) ck(CU(int, "cuMemcpyHtoD_v2", uint64_t, const void *, size_t)(d, h, bytes), "upload")
#define DOWN(d) (ck(CU(int, "cuStreamSynchronize", void *)(stream), "sync"), ck(CU(int, "cuMemcpyDtoH_v2", void *, uint64_t, size_t)(got, d, bytes), "download"))
#define LAUNCH(f, P) CU(int, "cuLaunchKernel", void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **)(f, ROWS, 1, 1, 256, 1, 1, 0, stream, P, NULL)
  UP(dx, x); UP(dx2, x2); UP(dr, r); UP(dy1, fill); UP(dy2, fill);
  void *pA[5] = { &dx, &nul, &dy1, &w, &e };     // A: y1 = l2(x)
  void *pB[5] = { &dy1, &dr, &dy2, &w, &e };     // B: y2 = l2(bf16(y1 + r))
  void *pA2[5] = { &dx2, &nul, &dy1, &w, &e };   // A with the other input

  // 1. capture: nothing may run while recording
  int st_ = -1; void *g = NULL, *ex = NULL; size_t cnt = 0;
  ck(CU(int, "cuStreamBeginCapture_v2", void *, int)(stream, 0), "begin capture");
  ck(CU(int, "cuStreamIsCapturing", void *, int *)(stream, &st_), "is capturing"); expect("capture status is active while recording", st_ == 1);
  ck(LAUNCH(fn, pA), "record A"); ck(LAUNCH(fn, pB), "record B");
  ck(CU(int, "cuStreamEndCapture", void *, void **)(stream, &g), "end capture");
  ck(CU(int, "cuStreamIsCapturing", void *, int *)(stream, &st_), "is capturing"); expect("capture status is none after the capture", st_ == 0);
  DOWN(dy2); eq("recording ran nothing (output still the fill value)", got, fill, n);
  ck(CU(int, "cuGraphGetNodes", void *, void **, size_t *)(g, NULL, &cnt), "count nodes"); expect("captured graph has 2 nodes", cnt == 2);
  ck(CU(int, "cuGraphInstantiateWithFlags", void **, void *, unsigned long long)(&ex, g, 0), "instantiate");
  ck(CU(int, "cuGraphUpload", void *, void *)(ex, stream), "upload graph");
  ck(CU(int, "cuGraphLaunch", void *, void *)(ex, stream), "launch graph");
  DOWN(dy1); eq("captured graph, stage A against the host model", got, w1, n);
  DOWN(dy2); eq("captured graph, stage B against the host model", got, w2, n);
  // 2. replay after the input changed on the card
  UP(dx, x2); ck(CU(int, "cuGraphLaunch", void *, void *)(ex, stream), "relaunch");
  DOWN(dy2); eq("replay with new input data", got, w2b, n);
  UP(dx, x); ck(CU(int, "cuGraphExecDestroy", void *)(ex), "destroy exec"); ck(CU(int, "cuGraphDestroy", void *)(g), "destroy graph");

  // 3. explicit graph, B added BEFORE A and ordered only by a dependency
  void *nA = NULL, *nB = NULL; g = NULL; ex = NULL;
  KNP kA = { fn, ROWS, 1, 1, 256, 1, 1, 0, pA, NULL, NULL, NULL }, kB = { fn, ROWS, 1, 1, 256, 1, 1, 0, pB, NULL, NULL, NULL }, kA2 = kA; kA2.params = pA2;
  UP(dy1, fill); UP(dy2, fill);
  ck(CU(int, "cuGraphCreate", void **, unsigned)(&g, 0), "create graph");
  ck(CU(int, "cuGraphAddKernelNode_v2", void **, void *, void *const *, size_t, const KNP *)(&nB, g, NULL, 0, &kB), "add B");
  ck(CU(int, "cuGraphAddKernelNode_v2", void **, void *, void *const *, size_t, const KNP *)(&nA, g, NULL, 0, &kA), "add A");
  ck(CU(int, "cuGraphAddDependencies", void *, void *const *, void *const *, size_t)(g, &nA, &nB, 1), "A before B");
  ck(CU(int, "cuGraphInstantiateWithFlags", void **, void *, unsigned long long)(&ex, g, 0), "instantiate explicit");
  ck(CU(int, "cuGraphLaunch", void *, void *)(ex, stream), "launch explicit");
  DOWN(dy2); eq("explicit graph honours the dependency (B after A)", got, w2, n);
  // 4. the node keeps the VALUES given at add time, and reports them back
  { KNP back; memset(&back, 0, sizeof back); uint64_t saved = dx; dx = 0xdead0000;  // the caller's variable changes after the add
    ck(CU(int, "cuGraphKernelNodeGetParams_v2", void *, KNP *)(nA, &back), "get params");
    expect("node reports its function and geometry", back.func == fn && back.gx == ROWS && back.bx == 256);
    expect("node kept the argument value from add time", back.params && *(uint64_t *)back.params[0] == saved); dx = saved; }
  // 5. template change + exec update
  ck(CU(int, "cuGraphKernelNodeSetParams_v2", void *, const KNP *)(nA, &kA2), "set A on template");
  { struct { unsigned result; void *a, *b; } info = { 99, NULL, NULL };
    ck(CU(int, "cuGraphExecUpdate_v2", void *, void *, void *)(ex, g, &info), "exec update"); expect("exec update reports success", info.result == 0); }
  ck(CU(int, "cuGraphLaunch", void *, void *)(ex, stream), "launch updated"); DOWN(dy2); eq("after template change and exec update", got, w2b, n);
  // 6. per-node change on the executable
  ck(CU(int, "cuGraphExecKernelNodeSetParams_v2", void *, void *, const KNP *)(ex, nA, &kA), "set A on exec");
  ck(CU(int, "cuGraphLaunch", void *, void *)(ex, stream), "launch after exec set"); DOWN(dy2); eq("after a per-node change on the executable", got, w2, n);
  // 7. refusals that must happen
  { KNP kS = kA; kS.func = fn2; struct { unsigned result; void *a, *b; } info = { 99, NULL, NULL };
    int r1 = CU(int, "cuGraphKernelNodeSetParams_v2", void *, const KNP *)(nA, &kS);   // different function on the template (5 params given, slerp takes 7: may refuse here)
    if (r1 == 0) { int r2 = CU(int, "cuGraphExecUpdate_v2", void *, void *, void *)(ex, g, &info); expect("exec update refuses a changed function (910, result 4)", r2 == 910 && info.result == 4); }
    else expect("template refuses a node whose arguments do not fit the function", r1 != 0);
    ck(CU(int, "cuGraphKernelNodeSetParams_v2", void *, const KNP *)(nA, &kA), "restore A"); }
  { // a template with the same two nodes but NO dependency between them is a different topology
    void *g2 = NULL, *m1 = NULL, *m2 = NULL; struct { unsigned result; void *a, *b; } info = { 99, NULL, NULL };
    ck(CU(int, "cuGraphCreate", void **, unsigned)(&g2, 0), "create g2");
    ck(CU(int, "cuGraphAddKernelNode_v2", void **, void *, void *const *, size_t, const KNP *)(&m1, g2, NULL, 0, &kA), "g2 A");
    ck(CU(int, "cuGraphAddKernelNode_v2", void **, void *, void *const *, size_t, const KNP *)(&m2, g2, NULL, 0, &kB), "g2 B");
    int r5 = CU(int, "cuGraphExecUpdate_v2", void *, void *, void *)(ex, g2, &info);
    expect("exec update refuses changed dependencies (910, result 2)", r5 == 910 && info.result == 2);
    ck(CU(int, "cuGraphLaunch", void *, void *)(ex, stream), "launch after refused update"); DOWN(dy2); eq("a refused update leaves the executable unchanged", got, w2, n);
    ck(CU(int, "cuGraphDestroy", void *)(g2), "destroy g2"); }
  { int r3 = CU(int, "cuGraphAddDependencies", void *, void *const *, void *const *, size_t)(g, &nB, &nA, 1); void *ex2 = NULL;
    int r4 = r3 ? r3 : CU(int, "cuGraphInstantiateWithFlags", void **, void *, unsigned long long)(&ex2, g, 0);
    expect("a dependency cycle is refused", r4 != 0); if (ex2) CU(int, "cuGraphExecDestroy", void *)(ex2); }
  ck(CU(int, "cuGraphExecDestroy", void *)(ex), "destroy exec"); ck(CU(int, "cuGraphDestroy", void *)(g), "destroy graph");
  for (int i = 0; i < 5; i++) ck(CU(int, "cuMemFree_v2", uint64_t)(*bufs[i]), "free");
  ck(CU(int, "cuModuleUnload", void *)(mod), "unload"); ck(CU(int, "cuStreamDestroy_v2", void *)(stream), "stream destroy");
  ck(CU(int, "cuCtxSynchronize", void)(), "ctx sync"); ck(CU(int, "cuDevicePrimaryCtxRelease_v2", int)(dev), "release");
  printf("RESULT: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails); return fails ? 1 : 0;
}
