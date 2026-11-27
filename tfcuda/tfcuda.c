// tfcuda.c: the CUDA driver API (the 77 cu* entry points TensorFold resolves) on macuda's libtinynv, for macOS.
// Built as a dylib whose FILE is named libcuda.so.1, because that is the name TensorFold opens on every non-Windows OS.
// Rule: a call this library cannot do truthfully is refused by name (CUDA_ERROR_NOT_SUPPORTED and a line on stderr),
// never answered with a made-up success.
#include "tinynv.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#define EXPORT __attribute__((visibility("default")))
typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;

enum { CU_OK = 0, CU_INVALID_VALUE = 1, CU_OOM = 2, CU_NOT_INITIALIZED = 3, CU_NO_DEVICE = 100, CU_INVALID_DEVICE = 101,
       CU_INVALID_IMAGE = 200, CU_INVALID_CONTEXT = 201, CU_NO_BINARY_FOR_GPU = 209, CU_INVALID_HANDLE = 400,
       CU_NOT_FOUND = 500, CU_NOT_READY = 600, CU_LAUNCH_FAILED = 719, CU_NOT_SUPPORTED = 801, CU_UNKNOWN = 999 };

static tinynv_device_t g_dev;
static tinynv_stream_t g_default;
static tinynv_device_props_t g_props;
static int g_inited, g_ctx_refs;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static __thread void *t_ctx;
static int g_ctx_token; // the primary context's address: one device, one context
static __thread char t_err[512];

static int g_ev_lazy = -1;
static int g_graph_resident;
static CUresult refuse(const char *what, const char *fmt, ...) {
  char why[384]; va_list ap; va_start(ap, fmt); vsnprintf(why, sizeof why, fmt, ap); va_end(ap);
  snprintf(t_err, sizeof t_err, "%s: %s", what, why);
  fprintf(stderr, "[tfcuda] refused %s\n", t_err);
  return CU_NOT_SUPPORTED;
}
static CUresult S(const char *what, tinynv_status_t s) {
  if (s == TINYNV_OK) return CU_OK;
  snprintf(t_err, sizeof t_err, "%s: %s (%s)", what, tinynv_status_str(s), tinynv_last_error());
  fprintf(stderr, "[tfcuda] %s\n", t_err);
  switch (s) {
    case TINYNV_ERR_NODEV: return CU_NO_DEVICE;
    case TINYNV_ERR_OOM: return CU_OOM;
    case TINYNV_ERR_LAUNCH: return CU_LAUNCH_FAILED;
    case TINYNV_ERR_INVALID: return CU_INVALID_VALUE;
    case TINYNV_ERR_UNSUPPORTED: return CU_NOT_SUPPORTED;
    default: return CU_UNKNOWN;
  }
}
// TFCUDA_STATS=1: how many times each kind of call was made, printed at exit. Divide by tokens generated to see
// what one token costs in launches, copies and synchronisations.
enum { C_LAUNCH, C_HTOD, C_HTOD_ASYNC, C_DTOD, C_DTOH, C_MEMSET, C_SYNC, C_EV_RECORD, C_EV_QUERY, C_EV_SYNC, C_GRAPH_LAUNCH, C_N };
static const char *const c_names[C_N] = { "kernel launches", "uploads (sync)", "uploads (async)", "device copies", "downloads", "fills", "synchronisations", "event records", "event queries", "event waits", "graph launches" };
static _Atomic unsigned long g_count[C_N]; static _Atomic unsigned long g_bytes[C_N];
// Opt-in counters must not add locked atomics to every production launch/copy.
// Relaxed ordering is enough: these counters observe calls, not memory dependencies.
static int g_stats_enabled;
#define COUNT(i, n) do { if (g_stats_enabled) { \
  atomic_fetch_add_explicit(&g_count[i], 1, memory_order_relaxed); \
  atomic_fetch_add_explicit(&g_bytes[i], (unsigned long)(n), memory_order_relaxed); \
} } while (0)
static void stats_dump(void) { fprintf(stderr, "[tfcuda] call counts:"); for (int i = 0; i < C_N; i++) fprintf(stderr, " %s %lu (%lu B);", c_names[i], (unsigned long)g_count[i], (unsigned long)g_bytes[i]); fprintf(stderr, "\n"); }
#define NEED_INIT() do { if (!g_inited) return CU_NOT_INITIALIZED; } while (0)
static tinynv_stream_t st(void *s) { return s ? (tinynv_stream_t)s : g_default; }

// ---------------------------------------------------------------- init / device / context
EXPORT CUresult cuInit(unsigned flags) {
  pthread_mutex_lock(&g_mu);
  if (g_inited) { pthread_mutex_unlock(&g_mu); return CU_OK; }
  g_stats_enabled = getenv("TFCUDA_STATS") != NULL;
  // libtinynv registers an atexit handler that puts the card down. A caller's dlclose must not unmap that code first,
  // so this image takes a reference on itself that is never released.
  Dl_info di; if (dladdr((void *)&cuInit, &di) && di.dli_fname) (void)dlopen(di.dli_fname, RTLD_NOW);
  tinynv_status_t s = tinynv_init();
  if (s == TINYNV_OK) s = tinynv_device_get(&g_dev, 0);
  if (s == TINYNV_OK) s = tinynv_device_props(g_dev, &g_props);
  if (s != TINYNV_OK) { pthread_mutex_unlock(&g_mu); return S("cuInit", s) == CU_OK ? CU_UNKNOWN : CU_NO_DEVICE; }
  // Without TINYNV_SOCKET libtinynv is a null device on which every call "succeeds" and nothing runs. Never present that.
  if (g_props.cc_major == 0) {
    pthread_mutex_unlock(&g_mu);
    snprintf(t_err, sizeof t_err, "cuInit: no card (libtinynv is on its null device; set TINYNV_SOCKET)");
    fprintf(stderr, "[tfcuda] %s\n", t_err);
    return CU_NO_DEVICE;
  }
  s = tinynv_stream_create(g_dev, &g_default);
  if (s != TINYNV_OK) { pthread_mutex_unlock(&g_mu); S("cuInit default stream", s); return CU_UNKNOWN; }
  // Lend the driver macuda's own copy kernel (libtinycudart/copy1d.cu), as its runtime layer does: device-to-device
  // copies and, under TINYNV_DOWNLOAD_VIA_COMPUTE=1, downloads are then served on the compute queue instead of
  // handing the launch chain over to the copy engine. TFCUDA_NO_COPY_KERNEL=1 leaves it out, for measuring the difference.
  if (!getenv("TFCUDA_NO_COPY_KERNEL")) {
    extern unsigned char copy1d_cubin[]; extern unsigned int copy1d_cubin_len; tinynv_module_t m = NULL; tinynv_kernel_t k = NULL;
    if (tinynv_module_load(g_dev, copy1d_cubin, copy1d_cubin_len, &m) != TINYNV_OK || tinynv_get_kernel(m, "tinycudart_copy1d", &k) != TINYNV_OK ||
        tinynv_set_download_kernel(g_dev, k, 256, 16) != TINYNV_OK)
      fprintf(stderr, "[tfcuda] the copy kernel was not registered (%s): copies stay on the copy engine\n", tinynv_last_error());
    else fprintf(stderr, "[tfcuda] copy kernel registered: device-to-device copies ride the compute queue\n");
  }
  // And macuda's keep-alive kernel (libtinycudart/keepalive.cu): under TINYNV_KEEPALIVE=1 the driver launches it after
  // a synchronisation so the compute engine stays scheduled while the host prepares the next token. Registration
  // alone changes nothing.
  if (!getenv("TFCUDA_NO_KEEPALIVE_KERNEL")) {
    extern unsigned char keepalive_cubin[]; extern unsigned int keepalive_cubin_len; tinynv_module_t m = NULL; tinynv_kernel_t k = NULL;
    if (tinynv_module_load(g_dev, keepalive_cubin, keepalive_cubin_len, &m) != TINYNV_OK || tinynv_get_kernel(m, "tinynv_keepalive", &k) != TINYNV_OK ||
        tinynv_set_keepalive_kernel(g_dev, k) != TINYNV_OK)
      fprintf(stderr, "[tfcuda] the keep-alive kernel was not registered (%s)\n", tinynv_last_error());
  }
  fprintf(stderr, "[tfcuda] %s, sm_%d%d, libtinynv build %s\n", g_props.name, g_props.cc_major, g_props.cc_minor, tinynv_build_id());
  if (g_stats_enabled) atexit(stats_dump);
  g_graph_resident = getenv("TFCUDA_GRAPH_RESIDENT") != NULL;
  g_ev_lazy = 0; // the measurement-only "events complete at record" mode is not part of the public build
  g_inited = 1;
  pthread_mutex_unlock(&g_mu);
  return CU_OK;
}
// The API level implemented is the CUDA 13.0 driver's, as a subset. TensorFold refuses anything below 13000.
EXPORT CUresult cuDriverGetVersion(int *v) { if (!v) return CU_INVALID_VALUE; *v = 13000; return CU_OK; }
EXPORT CUresult cuDeviceGet(CUdevice *dev, int ordinal) { NEED_INIT(); if (!dev) return CU_INVALID_VALUE; if (ordinal != 0) return CU_INVALID_DEVICE; *dev = 0; return CU_OK; }
EXPORT CUresult cuDeviceGetCount(int *n) { NEED_INIT(); if (!n) return CU_INVALID_VALUE; *n = 1; return CU_OK; }
EXPORT CUresult cuDeviceGetUuid_v2(unsigned char *uuid, CUdevice d) {
  NEED_INIT(); if (!uuid || d != 0) return CU_INVALID_VALUE;
  // Not the card's real UUID (libtinynv does not read it): a fixed, labelled value so callers can key on it.
  memcpy(uuid, "tfcuda-macuda-00", 16); return CU_OK;
}
EXPORT CUresult cuDeviceGetName(char *name, int len, CUdevice d) { NEED_INIT(); if (!name || len <= 0 || d != 0) return CU_INVALID_VALUE; snprintf(name, (size_t)len, "%s", g_props.name); return CU_OK; }
// Most dynamic shared memory a block may opt in to. max_shared_per_block is the 48 KiB a kernel gets without asking.
static int optin_smem(void) { return g_props.max_shared_per_sm > 1024 ? g_props.max_shared_per_sm - 1024 : g_props.max_shared_per_block; }
EXPORT CUresult cuDeviceGetAttribute(int *v, int attr, CUdevice d) {
  NEED_INIT(); if (!v || d != 0) return CU_INVALID_VALUE;
  switch (attr) {
    case 1: *v = g_props.max_threads_per_block; return CU_OK;
    case 10: *v = g_props.warp_size; return CU_OK;
    case 16: *v = g_props.sm_count; return CU_OK; // libtinynv reports an upper bound, not the enabled count
    case 18: *v = 0; return CU_OK;                // not integrated: a discrete card behind Thunderbolt
    case 75: *v = g_props.cc_major; return CU_OK;
    case 76: *v = g_props.cc_minor; return CU_OK;
    case 81: *v = g_props.max_shared_per_sm; return CU_OK;
    case 97: *v = optin_smem(); return CU_OK;      // the opt-in ceiling, macuda's rule: per-SM shared memory less 1 KiB
    case 102: *v = 0; return CU_OK;               // no virtual memory management API here
    case 110: *v = 0; return CU_OK;               // no GPUDirect RDMA here
    default: return refuse("cuDeviceGetAttribute", "attribute %d is not read from the card by libtinynv", attr);
  }
}
EXPORT CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice d) { NEED_INIT(); if (!bytes || d != 0) return CU_INVALID_VALUE; *bytes = g_props.total_mem; return CU_OK; }
EXPORT CUresult cuDevicePrimaryCtxRetain(void **ctx, CUdevice d) {
  NEED_INIT(); if (!ctx || d != 0) return CU_INVALID_VALUE;
  pthread_mutex_lock(&g_mu); g_ctx_refs++; pthread_mutex_unlock(&g_mu);
  *ctx = &g_ctx_token; return CU_OK;
}
EXPORT CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice d, unsigned flags) { NEED_INIT(); return d == 0 ? CU_OK : CU_INVALID_DEVICE; }
EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d) {
  NEED_INIT(); if (d != 0) return CU_INVALID_DEVICE;
  pthread_mutex_lock(&g_mu); int ok = g_ctx_refs > 0; if (ok) g_ctx_refs--; pthread_mutex_unlock(&g_mu);
  return ok ? CU_OK : CU_INVALID_CONTEXT;
}
EXPORT CUresult cuCtxSetCurrent(void *ctx) { NEED_INIT(); if (ctx && ctx != &g_ctx_token) return CU_INVALID_CONTEXT; t_ctx = ctx; return CU_OK; }
EXPORT CUresult cuCtxGetCurrent(void **ctx) { NEED_INIT(); if (!ctx) return CU_INVALID_VALUE; *ctx = t_ctx; return CU_OK; }
EXPORT CUresult cuCtxGetDevice(CUdevice *d) { NEED_INIT(); if (!d) return CU_INVALID_VALUE; *d = 0; return CU_OK; }
EXPORT CUresult cuCtxSynchronize(void) { NEED_INIT(); COUNT(C_SYNC, 0); return S("cuCtxSynchronize", tinynv_stream_sync(g_default)); }

// ---------------------------------------------------------------- memory
EXPORT CUresult cuMemGetInfo_v2(size_t *free_b, size_t *total_b) {
  NEED_INIT(); uint64_t f = 0, t = 0; CUresult r = S("cuMemGetInfo", tinynv_mem_info(g_dev, &f, &t));
  if (r == CU_OK) { if (free_b) *free_b = (size_t)f; if (total_b) *total_b = (size_t)t; } return r;
}
EXPORT CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) { NEED_INIT(); if (!p || !n) return CU_INVALID_VALUE; return S("cuMemAlloc", tinynv_malloc(g_dev, n, p)); }
EXPORT CUresult cuMemFree_v2(CUdeviceptr p) { NEED_INIT(); return S("cuMemFree", tinynv_free(g_dev, p)); }
EXPORT CUresult cuMemHostAlloc(void **p, size_t n, unsigned flags) {
  NEED_INIT(); if (!p || !n) return CU_INVALID_VALUE;
  if (flags & 2) return refuse("cuMemHostAlloc", "CU_MEMHOSTALLOC_DEVICEMAP: host pages are not mapped into the card's address space");
  return S("cuMemHostAlloc", tinynv_host_alloc(g_dev, n, p)); // page-aligned host memory; libtinynv does not DMA-pin it
}
EXPORT CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *d, void *h, unsigned flags) { NEED_INIT(); return refuse("cuMemHostGetDevicePointer", "host pages are not mapped into the card's address space"); }
EXPORT CUresult cuMemFreeHost(void *p) { NEED_INIT(); return S("cuMemFreeHost", tinynv_host_free(g_dev, p)); }
EXPORT CUresult cuMemHostRegister_v2(void *p, size_t n, unsigned flags) { NEED_INIT(); return refuse("cuMemHostRegister", "host pages cannot be pinned or mapped through libtinynv"); }
EXPORT CUresult cuMemHostUnregister(void *p) { NEED_INIT(); return refuse("cuMemHostUnregister", "nothing can have been registered"); }

static int capturing(void *s);
static int record_op(void *s, int type, CUdeviceptr dst, CUdeviceptr src, const void *host, size_t n, int fill, void *event, CUresult *out);
enum { R_DTOD = 1, R_HTOD = 2, R_MEMSET = 3, R_EVENT = 4, R_WAIT = 5 };
static CUresult sync_after(const char *what, tinynv_stream_t s, tinynv_status_t r) { if (r != TINYNV_OK) return S(what, r); return S(what, tinynv_stream_sync(s)); }
EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n) { NEED_INIT(); COUNT(C_HTOD, n); if (!n) return CU_OK; return sync_after("cuMemcpyHtoD", g_default, tinynv_memcpy_htod(g_default, dst, src, n)); }
EXPORT CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n) { NEED_INIT(); COUNT(C_DTOH, n); if (!n) return CU_OK; return S("cuMemcpyDtoH", tinynv_memcpy_dtoh(g_default, dst, src, n)); }
EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n) { NEED_INIT(); COUNT(C_DTOD, n); if (!n) return CU_OK; return sync_after("cuMemcpyDtoD", g_default, tinynv_memcpy_dtod(g_default, dst, src, n)); }
// libtinynv stages a host source before returning and a download waits for itself, so these are stricter than async.
EXPORT CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t n, void *s) { NEED_INIT(); COUNT(C_HTOD_ASYNC, n); if (!n) return CU_OK; { CUresult r; if (!record_op(s, R_HTOD, dst, 0, src, n, 0, NULL, &r)) return r; } return S("cuMemcpyHtoDAsync", tinynv_memcpy_htod(st(s), dst, src, n)); }
EXPORT CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t n, void *s) { NEED_INIT(); COUNT(C_DTOH, n); if (!n) return CU_OK; if (capturing(s)) return refuse("cuMemcpyDtoHAsync", "a download inside a stream capture"); return S("cuMemcpyDtoHAsync", tinynv_memcpy_dtoh(st(s), dst, src, n)); }
EXPORT CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src, size_t n, void *s) { NEED_INIT(); COUNT(C_DTOD, n); if (!n) return CU_OK; { CUresult r; if (!record_op(s, R_DTOD, dst, src, NULL, n, 0, NULL, &r)) return r; } return S("cuMemcpyDtoDAsync", tinynv_memcpy_dtod(st(s), dst, src, n)); }

typedef struct { size_t src_x, src_y; int src_type; const void *src_host; CUdeviceptr src; void *src_array; size_t src_pitch;
                 size_t dst_x, dst_y; int dst_type; void *dst_host; CUdeviceptr dst; void *dst_array; size_t dst_pitch;
                 size_t width, height; } Memcpy2D;
_Static_assert(sizeof(Memcpy2D) == 128, "CUDA_MEMCPY2D is 128 bytes");
EXPORT CUresult cuMemcpy2DAsync_v2(const Memcpy2D *c, void *s) {
  NEED_INIT(); if (!c) return CU_INVALID_VALUE;
  if (capturing(s)) return refuse("cuMemcpy2DAsync", "a 2D copy inside a stream capture");
  if ((c->src_type != 1 && c->src_type != 2) || (c->dst_type != 1 && c->dst_type != 2)) return refuse("cuMemcpy2DAsync", "only host and device memory types");
  for (size_t y = 0; y < c->height; y++) {
    size_t so = (c->src_y + y) * c->src_pitch + c->src_x, dof = (c->dst_y + y) * c->dst_pitch + c->dst_x; tinynv_status_t r;
    if (c->src_type == 2 && c->dst_type == 2) r = tinynv_memcpy_dtod(st(s), c->dst + dof, c->src + so, c->width);
    else if (c->src_type == 1 && c->dst_type == 2) r = tinynv_memcpy_htod(st(s), c->dst + dof, (const char *)c->src_host + so, c->width);
    else if (c->src_type == 2 && c->dst_type == 1) r = tinynv_memcpy_dtoh(st(s), (char *)c->dst_host + dof, c->src + so, c->width);
    else { memmove((char *)c->dst_host + dof, (const char *)c->src_host + so, c->width); r = TINYNV_OK; }
    if (r != TINYNV_OK) return S("cuMemcpy2DAsync", r);
  }
  return CU_OK;
}
static int one_byte(unsigned v) { unsigned b = v & 0xff; return v == (b | b << 8 | b << 16 | b << 24); }
EXPORT CUresult cuMemsetD8_v2(CUdeviceptr p, unsigned char v, size_t n) { NEED_INIT(); if (!n) return CU_OK; return sync_after("cuMemsetD8", g_default, tinynv_memset(g_default, p, v, n)); }
EXPORT CUresult cuMemsetD8Async(CUdeviceptr p, unsigned char v, size_t n, void *s) { NEED_INIT(); COUNT(C_MEMSET, n); if (!n) return CU_OK; { CUresult r; if (!record_op(s, R_MEMSET, p, 0, NULL, n, v, NULL, &r)) return r; } return S("cuMemsetD8Async", tinynv_memset(st(s), p, v, n)); }
EXPORT CUresult cuMemsetD32_v2(CUdeviceptr p, unsigned v, size_t n) {
  NEED_INIT(); if (!n) return CU_OK;
  if (!one_byte(v)) return refuse("cuMemsetD32", "pattern %#x: libtinynv fills one byte value only", v);
  return sync_after("cuMemsetD32", g_default, tinynv_memset(g_default, p, (int)(v & 0xff), n * 4));
}
EXPORT CUresult cuMemsetD32Async(CUdeviceptr p, unsigned v, size_t n, void *s) {
  NEED_INIT(); if (!n) return CU_OK;
  if (!one_byte(v)) return refuse("cuMemsetD32Async", "pattern %#x: libtinynv fills one byte value only", v);
  { CUresult r; if (!record_op(s, R_MEMSET, p, 0, NULL, n * 4, (int)(v & 0xff), NULL, &r)) return r; }
  return S("cuMemsetD32Async", tinynv_memset(st(s), p, (int)(v & 0xff), n * 4));
}

// ---------------------------------------------------------------- streams and events
// One in-order queue underneath: ordering is stricter than CUDA's, there is no concurrency between streams, no priority.
EXPORT CUresult cuStreamCreate(void **s, unsigned flags) { NEED_INIT(); if (!s) return CU_INVALID_VALUE; tinynv_stream_t t = NULL; CUresult r = S("cuStreamCreate", tinynv_stream_create(g_dev, &t)); if (r == CU_OK) *s = t; return r; }
EXPORT CUresult cuStreamCreateWithPriority(void **s, unsigned flags, int prio) { return cuStreamCreate(s, flags); }
EXPORT CUresult cuCtxGetStreamPriorityRange(int *least, int *greatest) { NEED_INIT(); if (least) *least = 0; if (greatest) *greatest = 0; return CU_OK; }
EXPORT CUresult cuStreamDestroy_v2(void *s) { NEED_INIT(); if (!s) return CU_INVALID_HANDLE; return S("cuStreamDestroy", tinynv_stream_destroy((tinynv_stream_t)s)); }
EXPORT CUresult cuStreamSynchronize(void *s) { NEED_INIT(); COUNT(C_SYNC, 0); return S("cuStreamSynchronize", tinynv_stream_sync(st(s))); }
EXPORT CUresult cuStreamWaitEvent(void *s, void *e, unsigned flags) { NEED_INIT(); if (!e) return CU_INVALID_HANDLE; { CUresult r; if (!record_op(s, R_WAIT, 0, 0, NULL, 0, 0, e, &r)) return r; } return S("cuStreamWaitEvent", tinynv_stream_wait_event(st(s), (tinynv_event_t)e)); }
EXPORT CUresult cuStreamQuery(void *s) {
  NEED_INIT(); tinynv_event_t e = NULL; int done = 0;
  tinynv_status_t r = tinynv_event_create(g_dev, &e);
  if (r == TINYNV_OK) r = tinynv_event_record(e, st(s));
  if (r == TINYNV_OK) r = tinynv_event_query(e, &done);
  if (e) tinynv_event_destroy(e);
  if (r != TINYNV_OK) return S("cuStreamQuery", r);
  return done ? CU_OK : CU_NOT_READY;
}
// Event bookkeeping for a measurement mode that is disabled in this build (g_ev_lazy is always 0).
#define EV_SLOTS 4096
static void *g_ev_done[EV_SLOTS];
static void ev_mark(void *e, int clear) { size_t h = ((uintptr_t)e >> 4) % EV_SLOTS; g_ev_done[h] = clear ? NULL : e; }
static int ev_lazy_done(void *e) { size_t h = ((uintptr_t)e >> 4) % EV_SLOTS; return g_ev_done[h] == e; }
EXPORT CUresult cuEventCreate(void **e, unsigned flags) { NEED_INIT(); if (!e) return CU_INVALID_VALUE; tinynv_event_t t = NULL; CUresult r = S("cuEventCreate", tinynv_event_create(g_dev, &t)); if (r == CU_OK) *e = t; return r; }
EXPORT CUresult cuEventDestroy_v2(void *e) { NEED_INIT(); if (!e) return CU_INVALID_HANDLE; return S("cuEventDestroy", tinynv_event_destroy((tinynv_event_t)e)); }
EXPORT CUresult cuEventRecord(void *e, void *s) { NEED_INIT(); COUNT(C_EV_RECORD, 0); if (!e) return CU_INVALID_HANDLE; if (g_ev_lazy) { ev_mark(e, 0); return CU_OK; } { CUresult r; if (!record_op(s, R_EVENT, 0, 0, NULL, 0, 0, e, &r)) return r; } return S("cuEventRecord", tinynv_event_record((tinynv_event_t)e, st(s))); }
EXPORT CUresult cuEventRecordWithFlags(void *e, void *s, unsigned flags) {
  return cuEventRecord(e, s); // in a capture this becomes an event record node, as CU_EVENT_RECORD_EXTERNAL asks
}
EXPORT CUresult cuEventSynchronize(void *e) { NEED_INIT(); COUNT(C_EV_SYNC, 0); if (!e) return CU_INVALID_HANDLE; if (g_ev_lazy && ev_lazy_done(e)) return CU_OK; return S("cuEventSynchronize", tinynv_event_sync((tinynv_event_t)e)); }
EXPORT CUresult cuEventQuery(void *e) { NEED_INIT(); COUNT(C_EV_QUERY, 0); if (!e) return CU_INVALID_HANDLE; if (g_ev_lazy && ev_lazy_done(e)) return CU_OK; int done = 0; tinynv_status_t r = tinynv_event_query((tinynv_event_t)e, &done); if (r != TINYNV_OK) return S("cuEventQuery", r); return done ? CU_OK : CU_NOT_READY; }
EXPORT CUresult cuEventElapsedTime(float *ms, void *a, void *b) { NEED_INIT(); return refuse("cuEventElapsedTime", "libtinynv takes no timestamp at an event's record point"); }

// ---------------------------------------------------------------- modules and functions
typedef struct Func { struct Mod *mod; tinynv_kernel_t k; tinynv_kernel_info_t ki; int max_dyn_smem, cluster_ok; struct Func *next; char name[]; } Func;
typedef struct Mod { tinynv_module_t mod; void *owned; Func *funcs; pthread_mutex_t mu; } Mod;

static uint16_t rd16(const unsigned char *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static size_t elf_len(const unsigned char *e) { return (size_t)(rd64(e + 0x28) + (uint64_t)rd16(e + 0x3a) * rd16(e + 0x3c)); }

// Find this card's code in an image: an nvcc fatbin container (the entry whose SM equals the card's), or a bare ELF.
// *owned is set when the payload had to be inflated.
static CUresult pick_image(const unsigned char *img, const unsigned char **elf, size_t *len, void **owned) {
  *owned = NULL;
  if (!memcmp(img, "\x7f" "ELF", 4)) { *elf = img; *len = elf_len(img); return CU_OK; }
  if (rd32(img) != 0xBA55ED50u) { snprintf(t_err, sizeof t_err, "module image is neither a fatbin container nor an ELF (PTX text is not compiled here)"); fprintf(stderr, "[tfcuda] %s\n", t_err); return CU_INVALID_IMAGE; }
  const unsigned want = (unsigned)(g_props.cc_major * 10 + g_props.cc_minor);
  size_t off = rd16(img + 6); const size_t end = off + (size_t)rd64(img + 8);
  char seen[128] = ""; size_t sl = 0;
  while (off + 48 <= end) {
    const unsigned char *e = img + off; unsigned kind = rd16(e), hs = rd32(e + 4), sm = rd32(e + 28); uint64_t ps = rd64(e + 8), fl = rd64(e + 40);
    if (hs < 48 || off + hs + ps > end) break;
    if (kind == 2) { if (sl < sizeof seen - 8) sl += (size_t)snprintf(seen + sl, sizeof seen - sl, " %u", sm); }
    if (kind == 2 && sm == want) {
      const unsigned char *p = e + hs;
      if (fl & 0x2000) {
        unsigned long long cap = ZSTD_getFrameContentSize(p, (size_t)ps);
        if (cap == ZSTD_CONTENTSIZE_ERROR || cap == ZSTD_CONTENTSIZE_UNKNOWN || cap > (1ull << 31)) { fprintf(stderr, "[tfcuda] compressed fatbin entry: unreadable zstd frame size\n"); return CU_INVALID_IMAGE; }
        void *buf = malloc((size_t)cap); if (!buf) return CU_OOM;
        size_t got = ZSTD_decompress(buf, (size_t)cap, p, (size_t)ps);
        if (ZSTD_isError(got) || got != cap || memcmp(buf, "\x7f" "ELF", 4)) { free(buf); fprintf(stderr, "[tfcuda] compressed fatbin entry did not inflate to an ELF\n"); return CU_INVALID_IMAGE; }
        *owned = buf; *elf = buf; *len = got; return CU_OK;
      }
      if (memcmp(p, "\x7f" "ELF", 4)) return CU_INVALID_IMAGE;
      *elf = p; *len = (size_t)ps; return CU_OK;
    }
    off += hs + ps;
  }
  snprintf(t_err, sizeof t_err, "module image has no sm_%u code (it carries:%s)", want, seen);
  fprintf(stderr, "[tfcuda] %s\n", t_err);
  return CU_NO_BINARY_FOR_GPU;
}
EXPORT CUresult cuModuleLoadData(void **out, const void *image) {
  NEED_INIT(); if (!out || !image) return CU_INVALID_VALUE;
  const unsigned char *elf; size_t len; void *owned; CUresult r = pick_image(image, &elf, &len, &owned); if (r != CU_OK) return r;
  Mod *m = calloc(1, sizeof *m); if (!m) { free(owned); return CU_OOM; }
  r = S("cuModuleLoadData", tinynv_module_load(g_dev, elf, len, &m->mod));
  if (r != CU_OK) { free(owned); free(m); return r; }
  m->owned = owned; pthread_mutex_init(&m->mu, NULL); *out = m; return CU_OK;
}
EXPORT CUresult cuModuleLoadDataEx(void **out, const void *image, unsigned n, int *opts, void **vals) { return cuModuleLoadData(out, image); }
EXPORT CUresult cuModuleUnload(void *mv) {
  NEED_INIT(); Mod *m = mv; if (!m) return CU_INVALID_HANDLE;
  CUresult r = S("cuModuleUnload", tinynv_module_unload(m->mod)); if (r != CU_OK) return r;
  for (Func *f = m->funcs; f;) { Func *n = f->next; free(f); f = n; }
  pthread_mutex_destroy(&m->mu); free(m->owned); free(m); return CU_OK;
}
EXPORT CUresult cuModuleGetFunction(void **out, void *mv, const char *name) {
  NEED_INIT(); Mod *m = mv; if (!out || !m || !name) return CU_INVALID_VALUE;
  pthread_mutex_lock(&m->mu);
  for (Func *f = m->funcs; f; f = f->next) if (!strcmp(f->name, name)) { pthread_mutex_unlock(&m->mu); *out = f; return CU_OK; }
  tinynv_kernel_t k = NULL;
  if (tinynv_get_kernel(m->mod, name, &k) != TINYNV_OK || !k) { pthread_mutex_unlock(&m->mu); snprintf(t_err, sizeof t_err, "cuModuleGetFunction: no kernel named %s", name); return CU_NOT_FOUND; }
  Func *f = calloc(1, sizeof *f + strlen(name) + 1); if (!f) { pthread_mutex_unlock(&m->mu); return CU_OOM; }
  strcpy(f->name, name); f->mod = m; f->k = k; f->max_dyn_smem = -1;
  if (tinynv_kernel_info(k, &f->ki) != TINYNV_OK) { pthread_mutex_unlock(&m->mu); free(f); return S("cuModuleGetFunction kernel info", TINYNV_ERR_DRIVER); }
  f->next = m->funcs; m->funcs = f; pthread_mutex_unlock(&m->mu); *out = f; return CU_OK;
}
EXPORT CUresult cuModuleGetGlobal_v2(CUdeviceptr *p, size_t *bytes, void *mv, const char *name) {
  NEED_INIT(); Mod *m = mv; if (!m || !name) return CU_INVALID_VALUE; tinynv_devptr_t a = 0; size_t n = 0;
  tinynv_status_t r = tinynv_get_global(m->mod, name, &a, &n); if (r != TINYNV_OK) return S("cuModuleGetGlobal", r);
  if (p) *p = a; if (bytes) *bytes = n; return CU_OK;
}
EXPORT CUresult cuFuncGetName(const char **name, void *fv) { NEED_INIT(); Func *f = fv; if (!name || !f) return CU_INVALID_VALUE; *name = f->name; return CU_OK; }
EXPORT CUresult cuFuncGetAttribute(int *v, int a, void *fv) {
  NEED_INIT(); Func *f = fv; if (!v || !f) return CU_INVALID_VALUE; const tinynv_kernel_info_t *k = &f->ki;
  switch (a) {
    case 0: if (k->max_threads <= 0) break; *v = k->max_threads; return CU_OK;
    case 1: *v = k->static_smem; return CU_OK;
    case 2: if (!(k->attr_reported & 1u << 2)) break; *v = k->const_size_bytes; return CU_OK;
    case 3: if (!(k->attr_reported & 1u << 3)) break; *v = k->local_size_bytes; return CU_OK;
    case 4: *v = k->regs; return CU_OK;
    case 5: if (!(k->attr_reported & 1u << 5)) break; *v = k->ptx_version; return CU_OK;
    case 6: *v = g_props.cc_major * 10 + g_props.cc_minor; return CU_OK;
    case 8: if (f->max_dyn_smem < 0) break; *v = f->max_dyn_smem; return CU_OK; // answered once the caller has set it
    case 14: *v = f->cluster_ok; return CU_OK;
    default: break;
  }
  return refuse("cuFuncGetAttribute", "attribute %d of %s is not reported for this image", a, f->name);
}
EXPORT CUresult cuFuncSetAttribute(void *fv, int a, int v) {
  NEED_INIT(); Func *f = fv; if (!f) return CU_INVALID_HANDLE;
  if (a == 8) { if (v < 0 || v > optin_smem()) return CU_INVALID_VALUE; f->max_dyn_smem = v; return CU_OK; }
  if (a == 14) { f->cluster_ok = v; return CU_OK; }
  return refuse("cuFuncSetAttribute", "attribute %d", a);
}
EXPORT CUresult cuFuncSetCacheConfig(void *fv, int c) { NEED_INIT(); return fv ? CU_OK : CU_INVALID_HANDLE; }
EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, void *fv, int block, size_t dyn) {
  // macuda's conservative model (libtinycudart cudart_api.c), not NVIDIA's table: registers, threads, shared memory.
  NEED_INIT(); Func *f = fv; if (!n || !f || block <= 0) return CU_INVALID_VALUE;
  int warps = (block + 31) / 32, regs = f->ki.regs > 0 ? f->ki.regs : 1;
  int by_regs = 65536 / (regs * warps * 32), by_threads = 2048 / (warps * 32);
  size_t smem = (size_t)f->ki.static_smem + dyn + 1024, per_sm = g_props.max_shared_per_sm ? (size_t)g_props.max_shared_per_sm : 100 * 1024;
  int r = by_regs < by_threads ? by_regs : by_threads; if ((int)(per_sm / smem) < r) r = (int)(per_sm / smem); if (r > 32) r = 32; if (r < 0) r = 0;
  *n = r; return CU_OK;
}

// ---------------------------------------------------------------- launch
typedef struct KNP { void *func; unsigned gx, gy, gz, bx, by, bz, shared; void **params; void **extra; void *kern; void *ctx; } KNP;
static int record_kernel(void *s, const KNP *p, CUresult *out);
static CUresult marshal(const Func *f, void **params, unsigned char *blob, size_t cap, size_t *len) {
  *len = 0; memset(blob, 0, cap);
  for (int i = 0; i < f->ki.num_params; i++) {
    size_t off = (size_t)f->ki.params[i].offset, sz = (size_t)f->ki.params[i].size;
    if (off + sz > cap || !params || !params[i]) return CU_INVALID_VALUE;
    memcpy(blob + off, params[i], sz); if (off + sz > *len) *len = off + sz;
  }
  return CU_OK;
}
static __thread unsigned t_cluster[3]; // the cluster shape cuLaunchKernelEx asked for, for the one launch it wraps
EXPORT CUresult cuLaunchKernel(void *fv, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, void *s, void **params, void **extra) {
  NEED_INIT(); COUNT(C_LAUNCH, 0); Func *f = fv; if (!f) return CU_INVALID_HANDLE;
  if (extra) return refuse("cuLaunchKernel", "the `extra` argument form");
  { KNP p = { fv, gx, gy, gz, bx, by, bz, shared, params, NULL, NULL, NULL }; CUresult r; if (!record_kernel(s, &p, &r)) return r; }
  unsigned char blob[4096]; size_t len; CUresult r = marshal(f, params, blob, sizeof blob, &len); if (r != CU_OK) return r;
  if (t_cluster[0]) {
    if (getenv("TFCUDA_TRACE")) fprintf(stderr, "[tfcuda] cluster launch %s grid %ux%ux%u block %ux%ux%u cluster %ux%ux%u dyn shared %u static shared %d regs %d\n", f->name, gx, gy, gz, bx, by, bz, t_cluster[0], t_cluster[1], t_cluster[2], shared, f->ki.static_smem, f->ki.regs);
#ifdef TFCUDA_HAVE_CLUSTER
    return S(f->name, tinynv_launch_cluster(st(s), f->k, gx, gy, gz, bx, by, bz, t_cluster[0], t_cluster[1], t_cluster[2], shared, blob, len));
#else
    return refuse("cuLaunchKernelEx", "cluster dimension: this build's libtinynv has no cluster launch");
#endif
  }
  return S(f->name, tinynv_launch(st(s), f->k, gx, gy, gz, bx, by, bz, shared, blob, len));
}
typedef struct { unsigned id; unsigned char pad[4]; union { unsigned char pad[64]; void *p; int i; } value; } LaunchAttr;
typedef struct { unsigned gx, gy, gz, bx, by, bz, shared; void *stream; LaunchAttr *attrs; unsigned n; } LaunchConfig;
_Static_assert(sizeof(LaunchAttr) == 72 && sizeof(LaunchConfig) == 56, "CUlaunchConfig layout");
EXPORT CUresult cuLaunchKernelEx(const LaunchConfig *c, void *fv, void **params, void **extra) {
  NEED_INIT(); if (!c) return CU_INVALID_VALUE; unsigned cl[3] = { 0, 0, 0 };
  for (unsigned i = 0; i < c->n; i++) {
    if (c->attrs[i].id == 2 && c->attrs[i].value.i) return refuse("cuLaunchKernelEx", "cooperative launch");
    if (c->attrs[i].id == 4) { memcpy(cl, &c->attrs[i].value, sizeof cl); if (!cl[0] || !cl[1] || !cl[2]) return CU_INVALID_VALUE; }
  }
  if (cl[0] * cl[1] * cl[2] == 1) cl[0] = 0; // a cluster of one block is an ordinary launch
  t_cluster[0] = cl[0]; t_cluster[1] = cl[1]; t_cluster[2] = cl[2];
  CUresult r = cuLaunchKernel(fv, c->gx, c->gy, c->gz, c->bx, c->by, c->bz, c->shared, c->stream, params, extra);
  t_cluster[0] = t_cluster[1] = t_cluster[2] = 0;
  return r;
}

// ---------------------------------------------------------------- graphs
// A graph here is a recorded list of operations replayed in dependency order on the driver's one in-order queue.
// Nothing is made resident on the card: a replay issues the same launches a direct run would, so results are the
// same as eager execution by construction; what it does not give is the launch-overhead saving of a real CUDA graph.
enum { N_KERNEL, N_DTOD, N_HTOD, N_MEMSET, N_EVENT, N_WAIT };
_Static_assert(sizeof(KNP) == 72 && __builtin_offsetof(KNP, params) == 40, "CUDA_KERNEL_NODE_PARAMS_v2 layout");
typedef struct Node { int type; Func *f; unsigned g[3], b[3], shared; unsigned char *vals; void **ptrs; size_t vals_len;
                      CUdeviceptr dst, src; const void *host; size_t n; int fill; void *event;
                      struct Node **deps; size_t ndeps; struct Node *origin; unsigned cl[3]; } Node;
struct Graph { Node **nodes; size_t n, cap; tinynv_stream_t capturing; };
typedef struct Graph Graph;
typedef struct { Node **nodes; size_t n; tinynv_graph_t res; tinynv_stream_t res_stream; int res_tried; } GExec;
#define MAX_CAPTURES 16
static struct { tinynv_stream_t s; Graph *g; } g_caps[MAX_CAPTURES];

static Graph *capture_of(void *s) {
  tinynv_stream_t t = st(s); Graph *g = NULL;
  pthread_mutex_lock(&g_mu);
  for (int i = 0; i < MAX_CAPTURES; i++) if (g_caps[i].g && g_caps[i].s == t) g = g_caps[i].g;
  pthread_mutex_unlock(&g_mu);
  return g;
}
static int capturing(void *s) { return g_inited && capture_of(s) != NULL; }
static void node_free(Node *n) { if (!n) return; free(n->vals); free(n->ptrs); free(n->deps); free(n); }
static CUresult graph_push(Graph *g, Node *n) {
  if (g->n == g->cap) { size_t c = g->cap ? g->cap * 2 : 32; Node **p = realloc(g->nodes, c * sizeof *p); if (!p) return CU_OOM; g->nodes = p; g->cap = c; }
  g->nodes[g->n++] = n; return CU_OK;
}
// Copy a kernel node's function, geometry and argument VALUES (CUDA's rule: the values are taken at this call).
static CUresult node_set_kernel(Node *n, const KNP *p) {
  Func *f = p->func; if (!f) return CU_INVALID_HANDLE;
  if (p->extra) return refuse("graph kernel node", "the `extra` argument form");
  size_t total = 0; for (int i = 0; i < f->ki.num_params; i++) total += (size_t)f->ki.params[i].size;
  if (f->ki.num_params && !p->params) return CU_INVALID_VALUE;
  unsigned char *vals = malloc(total ? total : 1); void **ptrs = calloc((size_t)f->ki.num_params + 1, sizeof *ptrs);
  if (!vals || !ptrs) { free(vals); free(ptrs); return CU_OOM; }
  size_t off = 0;
  for (int i = 0; i < f->ki.num_params; i++) {
    size_t sz = (size_t)f->ki.params[i].size; if (!p->params[i]) { free(vals); free(ptrs); return CU_INVALID_VALUE; }
    memcpy(vals + off, p->params[i], sz); ptrs[i] = vals + off; off += sz;
  }
  free(n->vals); free(n->ptrs);
  n->type = N_KERNEL; n->f = f; n->vals = vals; n->ptrs = ptrs; n->vals_len = total;
  n->g[0] = p->gx; n->g[1] = p->gy; n->g[2] = p->gz; n->b[0] = p->bx; n->b[1] = p->by; n->b[2] = p->bz; n->shared = p->shared;
  return CU_OK;
}
static Node *node_clone(const Node *o) {
  Node *n = calloc(1, sizeof *n); if (!n) return NULL; *n = *o; n->vals = NULL; n->ptrs = NULL; n->deps = NULL; n->ndeps = 0; n->origin = (Node *)o;
  if (o->type == N_KERNEL) {
    n->vals = malloc(o->vals_len ? o->vals_len : 1); n->ptrs = calloc((size_t)o->f->ki.num_params + 1, sizeof *n->ptrs);
    if (!n->vals || !n->ptrs) { node_free(n); return NULL; }
    memcpy(n->vals, o->vals, o->vals_len);
    for (int i = 0; i < o->f->ki.num_params; i++) n->ptrs[i] = n->vals + ((unsigned char *)o->ptrs[i] - o->vals);
  }
  return n;
}
static CUresult node_run(const Node *n, tinynv_stream_t s) {
  switch (n->type) {
    case N_KERNEL: { unsigned char blob[4096]; size_t len; CUresult r = marshal(n->f, n->ptrs, blob, sizeof blob, &len); if (r != CU_OK) return r;
#ifdef TFCUDA_HAVE_CLUSTER
                     if (n->cl[0]) return S(n->f->name, tinynv_launch_cluster(s, n->f->k, n->g[0], n->g[1], n->g[2], n->b[0], n->b[1], n->b[2], n->cl[0], n->cl[1], n->cl[2], n->shared, blob, len));
#else
                     if (n->cl[0]) return refuse("graph kernel node", "cluster dimension: this build's libtinynv has no cluster launch");
#endif
                     return S(n->f->name, tinynv_launch(s, n->f->k, n->g[0], n->g[1], n->g[2], n->b[0], n->b[1], n->b[2], n->shared, blob, len)); }
    case N_DTOD: return S("graph copy", tinynv_memcpy_dtod(s, n->dst, n->src, n->n));
    case N_HTOD: return S("graph upload", tinynv_memcpy_htod(s, n->dst, n->host, n->n));
    case N_MEMSET: return S("graph fill", tinynv_memset(s, n->dst, n->fill, n->n));
    case N_EVENT: return S("graph event record", tinynv_event_record((tinynv_event_t)n->event, s));
    case N_WAIT: return S("graph event wait", tinynv_stream_wait_event(s, (tinynv_event_t)n->event));
  }
  return CU_UNKNOWN;
}
// Record one non-kernel operation into the capture on `s`. Returns -1 when `s` is not capturing.
static int record_op(void *s, int type, CUdeviceptr dst, CUdeviceptr src, const void *host, size_t n, int fill, void *event, CUresult *out) {
  Graph *g = capture_of(s); if (!g) return -1;
  Node *nd = calloc(1, sizeof *nd); if (!nd) { *out = CU_OOM; return 0; }
  nd->type = type; nd->dst = dst; nd->src = src; nd->host = host; nd->n = n; nd->fill = fill; nd->event = event;
  *out = graph_push(g, nd); if (*out != CU_OK) node_free(nd);
  return 0;
}
static int record_kernel(void *s, const KNP *p, CUresult *out) {
  Graph *g = capture_of(s); if (!g) return -1;
  Node *nd = calloc(1, sizeof *nd); if (!nd) { *out = CU_OOM; return 0; }
  *out = node_set_kernel(nd, p); nd->cl[0] = t_cluster[0]; nd->cl[1] = t_cluster[1]; nd->cl[2] = t_cluster[2]; // a captured cluster launch keeps its shape
  if (*out == CU_OK) *out = graph_push(g, nd);
  if (*out != CU_OK) node_free(nd);
  return 0;
}

EXPORT CUresult cuStreamBeginCapture_v2(void *s, int mode) {
  NEED_INIT(); tinynv_stream_t t = st(s); CUresult r = CU_OK;
  Graph *g = calloc(1, sizeof *g); if (!g) return CU_OOM; g->capturing = t;
  pthread_mutex_lock(&g_mu);
  int slot = -1; for (int i = 0; i < MAX_CAPTURES; i++) { if (g_caps[i].g && g_caps[i].s == t) r = CU_INVALID_VALUE; if (!g_caps[i].g && slot < 0) slot = i; }
  if (r == CU_OK && slot < 0) r = CU_OOM;
  if (r == CU_OK) { g_caps[slot].s = t; g_caps[slot].g = g; }
  pthread_mutex_unlock(&g_mu);
  if (r != CU_OK) free(g);
  return r;
}
EXPORT CUresult cuStreamEndCapture(void *s, void **out) {
  NEED_INIT(); if (!out) return CU_INVALID_VALUE; tinynv_stream_t t = st(s); Graph *g = NULL;
  pthread_mutex_lock(&g_mu);
  for (int i = 0; i < MAX_CAPTURES; i++) if (g_caps[i].g && g_caps[i].s == t) { g = g_caps[i].g; g_caps[i].g = NULL; g_caps[i].s = NULL; }
  pthread_mutex_unlock(&g_mu);
  if (!g) return CU_INVALID_VALUE;
  g->capturing = NULL;
  // capture order is the dependency order: each node follows the one recorded before it
  for (size_t i = 1; i < g->n; i++) { g->nodes[i]->deps = malloc(sizeof(Node *)); if (!g->nodes[i]->deps) return CU_OOM; g->nodes[i]->deps[0] = g->nodes[i - 1]; g->nodes[i]->ndeps = 1; }
  *out = g; return CU_OK;
}
EXPORT CUresult cuStreamIsCapturing(void *s, int *status) { NEED_INIT(); if (!status) return CU_INVALID_VALUE; *status = capture_of(s) ? 1 : 0; return CU_OK; }
EXPORT CUresult cuGraphCreate(void **out, unsigned flags) { NEED_INIT(); if (!out) return CU_INVALID_VALUE; Graph *g = calloc(1, sizeof *g); if (!g) return CU_OOM; *out = g; return CU_OK; }
EXPORT CUresult cuGraphDestroy(void *gv) {
  NEED_INIT(); Graph *g = gv; if (!g) return CU_INVALID_HANDLE;
  for (size_t i = 0; i < g->n; i++) node_free(g->nodes[i]);
  free(g->nodes); free(g); return CU_OK;
}
static int in_graph(const Graph *g, const Node *n) { for (size_t i = 0; i < g->n; i++) if (g->nodes[i] == n) return 1; return 0; }
static CUresult add_dep(Node *to, Node *from) {
  for (size_t i = 0; i < to->ndeps; i++) if (to->deps[i] == from) return CU_OK;
  Node **d = realloc(to->deps, (to->ndeps + 1) * sizeof *d); if (!d) return CU_OOM;
  to->deps = d; to->deps[to->ndeps++] = from; return CU_OK;
}
EXPORT CUresult cuGraphAddKernelNode_v2(void **out, void *gv, void *const *deps, size_t nd, const KNP *p) {
  NEED_INIT(); Graph *g = gv; if (!out || !g || !p) return CU_INVALID_VALUE;
  for (size_t i = 0; i < nd; i++) if (!in_graph(g, deps[i])) return CU_INVALID_VALUE;
  Node *n = calloc(1, sizeof *n); if (!n) return CU_OOM;
  CUresult r = node_set_kernel(n, p);
  for (size_t i = 0; r == CU_OK && i < nd; i++) r = add_dep(n, deps[i]);
  if (r == CU_OK) r = graph_push(g, n);
  if (r != CU_OK) { node_free(n); return r; }
  *out = n; return CU_OK;
}
EXPORT CUresult cuGraphKernelNodeGetParams_v2(void *nv, KNP *p) {
  NEED_INIT(); Node *n = nv; if (!n || !p || n->type != N_KERNEL) return CU_INVALID_VALUE;
  memset(p, 0, sizeof *p); p->func = n->f; p->gx = n->g[0]; p->gy = n->g[1]; p->gz = n->g[2]; p->bx = n->b[0]; p->by = n->b[1]; p->bz = n->b[2];
  p->shared = n->shared; p->params = n->ptrs; return CU_OK;
}
EXPORT CUresult cuGraphKernelNodeSetParams_v2(void *nv, const KNP *p) { NEED_INIT(); Node *n = nv; if (!n || !p || n->type != N_KERNEL) return CU_INVALID_VALUE; return node_set_kernel(n, p); }
EXPORT CUresult cuGraphAddDependencies(void *gv, void *const *from, void *const *to, size_t n) {
  NEED_INIT(); Graph *g = gv; if (!g || (n && (!from || !to))) return CU_INVALID_VALUE;
  for (size_t i = 0; i < n; i++) { if (!in_graph(g, from[i]) || !in_graph(g, to[i]) || from[i] == to[i]) return CU_INVALID_VALUE; CUresult r = add_dep(to[i], from[i]); if (r != CU_OK) return r; }
  return CU_OK;
}
EXPORT CUresult cuGraphGetNodes(void *gv, void **nodes, size_t *n) {
  NEED_INIT(); Graph *g = gv; if (!g || !n) return CU_INVALID_VALUE;
  if (!nodes) { *n = g->n; return CU_OK; }
  size_t k = *n < g->n ? *n : g->n; for (size_t i = 0; i < k; i++) nodes[i] = g->nodes[i];
  for (size_t i = k; i < *n; i++) nodes[i] = NULL;
  *n = k; return CU_OK;
}
EXPORT CUresult cuGraphInstantiateWithFlags(void **out, void *gv, unsigned long long flags) {
  NEED_INIT(); Graph *g = gv; if (!out || !g) return CU_INVALID_VALUE;
  GExec *e = calloc(1, sizeof *e); if (!e) return CU_OOM;
  e->nodes = calloc(g->n ? g->n : 1, sizeof *e->nodes); if (!e->nodes) { free(e); return CU_OOM; }
  // Dependency order, stable in insertion order (Kahn). A cycle is refused.
  unsigned char *done = calloc(g->n ? g->n : 1, 1); if (!done) { free(e->nodes); free(e); return CU_OOM; }
  CUresult r = CU_OK;
  while (e->n < g->n && r == CU_OK) {
    size_t before = e->n;
    for (size_t i = 0; i < g->n && r == CU_OK; i++) {
      if (done[i]) continue; int ready = 1;
      for (size_t d = 0; d < g->nodes[i]->ndeps && ready; d++) { size_t j = 0; while (j < g->n && g->nodes[j] != g->nodes[i]->deps[d]) j++; if (j == g->n || !done[j]) ready = 0; }
      if (!ready) continue;
      Node *c = node_clone(g->nodes[i]); if (!c) { r = CU_OOM; break; }
      e->nodes[e->n++] = c; done[i] = 1;
    }
    if (r == CU_OK && e->n == before) { snprintf(t_err, sizeof t_err, "cuGraphInstantiate: the graph's dependencies form a cycle"); r = CU_INVALID_VALUE; }
  }
  free(done);
  // each clone remembers its dependencies as clones, so a later update can compare edges, not only nodes
  for (size_t i = 0; r == CU_OK && i < e->n; i++) {
    const Node *o = e->nodes[i]->origin; if (!o->ndeps) continue;
    e->nodes[i]->deps = calloc(o->ndeps, sizeof(Node *)); if (!e->nodes[i]->deps) { r = CU_OOM; break; }
    for (size_t d = 0; d < o->ndeps; d++) for (size_t j = 0; j < e->n; j++) if (e->nodes[j]->origin == o->deps[d]) e->nodes[i]->deps[d] = e->nodes[j];
    e->nodes[i]->ndeps = o->ndeps;
  }
  if (r != CU_OK) { for (size_t i = 0; i < e->n; i++) node_free(e->nodes[i]); free(e->nodes); free(e); return r; }
  *out = e; return CU_OK;
}
EXPORT CUresult cuGraphUpload(void *ev, void *s) { NEED_INIT(); return ev ? CU_OK : CU_INVALID_HANDLE; } // nothing is resident, so nothing to move
EXPORT CUresult cuGraphLaunch(void *ev, void *s) {
  NEED_INIT(); COUNT(C_GRAPH_LAUNCH, 0); GExec *e = ev; if (!e) return CU_INVALID_HANDLE;
  if (capturing(s)) return refuse("cuGraphLaunch", "a graph launch inside a stream capture (child graphs)");
  // Resident replay (TFCUDA_GRAPH_RESIDENT=1 together with libtinynv's TINYNV_GRAPH_RESIDENT=1): the first launch
  // records the executable's kernel launches into a chain that lives on the card and runs it once; later launches
  // on the same stream run that chain as one batch. Only for executables made purely of kernel launches without
  // clusters; any change to a node drops the recording (exec_drop_resident). Otherwise: replay node by node.
  if (g_graph_resident) {
    if (e->res && e->res_stream == st(s)) return S("cuGraphLaunch (resident)", tinynv_graph_launch(st(s), e->res));
    if (!e->res && !e->res_tried) {
      e->res_tried = 1; int pure = e->n > 0;
      for (size_t i = 0; i < e->n; i++) if (e->nodes[i]->type != N_KERNEL || e->nodes[i]->cl[0]) pure = 0;
      if (pure && tinynv_graph_begin(st(s)) == TINYNV_OK) {
        CUresult r = CU_OK; for (size_t i = 0; i < e->n && r == CU_OK; i++) r = node_run(e->nodes[i], st(s));
        tinynv_graph_t g = NULL; tinynv_status_t t = tinynv_graph_end(st(s), &g);   // seals the recording and runs it once
        if (r != CU_OK) { if (g) tinynv_graph_free(st(s), g); return r; }
        if (t != TINYNV_OK) return S("cuGraphLaunch (recording)", t);
        e->res = g; e->res_stream = st(s); return CU_OK;
      }
    }
  }
  for (size_t i = 0; i < e->n; i++) { CUresult r = node_run(e->nodes[i], st(s)); if (r != CU_OK) return r; }
  return CU_OK;
}
static void exec_drop_resident(GExec *e) { if (e->res) tinynv_graph_free(e->res_stream, e->res); e->res = NULL; e->res_stream = NULL; e->res_tried = 0; }
EXPORT CUresult cuGraphExecDestroy(void *ev) { NEED_INIT(); GExec *e = ev; if (!e) return CU_INVALID_HANDLE; exec_drop_resident(e); for (size_t i = 0; i < e->n; i++) node_free(e->nodes[i]); free(e->nodes); free(e); return CU_OK; }
static size_t exec_pos(const GExec *e, const Node *n) { for (size_t i = 0; i < e->n; i++) if (e->nodes[i] == n) return i; return (size_t)-1; }
typedef struct { unsigned result; void *error_node; void *error_from_node; } UpdateInfo;
_Static_assert(sizeof(UpdateInfo) == 24, "CUgraphExecUpdateResultInfo layout");
// Takes every node's parameters from `g` when it has the executable's topology: same count, and per position (in the
// executable's order, matched through insertion order of g) the same node type and, for kernels, the same function.
EXPORT CUresult cuGraphExecUpdate_v2(void *ev, void *gv, UpdateInfo *info) {
  NEED_INIT(); GExec *e = ev; Graph *g = gv; if (!e || !g) return CU_INVALID_HANDLE;
  if (info) { info->result = 0; info->error_node = NULL; info->error_from_node = NULL; }
  if (e->n != g->n) { if (info) info->result = 2; return 910; }
  GExec *tmp = NULL; CUresult r = cuGraphInstantiateWithFlags((void **)&tmp, g, 0); if (r != CU_OK) return r;
  for (size_t i = 0; i < e->n; i++) {
    Node *a = e->nodes[i], *b = tmp->nodes[i];
    if (a->type != b->type) { if (info) { info->result = 3; info->error_node = a->origin; info->error_from_node = b->origin; } cuGraphExecDestroy(tmp); return 910; }
    if (a->type == N_KERNEL && (a->f != b->f || a->cl[0] != b->cl[0] || a->cl[1] != b->cl[1] || a->cl[2] != b->cl[2])) { if (info) { info->result = 4; info->error_node = a->origin; info->error_from_node = b->origin; } cuGraphExecDestroy(tmp); return 910; }
  }
  // same edges: node i must depend on the same positions in both executables
  for (size_t i = 0; i < e->n; i++) {
    Node *x = e->nodes[i], *y = tmp->nodes[i]; int same = x->ndeps == y->ndeps;
    for (size_t d = 0; same && d < x->ndeps; d++) {
      size_t px = exec_pos(e, x->deps[d]); int found = 0;
      for (size_t k = 0; k < y->ndeps; k++) if (exec_pos(tmp, y->deps[k]) == px) found = 1;
      same = found;
    }
    if (!same) { if (info) { info->result = 2; info->error_node = x->origin; info->error_from_node = y->origin; } cuGraphExecDestroy(tmp); return 910; }
  }
  exec_drop_resident(e);
  for (size_t i = 0; i < e->n; i++) node_free(e->nodes[i]);
  free(e->nodes); e->nodes = tmp->nodes; free(tmp);
  return CU_OK;
}
EXPORT CUresult cuGraphExecKernelNodeSetParams_v2(void *ev, void *nv, const KNP *p) {
  NEED_INIT(); GExec *e = ev; if (!e || !nv || !p) return CU_INVALID_VALUE;
  for (size_t i = 0; i < e->n; i++) if (e->nodes[i]->origin == nv) {
    if (e->nodes[i]->type != N_KERNEL) return CU_INVALID_VALUE;
    if (p->func != (void *)e->nodes[i]->f) return refuse("cuGraphExecKernelNodeSetParams", "changing a node's function in an executable graph");
    exec_drop_resident(e);
    return node_set_kernel(e->nodes[i], p);
  }
  return CU_INVALID_VALUE; // not a node of the graph this executable came from
}

// ---------------------------------------------------------------- errors
EXPORT CUresult cuGetErrorName(CUresult e, const char **s) {
  if (!s) return CU_INVALID_VALUE;
  switch (e) {
    case 0: *s = "CUDA_SUCCESS"; break; case 1: *s = "CUDA_ERROR_INVALID_VALUE"; break; case 2: *s = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case 3: *s = "CUDA_ERROR_NOT_INITIALIZED"; break; case 100: *s = "CUDA_ERROR_NO_DEVICE"; break; case 101: *s = "CUDA_ERROR_INVALID_DEVICE"; break;
    case 200: *s = "CUDA_ERROR_INVALID_IMAGE"; break; case 201: *s = "CUDA_ERROR_INVALID_CONTEXT"; break; case 209: *s = "CUDA_ERROR_NO_BINARY_FOR_GPU"; break;
    case 400: *s = "CUDA_ERROR_INVALID_HANDLE"; break; case 500: *s = "CUDA_ERROR_NOT_FOUND"; break; case 600: *s = "CUDA_ERROR_NOT_READY"; break;
    case 719: *s = "CUDA_ERROR_LAUNCH_FAILED"; break; case 801: *s = "CUDA_ERROR_NOT_SUPPORTED"; break; case 999: *s = "CUDA_ERROR_UNKNOWN"; break;
    default: *s = NULL; return CU_INVALID_VALUE;
  }
  return CU_OK;
}
EXPORT CUresult cuGetErrorString(CUresult e, const char **s) {
  if (!s) return CU_INVALID_VALUE;
  if (e != CU_OK && t_err[0]) { *s = t_err; return CU_OK; } // the reason the last failing call on this thread gave
  return cuGetErrorName(e, s);
}
