// MoE routing and grouping in one launch, replacing the router GEMV (cuBLAS), vLLM's topk_softmax and
// group_shared: every warp of the grid computes router logits for some of the E experts plus the shared expert's gate
// (a dot product over the hidden width, fp32), and the last block to arrive does, per row,
//   softmax over the E logits (each rounded to bf16 first, as the bf16 router GEMV's output),
//   top-k by repeated argmax (lowest expert id on ties, as vLLM's topk_softmax), renormalised to sum 1,
//   the shared slot's weight sigmoid(gate . x) in fp32,
// and then group_shared's grouping (distinct experts in id order, members in row order). Its outputs are the same
// buffers group_shared fills, so the grouped kernels run unchanged.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int RT = 256;              // threads a block
constexpr int RW = RT / 32;
constexpr int MAX_K = 16;

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// 8 consecutive values of a 16-bit float row as fp32
template <typename T>
__device__ __forceinline__ void load8(const T* p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const T* h = reinterpret_cast<const T*>(&u);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = to_f<T>(h[j]);
}

template <typename T>
__global__ void __launch_bounds__(RT) route_group_kernel(
    const T* __restrict__ x, int x_stride, const T* __restrict__ gate, const T* __restrict__ sgate, int H, int E,
    int k, int shared_id, float* __restrict__ logits, int* __restrict__ counter, int* __restrict__ pick_out,
    float* __restrict__ wts_out, int* __restrict__ uids, int* __restrict__ ucount, int* __restrict__ members, int R,
    int maxm) {
    __shared__ int last;
    __shared__ int sh_pick[32 * 33];
    __shared__ int warp_tot[RW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int has = shared_id >= 0;
    const int C = E + has;                                        // logit columns: experts, then the shared gate
    const int slots = k + has;

    // logits[r][c] = x[r] . W[c] in fp32: one warp a column, lanes over 8-wide chunks, a fixed shuffle tree
    for (int c = blockIdx.x * RW + warp; c < C; c += gridDim.x * RW) {
        const T* w = c < E ? gate + (size_t)c * H : sgate;
        for (int r = 0; r < R; ++r) {
            float acc = 0.f;
            for (int j = lane * 8; j < H; j += 256) {
                float a[8], b[8];
                load8<T>(x + (size_t)r * x_stride + j, a);
                load8<T>(w + j, b);
#pragma unroll
                for (int q = 0; q < 8; ++q) acc = fmaf(a[q], b[q], acc);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) logits[(size_t)r * C + c] = acc;
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();

    // per row (one warp a row): softmax over bf16-rounded logits, top-k by argmax, renormalise
    for (int r = warp; r < R; r += RW) {
        const float* lr = logits + (size_t)r * C;
        float p[8];                                               // E <= 256: lane holds experts lane + 32 q
        float mx = -INFINITY;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int e = lane + 32 * q;
            p[q] = e < E ? __bfloat162float(__float2bfloat16_rn(__ldcg(lr + e))) : -INFINITY;
            mx = fmaxf(mx, p[q]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float sum = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            p[q] = lane + 32 * q < E ? expf(p[q] - mx) : 0.f;
            sum += p[q];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        int ids[MAX_K];
        float ws[MAX_K];
        float wsum = 0.f;
        for (int t = 0; t < k; ++t) {
            float bv = -1.f;
            int be = 1 << 30;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int e = lane + 32 * q;
                if (e < E && (p[q] > bv || (p[q] == bv && e < be))) { bv = p[q]; be = e; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
            }
            ids[t] = be;
            ws[t] = bv / sum;
            wsum += ws[t];
#pragma unroll
            for (int q = 0; q < 8; ++q)
                if (lane + 32 * q == be) p[q] = -1.f;
        }
        if (lane == 0) {
            for (int t = 0; t < k; ++t) {
                pick_out[r * slots + t] = ids[t];
                wts_out[r * slots + t] = ws[t] / wsum;
            }
            if (has) {
                pick_out[r * slots + k] = shared_id;
                wts_out[r * slots + k] = 1.f / (1.f + expf(-__ldcg(lr + E)));
            }
        }
    }
    __threadfence_block();
    __syncthreads();

    // group_shared's grouping over pick_out
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += RT) sh_pick[i] = pick_out[i];
    __syncthreads();
    constexpr int PER = 2;                                        // experts a thread (E + 1 <= 512)
    int cnt[PER];
    int used = 0;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
        const int e = threadIdx.x * PER + q;
        int c = 0;
        if (e < E + has)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int v = lane < RW ? warp_tot[lane] : 0;
        int s2 = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x2 = __shfl_up_sync(0xffffffffu, s2, o);
            if (lane >= o) s2 += x2;
        }
        if (lane < RW) warp_tot[lane] = s2 - v;
        if (lane == RW - 1) ucount[0] = s2;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * PER + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
    if (threadIdx.x == 0) *counter = 0;
}

}  // namespace

void esp_route_group_cuda(const at::Tensor& x, const at::Tensor& gate, const c10::optional<at::Tensor>& sgate,
                          int64_t k, int64_t shared_id, at::Tensor& logits, at::Tensor& counter, at::Tensor& pick_out,
                          at::Tensor& wts_out, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members) {
    const int R = (int)x.size(0), H = (int)x.size(1), E = (int)gate.size(0);
    const int has = shared_id >= 0;
    TORCH_CHECK(E <= 256 && E + has <= 2 * RT && k <= MAX_K && H % 256 == 0, "route_group: shape limits");
    TORCH_CHECK(R * (k + has) <= 32 * 33 && k + has <= 32, "route_group: window too large");
    TORCH_CHECK(!has || sgate.has_value(), "the shared gate weight is needed with a shared slot");
    TORCH_CHECK(logits.numel() >= (int64_t)R * (E + has), "logits scratch too small");
    const int C = E + has;
    const int blocks = (C + RW - 1) / RW;
    auto stream = at::cuda::getCurrentCUDAStream();
#define ESP_RG(T)                                                                                              \
    route_group_kernel<T><<<blocks, RT, 0, stream>>>(                                                          \
        reinterpret_cast<const T*>(x.data_ptr()), (int)x.stride(0), reinterpret_cast<const T*>(gate.data_ptr()), \
        has ? reinterpret_cast<const T*>(sgate->data_ptr()) : nullptr, H, E, (int)k, (int)shared_id,          \
        logits.data_ptr<float>(), counter.data_ptr<int>(), pick_out.data_ptr<int>(), wts_out.data_ptr<float>(), \
        uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), R, (int)members.size(1))
    if (x.scalar_type() == at::kBFloat16) ESP_RG(__nv_bfloat16);
    else ESP_RG(half);
#undef ESP_RG
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
