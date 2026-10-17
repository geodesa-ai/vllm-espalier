// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.
//
// Vendored from TensorFold (Apache-2.0, python-0.6 at ed78d6f, src/tensorfold/cuda/exl3/experts.cu). Espalier changes: a
// codebook table per projection beside the width table (experts_grouped.cuh), so the grouped launch is no longer
// templated on one codebook.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "experts_grouped.cuh"

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(to_f<TIN>(xr[j]), __half2float(suh[j]));   // never contracted
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = __fmul_rn(act, __half2float(suh_d[(size_t)e * N + n + j]));   // never contracted into the butterfly
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}


// Espalier: the shared expert as one more slot. Builds the extended pick table (slot `slots_in` = expert `shared_id`) and
// weight table (routed weights, then sigmoid(x . gate_w) for the shared slot, fp32), then groups it exactly as
// group_kernel does. Without a shared expert (shared_id < 0) it copies pick/wts and groups. One block.
template <typename TIN>
__global__ void __launch_bounds__(GROUP_THREADS) group_shared_kernel(
    const int* __restrict__ pick_in, const float* __restrict__ wts_in, const TIN* __restrict__ x, int x_stride,
    const TIN* __restrict__ gate_w, int H, int shared_id, int* __restrict__ pick_out, float* __restrict__ wts_out,
    int* __restrict__ uids, int* __restrict__ ucount, int* __restrict__ members, int R, int slots_in, int E,
    int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    __shared__ float red[GROUP_THREADS / 32];
    const int has = shared_id >= 0;
    const int slots = slots_in + has;
    const int n = R * slots;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) {
        const int r = i / slots, k = i % slots;
        const int e = k < slots_in ? pick_in[r * slots_in + k] : shared_id;
        sh_pick[i] = e;
        pick_out[i] = e;
        if (k < slots_in) wts_out[i] = wts_in[r * slots_in + k];
    }
    if (has) {
        // shared gate: dot(x[r], gate_w) in fp32, fixed per-thread strides then a fixed tree (row-invariant)
        for (int r = 0; r < R; ++r) {
            float acc = 0.f;
            for (int j = threadIdx.x; j < H; j += GROUP_THREADS)
                acc = fmaf(to_f<TIN>(x[(size_t)r * x_stride + j]), to_f<TIN>(gate_w[j]), acc);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) red[warp] = acc;
            __syncthreads();
            if (warp == 0) {
                float v = red[lane];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) wts_out[r * slots + slots_in] = 1.f / (1.f + expf(-v));
            }
            __syncthreads();
        }
    }
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s2 = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x2 = __shfl_up_sync(0xffffffffu, s2, o);
            if (lane >= o) s2 += x2;
        }
        warp_tot[lane] = s2 - v;
        if (lane == 31) ucount[0] = s2;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

// Espalier: down_combine with two outputs in the model's dtype: out = sum over the first `n_routed` slots of w * y (fp32
// fma chain from 0, slot order, as down_combine_kernel), and, when out_shared is given, out_shared = w * y of slot
// n_routed (the shared expert, its sigmoid gate as w). No fp32 output and no cast kernel after it.
template <typename TOUT>
__global__ void down_combine2_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y,
                                     const float* __restrict__ wts, TOUT* __restrict__ out,
                                     TOUT* __restrict__ out_shared, int P, int D, int SK, int E, int slots,
                                     int n_routed) {
    __shared__ float4 part[32][32];
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k == 0) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int q = 0; q < n_routed; ++q) {
            const float w = wts[r * slots + q];
            const float4 u = part[q][lane];
            acc[0] = fmaf(w, u.x, acc[0]);
            acc[1] = fmaf(w, u.y, acc[1]);
            acc[2] = fmaf(w, u.z, acc[2]);
            acc[3] = fmaf(w, u.w, acc[3]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = from_f<TOUT>(acc[j]);
    } else if (k == n_routed && out_shared != nullptr) {
        const float w = wts[r * slots + k];
        const float4 u = part[k][lane];
        out_shared[(size_t)r * D + n + 0] = from_f<TOUT>(w * u.x);
        out_shared[(size_t)r * D + n + 1] = from_f<TOUT>(w * u.y);
        out_shared[(size_t)r * D + n + 2] = from_f<TOUT>(w * u.z);
        out_shared[(size_t)r * D + n + 3] = from_f<TOUT>(w * u.w);
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------


void exl3x_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& C0, const at::Tensor& C1,
                        const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members, at::Tensor& Z,
                        int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK, int64_t slots, int64_t nt,
                        int64_t warps, int64_t pf, int64_t lo, int64_t hi) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.cb_0 = C0.data_ptr<int>();
    a.cb_1 = C1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    auto stream = at::cuda::getCurrentCUDAStream();
    tf_exl3x::grouped_launch(a, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_dequant_cuda(const at::Tensor& T, at::Tensor& out, int64_t K, int64_t N, int64_t k2, int64_t cb) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    auto o = reinterpret_cast<half*>(out.data_ptr());
    tf_exl3x::dequant_launch(t, o, (int)K, (int)N, (int)k2, (int)cb, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members, int64_t R,
                      int64_t slots, int64_t E) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    const size_t smem = (size_t)R * slots * sizeof(int);
    constexpr size_t static_smem = GROUP_THREADS / 32 * sizeof(int);
    if (smem + static_smem > 48 * 1024) {
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, group_kernel));
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const size_t limit = device->sharedMemPerBlockOptin - attributes.sharedSizeBytes;
        TORCH_CHECK(smem <= limit, "EXL3 grouping needs ", smem, " dynamic shared-memory bytes; this GPU allows ",
                    limit, " after the kernel's static storage");
        if (smem > (size_t)attributes.maxDynamicSharedSizeBytes)
            C10_CUDA_CHECK(cudaFuncSetAttribute(group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)limit));
    }
    group_kernel<<<1, GROUP_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        pick.data_ptr<int>(), uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), (int)R,
        (int)slots, (int)E, (int)members.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_rot_in_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                       const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, int64_t rows, int64_t K,
                       int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                              (int)x_stride, pick.data_ptr<int>(), s0, s1, o0, o1,
                                                              (int)K, (int)slots, (int)E);
    else
        rot_in_kernel<half><<<grid, 32, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()), (int)x_stride,
                                                     pick.data_ptr<int>(), s0, s1, o0, o1, (int)K, (int)slots,
                                                     (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_gateup_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g,
                                const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd, int64_t rows,
                                int64_t P, int64_t N, int64_t SK, int64_t slots, int64_t E, double limit,
                                int64_t act_mode) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    gateup_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_g.data_ptr()),
        reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
        reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)SK, (int)E, (float)limit, (int)act_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(D / 128));
    down_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_combine_cuda(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t D,
                        int64_t slots) {
    dim3 grid((unsigned)rows, (unsigned)((D + 255) / 256));
    combine_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<float>(), wts.data_ptr<float>(),
                                                                        out.data_ptr<float>(), (int)D, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_combine_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                             const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t P, int64_t D, int64_t SK,
                             int64_t slots, int64_t E) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    down_combine_kernel<<<grid, (unsigned)(32 * slots), 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), wts.data_ptr<float>(), out.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E,
        (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_shared_cuda(const at::Tensor& pick_in, const at::Tensor& wts_in, const at::Tensor& x, int64_t x_stride,
                             const c10::optional<at::Tensor>& gate_w, int64_t shared_id, at::Tensor& pick_out,
                             at::Tensor& wts_out, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members,
                             int64_t R, int64_t slots_in, int64_t E) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    const int slots = (int)slots_in + (shared_id >= 0);
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    const size_t smem = (size_t)R * slots * sizeof(int);
    TORCH_CHECK(smem <= 48 * 1024, "grouping window too large");
    const int H = (int)x.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
#define ESP_GS(T)                                                                                              \
    group_shared_kernel<T><<<1, GROUP_THREADS, smem, stream>>>(                                               \
        pick_in.data_ptr<int>(), wts_in.data_ptr<float>(), reinterpret_cast<const T*>(x.data_ptr()), (int)x_stride, \
        gate_w.has_value() ? reinterpret_cast<const T*>(gate_w->data_ptr()) : nullptr, H, (int)shared_id,       \
        pick_out.data_ptr<int>(), wts_out.data_ptr<float>(), uids.data_ptr<int>(), ucount.data_ptr<int>(),     \
        members.data_ptr<int>(), (int)R, (int)slots_in, (int)E, (int)members.size(1))
    if (x.scalar_type() == at::kBFloat16) ESP_GS(__nv_bfloat16);
    else ESP_GS(half);
#undef ESP_GS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_combine2_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              const at::Tensor& wts, at::Tensor& out, const c10::optional<at::Tensor>& out_shared,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E,
                              int64_t n_routed) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    auto stream = at::cuda::getCurrentCUDAStream();
#define ESP_DC(T)                                                                                              \
    down_combine2_kernel<T><<<grid, (unsigned)(32 * slots), 0, stream>>>(                                     \
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),             \
        y.data_ptr<float>(), wts.data_ptr<float>(), reinterpret_cast<T*>(out.data_ptr()),                      \
        out_shared.has_value() ? reinterpret_cast<T*>(out_shared->data_ptr()) : nullptr, (int)P, (int)D, (int)SK, \
        (int)E, (int)slots, (int)n_routed)
    if (out.scalar_type() == at::kBFloat16) ESP_DC(__nv_bfloat16);
    else if (out.scalar_type() == at::kHalf) ESP_DC(half);
    else ESP_DC(float);
#undef ESP_DC
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
