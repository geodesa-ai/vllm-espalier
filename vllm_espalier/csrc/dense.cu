// EXL3 dense linears for decode (1-32 rows): every matrix of one vLLM layer (q/k/v, in_proj_qkv/z) in one launch,
// input in the model's dtype, output written straight into its columns of the layer's output, no casts or cat.
//
// Each matrix keeps its own suh/svh: y_s = H(H(x * suh_s) @ W_s) * svh_s. The grid is (128-column blocks of all the
// layer's matrices, K splits). Up to ROT_SMEM bytes, a program rotates the rows of x over its own K range into shared
// memory itself (the input rotation is fused); above that a rot launch writes fp16 H(x * suh_s) for every matrix
// first. Both compute the same fp16 bits, so the choice is free. Row-invariant: mma keeps rows apart, the K split and
// warp ranges depend on the shape alone, warps and splits are added in a fixed order. After TensorFold's linear.cu
// (Apache-2.0, python-0.6 at ed78d6f); the trellis is read in its stored layout through experts_grouped.cuh.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "experts_grouped.cuh"

using namespace tf_exl3x;

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
constexpr int MAX_SHARDS = 4;

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

struct Shard {
    const uint32_t* T;      // trellis words, stored layout [K/16, N/16, 4 * K2]
    const half* suh;        // [K]
    const half* svh;        // [N]
    int nb0;                // first 128-column block of this matrix in the launch
    int nb;                 // its 128-column blocks
    int col0;               // its first output column
    float scale;            // grid step c of an int5 / int6 codebook (decoded as its integer index), else 1
};

struct Shards {
    Shard s[MAX_SHARDS];
    int n;
};

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

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// fp16 H(x * suh) / sqrt(128) of one row's 128 inputs starting at k, 4 a lane (the same bits in both paths)
__device__ __forceinline__ void rot128(const void* x, int x_dtype, size_t xi, const half* suh, int k, int lane,
                                       half* out) {
    float v[4];
    load4(x, x_dtype, xi + 4 * lane, v);
    const uint2 su = *reinterpret_cast<const uint2*>(suh + k + 4 * lane);
    const float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&su.x));
    const float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&su.y));
    // __fmul_rn: never contracted into the butterfly, so the fused and the rot-launch paths give the same bits
    v[0] = __fmul_rn(v[0], s01.x); v[1] = __fmul_rn(v[1], s01.y); v[2] = __fmul_rn(v[2], s23.x); v[3] = __fmul_rn(v[3], s23.y);
    fwht128(v, lane);
    const half2 a = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
    const half2 b = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&a);
    u.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(out + 4 * lane) = u;
}

// Program (128-block of K, row, matrix): xh[s][row] = fp16 H(x * suh_s) / sqrt(128).
__global__ void __launch_bounds__(128) dense_rot_kernel(Shards A, const void* __restrict__ x, int x_dtype,
                                                        int x_stride, half* __restrict__ xh, int M, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, s = blockIdx.z, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    rot128(x, x_dtype, (size_t)row * x_stride + blk * 128, A.s[s].suh, blk * 128, lane,
           xh + ((size_t)s * M + row) * K + blk * 128);
}

// The finished outputs of one row's 128 columns from their fp32 sums, 4 a lane: H / sqrt(128), * c, * svh.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, float scale) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= scale;
    const uint2 sv = *reinterpret_cast<const uint2*>(svh + 4 * lane);
    const float2 s01 = __half22float2(*reinterpret_cast<const half2*>(&sv.x));
    const float2 s23 = __half22float2(*reinterpret_cast<const half2*>(&sv.y));
    v[0] *= HAD_SCALE * s01.x; v[1] *= HAD_SCALE * s01.y; v[2] *= HAD_SCALE * s23.x; v[3] *= HAD_SCALE * s23.y;
}

template <int CB, int K2, int W, int PF>
__global__ void __launch_bounds__(W * 32) dense_kernel(
    Shards A, const void* __restrict__ x, int x_dtype, int x_stride, const half* __restrict__ xh, void* __restrict__ y,
    int y_dtype, int y_stride, float* __restrict__ Z, int* __restrict__ counters, int M, int K, int SK, int NBT,
    int fused) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int last;
    const int nbg = blockIdx.x, split = blockIdx.y;
    int si = 0;
#pragma unroll
    for (int i = 1; i < MAX_SHARDS; ++i)
        if (i < A.n && nbg >= A.s[i].nb0) si = i;
    const Shard& S = A.s[si];
    const int nbl = nbg - S.nb0, nt0 = nbl * 8, NTILES = S.nb * 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int kr = K / SK, k0 = split * kr;
    const int per_warp = (kr >> 4) / W;
    const int kt0 = (k0 >> 4) + warp * per_warp;
    const int RH = min(M, 8);
    half* xs = reinterpret_cast<half*>(smem);                                   // fused: [16][kr]
    float* red = reinterpret_cast<float*>(smem + (fused ? (size_t)min(M, 16) * kr * sizeof(half) : 0));  // [W][RH][128]
    const half* svh = S.svh + nt0 * 16;

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);
        const half *x0, *x1;
        const bool ok0 = g < R, ok1 = g + 8 < R;
        if (fused) {
            if (m0) __syncthreads();                                            // xs of the last pass is read
            const int chunks = kr >> 7;
            for (int idx = warp; idx < R * chunks; idx += W) {
                const int r = idx / chunks, c = idx % chunks;
                rot128(x, x_dtype, (size_t)(m0 + r) * x_stride + k0 + c * 128, S.suh, k0 + c * 128, lane,
                       xs + (size_t)r * kr + c * 128);
            }
            __syncthreads();
            x0 = xs + (size_t)(ok0 ? g : 0) * kr - k0 + 2 * t;                    // indexed by the absolute k below
            x1 = xs + (size_t)(ok1 ? g + 8 : 0) * kr - k0 + 2 * t;
        } else {
            const half* X = xh + (size_t)si * M * K;
            x0 = X + (size_t)(m0 + (ok0 ? g : 0)) * K + 2 * t;
            x1 = X + (size_t)(m0 + (ok1 ? g + 8 : 0)) * K + 2 * t;
        }

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
        warp_tiles<CB, K2, 8, PF>(S.T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, lane, acc);

        // the warps' sums, added in warp order: rows 0-7 of the pass, then rows 8-15
        for (int hl = 0; hl < R; hl += 8) {
            const int rn = min(R - hl, 8);
            __syncthreads();
            if (g < rn) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            hl ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();
            for (int r = warp; r < rn; r += W) {
                float4 s = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
#pragma unroll
                for (int w = 1; w < W; ++w) {
                    const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                    s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                }
                const int row = m0 + hl + r;
                if (SK == 1) {
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, S.scale);
                    store4(y, y_dtype, (size_t)row * y_stride + S.col0 + nt0 * 16 + 4 * lane, v);
                } else {
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + row) * NBT * 128 + nbg * 128 + 4 * lane) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NBT + nbg, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += W) {
                    const size_t at = (size_t)(m0 + r) * NBT * 128 + nbg * 128 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * NBT * 128 + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, S.scale);
                    store4(y, y_dtype, (size_t)(m0 + r) * y_stride + S.col0 + nt0 * 16 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NBT + nbg] = 0;          // every split of the block arrived
            }
        }
    }
}

// Program (128-block of K x4, row): y = fp16(H(x * pre) / sqrt(128) * post), pre / post optional (the input and
// output transforms of the reconstruct + GEMM path, without exllamav3).
__global__ void __launch_bounds__(128) had_kernel(const half* __restrict__ x, const half* __restrict__ pre,
                                                  const half* __restrict__ post, half* __restrict__ y, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const size_t i = (size_t)row * K + blk * 128 + 4 * lane;
    float v[4];
    load4(x, F16, i, v);
    if (pre) {
        float s[4];
        load4(pre, F16, blk * 128 + 4 * lane, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __fmul_rn(v[j], s[j]);
    }
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    if (post) {
        float s[4];
        load4(post, F16, blk * 128 + 4 * lane, s);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] *= s[j];
    }
    store4(y, F16, i, v);
}

int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace

// y[:, col0_s : col0_s + N_s] = H(H(x * suh_s) @ W_s) * svh_s for every matrix s (all of one codebook and K2).
// xh: fp16 [n, M, K] scratch for the unfused path; Z: fp32 [SK, M, NBT * 128] scratch when SK > 1; counters: int32
// [2 * NBT] zeros (left zero).
void esp_dense_cuda(const at::Tensor& x, const std::vector<at::Tensor>& T, const std::vector<at::Tensor>& suh,
                    const std::vector<at::Tensor>& svh, const std::vector<int64_t>& col0, at::Tensor& y,
                    at::Tensor& xh, at::Tensor& Z, at::Tensor& counters, int64_t k2, int64_t cb, int64_t SK,
                    int64_t W, int64_t PF, int64_t rot_smem) {
    const int M = (int)x.size(0), K = (int)x.size(1), n = (int)T.size();
    TORCH_CHECK(n >= 1 && n <= MAX_SHARDS, "1 to 4 matrices a launch");
    TORCH_CHECK(M >= 1 && M <= 32, "1 to 32 rows");
    TORCH_CHECK(K % (128 * SK) == 0 && (K / 16 / SK) % W == 0, "K must split evenly over SK * W warps in 128s");
    Shards A{};
    int nbt = 0;
    for (int s = 0; s < n; ++s) {
        const int N = (int)svh[s].numel();
        TORCH_CHECK(N % 128 == 0 && T[s].size(1) * 16 == N && T[s].size(0) * 16 == K, "matrix shape");
        A.s[s] = Shard{reinterpret_cast<const uint32_t*>(T[s].data_ptr()),
                       reinterpret_cast<const half*>(suh[s].data_ptr()),
                       reinterpret_cast<const half*>(svh[s].data_ptr()), nbt, N / 128, (int)col0[s],
                       cb == 6 ? 3.94921875f / 15.0f : cb == 7 ? 3.94921875f / 31.0f : 1.0f};
        nbt += N / 128;
    }
    A.n = n;
    TORCH_CHECK(SK == 1 || (Z.numel() >= SK * M * nbt * 128 && counters.numel() >= 2 * nbt), "scratch too small");
    const int kr = K / (int)SK;
    const int fused = (size_t)std::min(M, 16) * kr * sizeof(half) <= (size_t)rot_smem;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (!fused) {
        TORCH_CHECK(xh.numel() >= (int64_t)n * M * K && xh.scalar_type() == at::kHalf, "xh too small");
        dim3 g((unsigned)((K / 128 + 3) / 4), (unsigned)M, (unsigned)n);
        dense_rot_kernel<<<g, 128, 0, stream>>>(A, x.data_ptr(), dtype_of(x), (int)x.stride(0),
                                                reinterpret_cast<half*>(xh.data_ptr()), M, K);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const size_t smem = (fused ? (size_t)std::min(M, 16) * kr * sizeof(half) : 0) +
                        (size_t)W * std::min(M, 8) * 128 * sizeof(float);
    dim3 grid((unsigned)nbt, (unsigned)SK);
#define ESP_DENSE_LAUNCH(CB_, K2_, W_, PF_)                                                                      \
    {                                                                                                           \
        auto kern = dense_kernel<(CB_ == 6 || CB_ == 7) ? CB_ + 16 : CB_, K2_, W_, PF_>;                                                            \
        if (smem > 40 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kern<<<grid, W_ * 32, smem, stream>>>(A, x.data_ptr(), dtype_of(x), (int)x.stride(0),                   \
                                              reinterpret_cast<const half*>(xh.data_ptr()), y.data_ptr(),       \
                                              dtype_of(y), (int)y.stride(0), Z.data_ptr<float>(),               \
                                              counters.data_ptr<int>(), M, K, (int)SK, nbt, fused);             \
    }
#define ESP_DENSE_CASE(CB_, K2_)                                                                                 \
    case CB_ * 32 + K2_:                                                                                        \
        if (W == 4 && PF == 1) ESP_DENSE_LAUNCH(CB_, K2_, 4, 1)                                                 \
        else if (W == 4 && PF == 2) ESP_DENSE_LAUNCH(CB_, K2_, 4, 2)                                            \
        else if (W == 4 && PF == 4) ESP_DENSE_LAUNCH(CB_, K2_, 4, 4)                                            \
        else if (W == 4 && PF == 8) ESP_DENSE_LAUNCH(CB_, K2_, 4, 8)                                            \
        else if (W == 8 && PF == 1) ESP_DENSE_LAUNCH(CB_, K2_, 8, 1)                                            \
        else if (W == 8 && PF == 2) ESP_DENSE_LAUNCH(CB_, K2_, 8, 2)                                            \
        else if (W == 8 && PF == 4) ESP_DENSE_LAUNCH(CB_, K2_, 8, 4)                                            \
        else if (W == 16 && PF == 1) ESP_DENSE_LAUNCH(CB_, K2_, 16, 1)                                          \
        else if (W == 16 && PF == 2) ESP_DENSE_LAUNCH(CB_, K2_, 16, 2)                                          \
        else TORCH_CHECK(false, "W 4 (PF 1, 2, 4, 8), 8 (PF 1, 2, 4) or 16 (PF 1, 2)");                          \
        break;
    switch (cb * 32 + k2) {
        ESP_CASES(ESP_DENSE_CASE)
        default:
            TORCH_CHECK(false, "unsupported codebook/width: cb=", cb, " k2=", k2);
    }
#undef ESP_DENSE_CASE
#undef ESP_DENSE_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void esp_had_cuda(const at::Tensor& x, const c10::optional<at::Tensor>& pre, const c10::optional<at::Tensor>& post,
                  at::Tensor& y) {
    const int M = (int)x.size(0), K = (int)x.size(1);
    TORCH_CHECK(K % 128 == 0, "had: K must be a multiple of 128");
    dim3 grid((unsigned)((K / 128 + 3) / 4), (unsigned)M);
    had_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr()), pre ? reinterpret_cast<const half*>(pre->data_ptr()) : nullptr,
        post ? reinterpret_cast<const half*>(post->data_ptr()) : nullptr, reinterpret_cast<half*>(y.data_ptr()), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
