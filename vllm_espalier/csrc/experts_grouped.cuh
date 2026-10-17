// Grouped EXL3 expert GEMV (after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): rows stay independent, K ranges fixed by shape, warps summed in order.
//
// Vendored from TensorFold (Apache-2.0, https://github.com/ashhart/TensorFold, python-0.6 at ed78d6f,
// src/tensorfold/cuda/exl3/experts_grouped.cuh). Espalier changes: the Espalier codebooks (E2M1-grid, int5-grid and int6-grid
// snaps of mcg, bit-identical to the format's reference decode) and a codebook per expert matrix, so one launch
// serves a layer whose experts mix widths and codebooks; the instance set is the (width, codebook) pairs Espalier
// checkpoints use.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace tf_exl3x {

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
// Espalier codebooks (exllamav3 kernel ids): 3 = the mcg value snapped to ESP_E2M1_C * E2M1, 6 / 7 = snapped to the
// int5 / int6 grid c * round(v / c), c = max|mcg| / top. Computed as in the reference decode's decode_3inst_2<3/6/7>.
#define ESP_E2M1_C 0.75f

template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 3) {
        // |v| against C * {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5} (exact in fp16), summing the grid steps, then v's sign
        uint32_t m = cb_pair<1>(s0, s1);
        half2 v = *reinterpret_cast<half2*>(&m);
        half2 a = __habs2(v);
        const float C = ESP_E2M1_C;
        half2 acc = __hmul2(__hgt2(a, __float2half2_rn(0.25f * C)), __float2half2_rn(0.5f * C));
        acc = __hfma2(__hgt2(a, __float2half2_rn(0.75f * C)), __float2half2_rn(0.5f * C), acc);
        acc = __hfma2(__hgt2(a, __float2half2_rn(1.25f * C)), __float2half2_rn(0.5f * C), acc);
        acc = __hfma2(__hgt2(a, __float2half2_rn(1.75f * C)), __float2half2_rn(0.5f * C), acc);
        acc = __hfma2(__hgt2(a, __float2half2_rn(2.5f * C)), __float2half2_rn(1.0f * C), acc);
        acc = __hfma2(__hgt2(a, __float2half2_rn(3.5f * C)), __float2half2_rn(1.0f * C), acc);
        acc = __hfma2(__hgt2(a, __float2half2_rn(5.0f * C)), __float2half2_rn(2.0f * C), acc);
        return *reinterpret_cast<uint32_t*>(&acc) | (m & 0x80008000u);
    } else if constexpr (CB == 6 || CB == 7) {
        // rintf(v * k), k the reciprocal of c that reproduces rintf(v / c) for every fp16 v (checked exhaustively)
        constexpr float c = 3.94921875f / (CB == 6 ? 15.0f : 31.0f);
        constexpr float k = CB == 6 ? 0x1.e62c10p+1f : 0x1.f660bap+2f;
        // no clamp to +-top: max|mcg| = 3.94921875 = c * top, so |rintf(v * k)| <= top for all 65536 states
        // (checked exhaustively over the states)
        uint32_t m = cb_pair<1>(s0, s1);
        float2 v = __half22float2(*reinterpret_cast<half2*>(&m));
        half2 r = __floats2half2_rn(rintf(v.x * k) * c, rintf(v.y * k) * c);
        return *reinterpret_cast<uint32_t*>(&r);
    } else if constexpr (CB == 22 || CB == 23) {
        // int5 / int6 grid index q = rintf(v * k) itself, exact in fp16; the caller multiplies its sums by c
        constexpr float k = CB == 22 ? 0x1.e62c10p+1f : 0x1.f660bap+2f;
        uint32_t m = cb_pair<1>(s0, s1);
        float2 v = __half22float2(*reinterpret_cast<half2*>(&m));
        half2 r = __floats2half2_rn(rintf(v.x * k), rintf(v.y * k));
        return *reinterpret_cast<uint32_t*>(&r);
    } else if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    // windows sharing one 64-bit merge (tests/cuda/test_exl3_experts.py checks every K2)
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    // right shift of window j of a run (run starts at an even position) relative to the run's last window
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves.
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        // 4 bits: lane L's windows are exactly words L-1 and L (the GLM kernel's decode)
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

// One warp's k tiles [kt0, kt0 + nkt) of an expert matrix into acc, PF tiles in flight.
template <int CB, int K2, int NT, int PF>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                           float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[4] = {load_pair(x0 + k, ok0), load_pair(x1 + k, ok1), load_pair(x0 + k + 8, ok0),
                                 load_pair(x1 + k + 8, ok1)};
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

// The (codebook, K2) pairs compiled: mcg at every integer width 1..8 bits; E2M1 at 1..3 bits, int5 at 2..4, int6 at
// 3..5 (the reference decode's slot set).
#define ESP_CASES(X) \
    X(1, 2) X(1, 4) X(1, 6) X(1, 8) X(1, 10) X(1, 12) X(1, 14) X(1, 16) \
    X(3, 2) X(3, 4) X(3, 6) X(6, 4) X(6, 6) X(6, 8) X(7, 6) X(7, 8) X(7, 10)

// The K2 values an instance covering [LO, HI] compiles (half-bits 2..16).
__host__ __device__ constexpr bool k2_supported(int k2) {
    return k2 >= 2 && k2 <= 16;
}

// Program (expert u, n block, split and member tile): up to 16 members times W_q over the split's K range; warps added in order.
template <int NT, int W, int PF, int LO, int HI>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ CB_0, const int* __restrict__ CB_1, const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int cb = mat ? CB_1[e] : CB_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    switch (cb * 32 + k2) {      // warp-uniform: one (codebook, width) per expert matrix
#define TF_EXL3X_CASE(CB_, K2_)                                                                                 \
    case CB_ * 32 + K2_:                                                                                        \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB_, K2_, NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);   \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        ESP_CASES(TF_EXL3X_CASE)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }

    // warps' partial sums through shared memory, added in warp order
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

// W_q [K, N] fp16 of one matrix through the same lane decode (tests; not on the forward path).
template <int CB, int K2>
__global__ void dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int K, int N) {
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x;
    const int NTILES = N >> 4;
    uint32_t w[Fmt<K2>::LW];
    load_words<K2>(w, T + ((size_t)kt * NTILES + nt) * Fmt<K2>::TW + lane, lane);
    const LaneMap<K2> map(lane);
    uint32_t b0[2], b1[2];
    decode_tile<CB, K2>(w, map, lane, b0, b1);
    const int g = lane >> 2, t = lane & 3;
    uint32_t v[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h = *reinterpret_cast<half2*>(&v[q]);
        const int col = nt * 16 + g + 8 * (q >> 1);
        const int row = kt * 16 + 2 * t + 8 * (q & 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

struct GroupedArgs {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* cb_0;
    const int* cb_1;
    const int* uids;
    const int* ucount;
    const int* members;
    float* z;
    int K, N, P, SK, maxm, slots;
    int nexp_max;        // grid.x (upper bound of distinct experts)
    int mats, nt, warps, pf, lo, hi;
};

inline void grouped_launch(const GroupedArgs& a, cudaStream_t stream) {
    const int MT = (a.maxm + 15) / 16;
    dim3 grid((unsigned)a.nexp_max, (unsigned)(a.N / (16 * a.nt)), (unsigned)(a.mats * a.SK * MT));
#define TF_LAUNCH(NT_, W_, PF_, LO_, HI_)                                                                       \
    grouped_kernel<NT_, W_, PF_, LO_, HI_><<<grid, W_ * 32, 0, stream>>>(                                       \
        a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.cb_0, a.cb_1, a.uids, a.ucount, a.members, a.z, a.K, a.N, a.P, \
        a.SK, a.maxm, a.slots)
#define TF_RANGES(NT_, W_, PF_)                                                                                 \
    if (a.lo >= 2 && a.hi <= 6) TF_LAUNCH(NT_, W_, PF_, 2, 6);                                                  \
    else if (a.lo >= 2 && a.hi <= 10) TF_LAUNCH(NT_, W_, PF_, 2, 10);                                           \
    else TF_LAUNCH(NT_, W_, PF_, 2, 16);
    if (a.nt == 8 && a.warps == 4 && a.pf == 1) { TF_RANGES(8, 4, 1) }
    else if (a.nt == 8 && a.warps == 4 && a.pf == 2) { TF_RANGES(8, 4, 2) }
    else if (a.nt == 4 && a.warps == 4 && a.pf == 2) { TF_RANGES(4, 4, 2) }
    else TORCH_CHECK(false, "unsupported tile setting nt=", a.nt, " warps=", a.warps, " pf=", a.pf);
#undef TF_RANGES
#undef TF_LAUNCH
}

inline void dequant_launch(const uint32_t* t, half* o, int K, int N, int k2, int cb, cudaStream_t stream) {
    dim3 grid((unsigned)(K / 16), (unsigned)(N / 16));
    switch (cb * 32 + k2) {
#define TF_DQ(CB_, K2_) case CB_ * 32 + K2_: dequant_kernel<CB_, K2_><<<grid, 32, 0, stream>>>(t, o, K, N); break;
        ESP_CASES(TF_DQ)
#undef TF_DQ
        default: TORCH_CHECK(false, "no instance for codebook ", cb, " K2 ", k2);
    }
}

}  // namespace tf_exl3x
