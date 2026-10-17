// Bindings for the grouped EXL3 expert kernels. Vendored from TensorFold (Apache-2.0, python-0.6 at ed78d6f,
// src/tensorfold/cuda/exl3/experts.cpp); Espalier change: per-projection codebook tables in `grouped`.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3x_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void exl3x_dequant_cuda(const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_group_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_rot_in_cuda(const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t);
void exl3x_gateup_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                                double, int64_t);
void exl3x_down_epilogue_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t);
void exl3x_combine_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_group_shared_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
                             const c10::optional<at::Tensor>&, int64_t, at::Tensor&, at::Tensor&, at::Tensor&,
                             at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t);
void exl3x_down_combine2_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
                              at::Tensor&, const c10::optional<at::Tensor>&, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t);
void exl3x_down_combine_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
                             at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

void esp_dense_cuda(const at::Tensor&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                    const std::vector<at::Tensor>&, const std::vector<int64_t>&, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

void esp_route_group_cuda(const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&, int64_t, int64_t,
                          at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);

void esp_had_cuda(const at::Tensor&, const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, at::Tensor&);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& C0, const at::Tensor& C1,
             const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members, at::Tensor Z, int64_t mats,
             int64_t K, int64_t N, int64_t P, int64_t SK, int64_t slots, int64_t nt, int64_t warps, int64_t pf,
             int64_t lo, int64_t hi) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(C0, at::kInt, "C0");
    check(C1, at::kInt, "C1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3x_grouped_cuda(X0, X1, TP0, TP1, B0, B1, C0, C1, uids, ucount, members, Z, mats, K, N, P, SK, slots, nt,
                       warps, pf, lo, hi);
}

void dequant(const at::Tensor& T, at::Tensor out, int64_t k2, int64_t cb) {
    TORCH_CHECK(T.is_cuda() && T.scalar_type() == at::kShort && T.is_contiguous() && T.dim() == 3,
                "T: int16 [K/16, N/16, 8 * k2]");
    TORCH_CHECK(T.size(2) == 8 * k2, "trellis last dim must be 8 * k2");
    check(out, at::kHalf, "out");
    const int64_t K = T.size(0) * 16, N = T.size(1) * 16;
    TORCH_CHECK(out.numel() == K * N, "out must be [K, N]");
    c10::cuda::CUDAGuard guard(T.device());
    exl3x_dequant_cuda(T, out, K, N, k2, cb);
}

void group(const at::Tensor& pick, at::Tensor uids, at::Tensor ucount, at::Tensor members, int64_t R, int64_t slots,
           int64_t E) {
    check(pick, at::kInt, "pick");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(pick.numel() >= R * slots, "pick too small");
    TORCH_CHECK(uids.numel() >= std::min<int64_t>(R * slots, E), "uids too small");
    TORCH_CHECK(members.size(0) >= uids.numel() && members.size(1) >= 1, "members too small");
    c10::cuda::CUDAGuard guard(pick.device());
    exl3x_group_cuda(pick, uids, ucount, members, R, slots, E);
}

void rot_in(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
            const at::Tensor& suh1, at::Tensor out0, at::Tensor out1, int64_t rows, int64_t K, int64_t slots,
            int64_t E) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: bf16/fp16 CUDA");
    check(pick, at::kInt, "pick");
    check(suh0, at::kHalf, "suh0");
    check(suh1, at::kHalf, "suh1");
    check(out0, at::kHalf, "out0");
    check(out1, at::kHalf, "out1");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_rot_in_cuda(x, x_stride, pick, suh0, suh1, out0, out1, rows, K, slots, E);
}

void gateup_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g, const at::Tensor& svh_u,
                     const at::Tensor& suh_d, at::Tensor xd, int64_t rows, int64_t P, int64_t N, int64_t SK,
                     int64_t slots, int64_t E, double limit, int64_t act_mode) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_g, at::kHalf, "svh_g");
    check(svh_u, at::kHalf, "svh_u");
    check(suh_d, at::kHalf, "suh_d");
    check(xd, at::kHalf, "xd");
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_gateup_epilogue_cuda(Z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, E, limit, act_mode);
}

void down_epilogue(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_epilogue_cuda(Z, pick, svh_d, y, rows, P, D, SK, slots, E);
}

void combine(const at::Tensor& y, const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t D, int64_t slots) {
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    c10::cuda::CUDAGuard guard(y.device());
    exl3x_combine_cuda(y, wts, out, rows, D, slots);
}

void down_combine(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y,
                  const at::Tensor& wts, at::Tensor out, int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots,
                  int64_t E) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    check(out, at::kFloat, "out");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_combine_cuda(Z, pick, svh_d, y, wts, out, rows, P, D, SK, slots, E);
}

// Espalier: shared expert as an extra slot (pick/weight tables built and grouped in one launch)
void group_shared(const at::Tensor& pick_in, const at::Tensor& wts_in, const at::Tensor& x, int64_t x_stride,
                  const c10::optional<at::Tensor>& gate_w, int64_t shared_id, at::Tensor pick_out, at::Tensor wts_out,
                  at::Tensor uids, at::Tensor ucount, at::Tensor members, int64_t R, int64_t slots_in, int64_t E) {
    check(pick_in, at::kInt, "pick_in");
    check(wts_in, at::kFloat, "wts_in");
    check(pick_out, at::kInt, "pick_out");
    check(wts_out, at::kFloat, "wts_out");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf), "x: bf16/fp16 CUDA");
    if (gate_w.has_value())
        TORCH_CHECK(gate_w->scalar_type() == x.scalar_type() && gate_w->is_contiguous() && gate_w->numel() == x.size(1),
                    "gate_w: x's dtype, [H]");
    TORCH_CHECK(shared_id < 0 || gate_w.has_value(), "a shared slot needs its gate weight");
    c10::cuda::CUDAGuard guard(x.device());
    exl3x_group_shared_cuda(pick_in, wts_in, x, x_stride, gate_w, shared_id, pick_out, wts_out, uids, ucount, members,
                            R, slots_in, E);
}

// Espalier: down epilogue + combine writing the routed sum and the shared slot in the model's dtype
void down_combine2(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor y,
                   const at::Tensor& wts, at::Tensor out, const c10::optional<at::Tensor>& out_shared, int64_t rows,
                   int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E, int64_t n_routed) {
    check(Z, at::kFloat, "Z");
    check(pick, at::kInt, "pick");
    check(svh_d, at::kHalf, "svh_d");
    check(y, at::kFloat, "y");
    check(wts, at::kFloat, "wts");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out: contiguous CUDA");
    if (out_shared.has_value())
        TORCH_CHECK(out_shared->scalar_type() == out.scalar_type() && out_shared->is_contiguous(), "out_shared: out's dtype");
    TORCH_CHECK(D % 128 == 0, "D must be a multiple of 128");
    TORCH_CHECK(n_routed <= slots, "n_routed <= slots");
    c10::cuda::CUDAGuard guard(Z.device());
    exl3x_down_combine2_cuda(Z, pick, svh_d, y, wts, out, out_shared, rows, P, D, SK, slots, E, n_routed);
}

// Dense layer (Espalier, csrc/dense.cu): every matrix of a vLLM layer in one launch, written into its columns of y.
void dense(const at::Tensor& x, const std::vector<at::Tensor>& T, const std::vector<at::Tensor>& suh,
           const std::vector<at::Tensor>& svh, const std::vector<int64_t>& col0, at::Tensor y, at::Tensor xh,
           at::Tensor Z, at::Tensor counters, int64_t k2, int64_t cb, int64_t SK, int64_t W, int64_t PF,
           int64_t rot_smem) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1, "x: CUDA [M, K], unit column stride");
    TORCH_CHECK(y.is_cuda() && y.dim() == 2 && y.stride(1) == 1, "y: CUDA [M, N], unit column stride");
    TORCH_CHECK(T.size() == suh.size() && T.size() == svh.size() && T.size() == col0.size(), "one entry a matrix");
    for (size_t s = 0; s < T.size(); ++s) {
        TORCH_CHECK(T[s].is_cuda() && T[s].scalar_type() == at::kShort && T[s].is_contiguous() && T[s].dim() == 3,
                    "trellis: contiguous int16 [K/16, N/16, 16K]");
        check(suh[s], at::kHalf, "suh");
        check(svh[s], at::kHalf, "svh");
        TORCH_CHECK(col0[s] + svh[s].numel() <= y.size(1), "matrix columns past y");
    }
    check(Z, at::kFloat, "Z");
    check(counters, at::kInt, "counters");
    c10::cuda::CUDAGuard guard(x.device());
    esp_dense_cuda(x, T, suh, svh, col0, y, xh, Z, counters, k2, cb, SK, W, PF, rot_smem);
}

// Router GEMV + softmax/top-k + grouping in one launch (Espalier, csrc/route.cu).
void route_group(const at::Tensor& x, const at::Tensor& gate, const c10::optional<at::Tensor>& sgate, int64_t k,
                 int64_t shared_id, at::Tensor logits, at::Tensor counter, at::Tensor pick_out, at::Tensor wts_out,
                 at::Tensor uids, at::Tensor ucount, at::Tensor members) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1 && gate.is_contiguous(), "x / gate layout");
    TORCH_CHECK(gate.scalar_type() == x.scalar_type(), "gate and x must share a dtype");
    check(logits, at::kFloat, "logits");
    check(counter, at::kInt, "counter");
    c10::cuda::CUDAGuard guard(x.device());
    esp_route_group_cuda(x, gate, sgate, k, shared_id, logits, counter, pick_out, wts_out, uids, ucount, members);
}

// y = fp16(H(x * pre) / sqrt(128) * post) along rows of 128 (Espalier, csrc/dense.cu).
void had(const at::Tensor& x, const c10::optional<at::Tensor>& pre, const c10::optional<at::Tensor>& post,
         at::Tensor y) {
    check(x, at::kHalf, "x");
    check(y, at::kHalf, "y");
    c10::cuda::CUDAGuard guard(x.device());
    esp_had_cuda(x, pre, post, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("had", &had);
    m.def("route_group", &route_group);
    m.def("dense", &dense);
    m.def("group_shared", &group_shared);
    m.def("down_combine2", &down_combine2);
    m.def("grouped", &grouped);
    m.def("dequant", &dequant);
    m.def("group", &group);
    m.def("rot_in", &rot_in);
    m.def("gateup_epilogue", &gateup_epilogue);
    m.def("down_epilogue", &down_epilogue);
    m.def("combine", &combine);
    m.def("down_combine", &down_combine);
}
