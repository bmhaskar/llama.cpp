#include "gdn-chunked.cuh"

// ============================================================================
// Chunked Gated DeltaNet (scalar gate, non-KDA).
//
// Sequential recurrence implemented by gated_delta_net.cu, per token t:
//     g_t   = exp(g_raw_t)                       (scalar per head per token)
//     u_t   = beta_t * (v_t - (g_t S_{t-1})^T k_t)
//     S_t   = g_t S_{t-1} + k_t u_t^T
//     o_t   = S_t^T q_t                          (uses the UPDATED state)
//
// Within a chunk of C tokens starting from state S0, with
// cg_t = sum_{j<=t} g_raw_j   (so G_t = exp(cg_t) is the cumulative decay):
//
//     S_t = G_t S0 + sum_{j<=t} (G_t/G_j) k_j u_j^T
//
// Substituting into u_t and collecting terms gives a triangular system.
// With rows indexed by token (K,Q,V,U are C x D, S0 is D x D):
//
//     A   = tril(Kbar Khat^T, -1),  Kbar_t = G_t k_t,  Khat_j = k_j / G_j
//     M   = I + diag(beta) A                     (unit lower triangular)
//     U   = M^{-1} diag(beta) (V - Gamma K S0)
//     P   = tril(Qbar Khat^T, 0)                 (diagonal included: o_t uses S_t)
//     O   = Gamma Q S0 + P U
//     S_C = G_C S0 + (Gamma_C Khat)^T U
//
// Every exponential is exp(cg_t - cg_j) with t >= j. g_raw <= 0 (it is a log
// decay), so the exponent is <= 0 and cannot overflow. Khat is never formed
// explicitly for that reason - the decay is always applied as a difference.
//
// This mirrors the structure of the Mamba-2 SSD path in ssm-scan.cu; the extra
// piece is the triangular solve, which the delta rule needs and SSD does not.
// ============================================================================


// ---------------------------------------------------------------------------
// STATUS, measured 2026-09-16 on RTX 5070 Ti (sm_120), Qwen3.8-27B:
//
//   correctness  matches the CPU reference; NMSE 1.45e-7 .. 2.0e-7 against the
//                harness' 1e-7 threshold. fp32 epsilon is 1.19e-7, so this is
//                summation-order noise, not error: NMSE is FLAT from 128 to
//                2048 tokens (1.57 / 1.45 / 1.64 / 1.95 / 1.85 e-7), i.e. the
//                recurrence is numerically stable. Partial final chunks work.
//
//   speed        NOT yet a win. End-to-end prefill at the real shape:
//                sequential 1499 tok/s vs chunked 1485 tok/s (~1% slower).
//                Isolated at head_count=4 it is 3.6x slower (547 vs 153 us).
//                The shape dependence is the diagnosis: this issues ~15 kernel
//                launches per chunk (7 cuBLAS + 8 elementwise) x n/64 chunks,
//                so ~120 launches per op. At 4 heads that overhead dominates
//                completely; at 48 heads the GEMMs are large enough to nearly
//                amortise it, landing at parity.
//
//   to win       fuse into ONE launch per op: chunk loop inside the kernel,
//                state tile resident in registers, and the GEMMs done with
//                in-kernel tensor cores (mma.sync m16n8k8 f32.tf32.tf32.f32 is
//                available on sm_120 via mma.cuh and would also lift the ~7%
//                of fp32 peak the sequential kernel currently achieves). That
//                removes both the launch overhead and the global-memory
//                round-trip through the scratch buffers below.
//
//   not covered  K > 1 (rollback snapshots). serve.sh runs --spec-draft-n-max 4
//                => K=5, so this path does not engage there at all. The fix is
//                the split: chunked over tokens [0, n-K), then the existing
//                sequential kernel over the final K tokens seeded with the
//                carried state - it alone writes the snapshot slots.
// ---------------------------------------------------------------------------

#define GDN_CHUNK 64   // tokens per chunk; also the size of the triangular solve

// ---------------------------------------------------------------------------
// Per (seq, head): local cumulative sum of g_raw over the chunk.
// ---------------------------------------------------------------------------
__global__ void gdn_cumsum_kernel(
        const float * __restrict__ g, float * __restrict__ cg,
        int64_t H, int64_t n_tokens, int64_t chunk_off, int64_t chunk_len,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t Ccap) {
    const int64_t h   = blockIdx.x;
    const int64_t seq = blockIdx.y;
    if (threadIdx.x != 0) return;

    float acc = 0.0f;
    float * out = cg + (seq * H + h) * Ccap;
    for (int64_t t = 0; t < chunk_len; t++) {
        acc += g[seq * sb3 + (chunk_off + t) * sb2 + h * sb1];
        out[t] = acc;
    }
}

// ---------------------------------------------------------------------------
// Expand q/k from n_k_heads to H value heads (head h uses k-head h % neqk1),
// so every later GEMM can use uniform strided batching.
// ---------------------------------------------------------------------------
__global__ void gdn_expand_qk_kernel(
        const float * __restrict__ q, const float * __restrict__ k,
        float * __restrict__ q_e, float * __restrict__ k_e,
        int64_t S_v, int64_t H, int64_t chunk_off, int64_t chunk_len,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t neqk1, int64_t rq3, int64_t Ccap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_len * S_v) return;

    const int64_t t = idx / S_v;
    const int64_t i = idx % S_v;
    const int64_t h   = blockIdx.y;
    const int64_t seq = blockIdx.z;

    const int64_t iq1 = h % neqk1;
    const int64_t iq3 = seq / rq3;
    const int64_t src = iq3 * sq3 + (chunk_off + t) * sq2 + iq1 * sq1 + i;
    // block stride is the FIXED chunk size: the cuBLAS batch strides below are
    // GDN_CHUNK-based, so a short final chunk must still use the full stride.
    const int64_t dst = (seq * H + h) * Ccap * S_v + t * S_v + i;

    q_e[dst] = q[src];
    k_e[dst] = k[src];
}

// ---------------------------------------------------------------------------
// Build M = I + diag(beta) tril(Kbar Khat^T, -1) and P = tril(Qbar Khat^T, 0)
// from the raw Gram matrices, applying the decay and the causal mask.
// KKT/QKT are [chunk_len x chunk_len] per (seq, head), row-major t (i) by j.
// ---------------------------------------------------------------------------
__global__ void gdn_build_MP_kernel(
        const float * __restrict__ KKT, const float * __restrict__ QKT,
        const float * __restrict__ cg,  const float * __restrict__ beta,
        float * __restrict__ M, float * __restrict__ P,
        int64_t H, int64_t chunk_off, int64_t chunk_len,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t Ccap) {
    const int64_t h   = blockIdx.y;
    const int64_t seq = blockIdx.z;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_len * chunk_len) return;

    const int64_t t = idx / chunk_len;   // row
    const int64_t j = idx % chunk_len;   // col

    const int64_t hs   = (seq * H + h);
    const float * cg_h = cg + hs * Ccap;
    const int64_t base = hs * Ccap * Ccap;

    const float beta_t = beta[seq * sb3 + (chunk_off + t) * sb2 + h * sb1];

    // decay across [j, t]; always <= 0 in the exponent
    const float decay = (j <= t) ? expf(cg_h[t] - cg_h[j]) : 0.0f;

    float m_val;
    if (j < t)       m_val = beta_t * decay * KKT[base + idx];
    else if (j == t) m_val = 1.0f;                 // unit diagonal
    else             m_val = 0.0f;
    M[base + idx] = m_val;

    P[base + idx] = (j <= t) ? decay * QKT[base + idx] : 0.0f;
}

// ---------------------------------------------------------------------------
// W = M^{-1} diag(beta), M unit lower triangular.
// One block per (seq, head). Row i of the inverse depends only on rows < i, so
// the rows are produced in order while all columns are computed in parallel.
// ---------------------------------------------------------------------------
__global__ void gdn_invert_M_kernel(
        float * __restrict__ M, const float * __restrict__ beta,
        int64_t H, int64_t chunk_off, int64_t chunk_len,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t Ccap) {
    extern __shared__ float sm[];
    float * Ts = sm;                      // chunk_len * chunk_len

    const int64_t h   = blockIdx.x;
    const int64_t seq = blockIdx.y;
    const int64_t base = (seq * H + h) * Ccap * Ccap;

    // Stage M in shared when it fits (measured ~2% faster at C=64); for large C
    // both matrices would exceed the shared limit, so read M from global instead.
    const bool m_shared = (2 * chunk_len * chunk_len * (int64_t) sizeof(float)) <= 48 * 1024;
    const float * Ms = m_shared ? (sm + chunk_len * chunk_len) : (M + base);
    if (m_shared) {
        float * dst = sm + chunk_len * chunk_len;
        for (int64_t idx = threadIdx.x; idx < chunk_len * chunk_len; idx += blockDim.x) {
            dst[idx] = M[base + idx];
        }
    }
    for (int64_t idx = threadIdx.x; idx < chunk_len * chunk_len; idx += blockDim.x) {
        Ts[idx] = 0.0f;
    }
    __syncthreads();

    // T[i][j] = -sum_{k=j..i-1} M[i][k] T[k][j], T[i][i] = 1
    for (int64_t i = 0; i < chunk_len; i++) {
        for (int64_t j = threadIdx.x; j <= (int64_t) i; j += blockDim.x) {
            if (j == i) {
                Ts[i * chunk_len + j] = 1.0f;
            } else {
                float acc = 0.0f;
                for (int64_t k = j; k < i; k++) {
                    acc += Ms[i * chunk_len + k] * Ts[k * chunk_len + j];
                }
                Ts[i * chunk_len + j] = -acc;
            }
        }
        __syncthreads();
    }

    // fold diag(beta) on the right: W[i][j] = T[i][j] * beta_j
    for (int64_t idx = threadIdx.x; idx < chunk_len * chunk_len; idx += blockDim.x) {
        const int64_t j = idx % chunk_len;
        const float beta_j = beta[seq * sb3 + (chunk_off + j) * sb2 + h * sb1];
        M[base + idx] = Ts[idx] * beta_j;
    }
}

// ---------------------------------------------------------------------------
// rhs = V - Gamma (K S0)     (KS holds K S0 on entry, rhs on exit)
// ---------------------------------------------------------------------------
__global__ void gdn_make_rhs_kernel(
        float * __restrict__ KS, const float * __restrict__ v,
        const float * __restrict__ cg,
        int64_t S_v, int64_t H, int64_t chunk_off, int64_t chunk_len,
        int64_t sv1, int64_t sv2, int64_t sv3, int64_t Ccap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_len * S_v) return;

    const int64_t t   = idx / S_v;
    const int64_t c   = idx % S_v;
    const int64_t h   = blockIdx.y;
    const int64_t seq = blockIdx.z;

    const int64_t hs = (seq * H + h);
    const float G_t  = expf(cg[hs * Ccap + t]);
    const float v_val = v[seq * sv3 + (chunk_off + t) * sv2 + h * sv1 + c];

    float * dst = KS + hs * Ccap * S_v + idx;
    *dst = v_val - G_t * (*dst);
}

// ---------------------------------------------------------------------------
// out = scale * (Gamma (Q S0) + PU)
// QS holds Q S0, PU holds P U.
// ---------------------------------------------------------------------------
__global__ void gdn_finish_out_kernel(
        const float * __restrict__ QS, const float * __restrict__ PU,
        const float * __restrict__ cg, float * __restrict__ dst,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t chunk_off, int64_t chunk_len,
        float scale, int64_t Ccap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_len * S_v) return;

    const int64_t t   = idx / S_v;
    const int64_t c   = idx % S_v;
    const int64_t h   = blockIdx.y;
    const int64_t seq = blockIdx.z;

    const int64_t hs = (seq * H + h);
    const float G_t  = expf(cg[hs * Ccap + t]);
    const int64_t off = hs * Ccap * S_v + idx;

    // dst layout matches the sequential kernel: [S_v, H, n_tokens, n_seqs]
    dst[(seq * n_tokens + chunk_off + t) * H * S_v + h * S_v + c] =
        scale * (G_t * QS[off] + PU[off]);
}

// ---------------------------------------------------------------------------
// Scale K rows by (G_C / G_j) so that S_C = G_C S0 + (that)^T U in one GEMM.
// ---------------------------------------------------------------------------
__global__ void gdn_scale_k_kernel(
        const float * __restrict__ k_e, float * __restrict__ k_s,
        const float * __restrict__ cg,
        int64_t S_v, int64_t H, int64_t chunk_len, int64_t Ccap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_len * S_v) return;

    const int64_t t   = idx / S_v;
    const int64_t h   = blockIdx.y;
    const int64_t seq = blockIdx.z;

    const int64_t hs = (seq * H + h);
    const float * cg_h = cg + hs * Ccap;
    const float w = expf(cg_h[chunk_len - 1] - cg_h[t]);

    const int64_t off = hs * Ccap * S_v + idx;
    k_s[off] = w * k_e[off];
}

// ---------------------------------------------------------------------------
// S *= G_C   (applied before the rank-C update is accumulated into it)
// ---------------------------------------------------------------------------
__global__ void gdn_decay_state_kernel(
        float * __restrict__ S, const float * __restrict__ cg,
        int64_t S_v, int64_t H, int64_t chunk_len, int64_t Ccap) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S_v * S_v) return;

    const int64_t h   = blockIdx.y;
    const int64_t seq = blockIdx.z;
    const int64_t hs  = (seq * H + h);

    S[hs * S_v * S_v + idx] *= expf(cg[hs * Ccap + chunk_len - 1]);
}

// ---------------------------------------------------------------------------
// Orchestration. All GEMMs are strided-batched over (seq, head).
//
// cuBLAS is column-major; every matrix here is stored row-major. A row-major
// (m x n) matrix is the same bytes as a column-major (n x m), so each call
// below is expressed in terms of the transposed problem.
// ---------------------------------------------------------------------------
bool ggml_cuda_gdn_chunked_try(
        ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K) {

    // Only the plain prefill case: long sequences, no rollback snapshots.
    // K > 1 needs per-token state snapshots for the last K-1 tokens, which the
    // chunked form does not produce; that is left to the sequential kernel.
    if (n_tokens < 128 || K > 1) {
        return false;
    }

    // OPT-IN. Measured at the real shape (H=48, D=128, 512-token ubatches) this
    // path is ~1% slower end-to-end than the sequential kernel, so it is off by
    // default; set GGML_GDN_CHUNKED=1 to enable. See the note at the top of this
    // file for what it would take to make it win.
    static const bool enabled = getenv("GGML_GDN_CHUNKED") && atoi(getenv("GGML_GDN_CHUNKED")) != 0;
    if (!enabled) {
        return false;
    }

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const int64_t HS = H * n_seqs;

    // Tunable: chunk size trades kernel-launch count (n/C) against O(C^2) work
    // and shared memory in the triangular solve.
    static const int64_t C = [] {
        const char * e = getenv("GDN_CHUNK_SIZE");
        const int64_t v = e ? atoi(e) : GDN_CHUNK;
        return (v >= 16 && v <= 256 && (v % 16) == 0) ? v : (int64_t) GDN_CHUNK;
    }();

    // Tunable: TF32 tensor cores for the GEMMs (fp32 accumulate, ~10-bit mantissa).
    static const cublasComputeType_t comp = getenv("GDN_TF32") && atoi(getenv("GDN_TF32"))
        ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    const int64_t D  = S_v;

    ggml_cuda_pool_alloc<float> cg_buf (ctx.pool(), HS * C);
    ggml_cuda_pool_alloc<float> qe_buf (ctx.pool(), HS * C * D);
    ggml_cuda_pool_alloc<float> ke_buf (ctx.pool(), HS * C * D);
    ggml_cuda_pool_alloc<float> ks_buf (ctx.pool(), HS * C * D);
    ggml_cuda_pool_alloc<float> kkt_buf(ctx.pool(), HS * C * C);
    ggml_cuda_pool_alloc<float> qkt_buf(ctx.pool(), HS * C * C);
    ggml_cuda_pool_alloc<float> M_buf  (ctx.pool(), HS * C * C);
    ggml_cuda_pool_alloc<float> P_buf  (ctx.pool(), HS * C * C);
    ggml_cuda_pool_alloc<float> KS_buf (ctx.pool(), HS * C * D);
    ggml_cuda_pool_alloc<float> QS_buf (ctx.pool(), HS * C * D);
    ggml_cuda_pool_alloc<float> U_buf  (ctx.pool(), HS * C * D);
    ggml_cuda_pool_alloc<float> PU_buf (ctx.pool(), HS * C * D);

    float * cg  = cg_buf.get();
    float * q_e = qe_buf.get();
    float * k_e = ke_buf.get();
    float * k_s = ks_buf.get();
    float * KKT = kkt_buf.get();
    float * QKT = qkt_buf.get();
    float * M   = M_buf.get();
    float * P   = P_buf.get();
    float * KS  = KS_buf.get();
    float * QS  = QS_buf.get();
    float * U   = U_buf.get();
    float * PU  = PU_buf.get();

    // running state, [D, D, H, n_seqs], initialised from s_d
    CUDA_CHECK(cudaMemcpyAsync(state_d, s_d, HS * D * D * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    const float one = 1.0f, zero = 0.0f;
    const int64_t n_chunks = (n_tokens + C - 1) / C;

    for (int64_t kc = 0; kc < n_chunks; kc++) {
        const int64_t off = kc * C;
        const int64_t len = (off + C <= n_tokens) ? C : (n_tokens - off);

        // --- per-chunk scalars and expanded q/k -----------------------------
        gdn_cumsum_kernel<<<dim3(H, n_seqs), 32, 0, stream>>>(
            g_d, cg, H, n_tokens, off, len, sb1, sb2, sb3, C);

        {
            const int64_t work = len * D;
            dim3 grid((work + 255) / 256, H, n_seqs);
            gdn_expand_qk_kernel<<<grid, 256, 0, stream>>>(
                q_d, k_d, q_e, k_e, D, H, off, len, sq1, sq2, sq3, neqk1, rq3, C);
        }

        // --- Gram matrices: KKT = K K^T, QKT = Q K^T  (row-major C x C) -----
        // row-major C = A B^T  <=>  column-major C^T = B A^T
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            len, len, D,
            &one,
            k_e, CUDA_R_32F, D, C * D,
            k_e, CUDA_R_32F, D, C * D,
            &zero,
            KKT, CUDA_R_32F, len, C * C,
            HS, comp, CUBLAS_GEMM_DEFAULT));

        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            len, len, D,
            &one,
            k_e, CUDA_R_32F, D, C * D,
            q_e, CUDA_R_32F, D, C * D,
            &zero,
            QKT, CUDA_R_32F, len, C * C,
            HS, comp, CUBLAS_GEMM_DEFAULT));

        // --- M, P, then W = M^{-1} diag(beta) -------------------------------
        {
            dim3 grid((len * len + 255) / 256, H, n_seqs);
            gdn_build_MP_kernel<<<grid, 256, 0, stream>>>(
                KKT, QKT, cg, b_d, M, P, H, off, len, sb1, sb2, sb3, C);
        }
        {
            const size_t one_mat = len * len * sizeof(float);
            const size_t shmem = (2 * one_mat <= 48 * 1024) ? 2 * one_mat : one_mat;
            if (shmem > 48 * 1024) {
                cudaFuncSetAttribute(gdn_invert_M_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
            }
            gdn_invert_M_kernel<<<dim3(H, n_seqs), 256, shmem, stream>>>(
                M, b_d, H, off, len, sb1, sb2, sb3, C);
        }

        // --- KS = K S,  QS = Q S   (row-major C x D) ------------------------
        // row-major (C x D) = (C x D)(D x D); column-major: C^T = S^T K^T
        // (K S)^T = S^T K^T : state transposed, k_e already holds K^T column-major
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            D, len, D,
            &one,
            state_d, CUDA_R_32F, D, D * D,
            k_e,     CUDA_R_32F, D, C * D,
            &zero,
            KS, CUDA_R_32F, D, C * D,
            HS, comp, CUBLAS_GEMM_DEFAULT));

        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            D, len, D,
            &one,
            state_d, CUDA_R_32F, D, D * D,
            q_e,     CUDA_R_32F, D, C * D,
            &zero,
            QS, CUDA_R_32F, D, C * D,
            HS, comp, CUBLAS_GEMM_DEFAULT));

        // --- rhs = V - Gamma KS, then U = W rhs -----------------------------
        {
            dim3 grid((len * D + 255) / 256, H, n_seqs);
            gdn_make_rhs_kernel<<<grid, 256, 0, stream>>>(
                KS, v_d, cg, D, H, off, len, sv1, sv2, sv3, C);
        }
        // row-major U(C x D) = W(C x C) rhs(C x D); column-major: U^T = rhs^T W^T
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            D, len, len,
            &one,
            KS, CUDA_R_32F, D,   C * D,
            M,  CUDA_R_32F, len, C * C,
            &zero,
            U, CUDA_R_32F, D, C * D,
            HS, comp, CUBLAS_GEMM_DEFAULT));

        // --- PU = P U, then out = scale (Gamma QS + PU) ---------------------
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            D, len, len,
            &one,
            U, CUDA_R_32F, D,   C * D,
            P, CUDA_R_32F, len, C * C,
            &zero,
            PU, CUDA_R_32F, D, C * D,
            HS, comp, CUBLAS_GEMM_DEFAULT));
        {
            dim3 grid((len * D + 255) / 256, H, n_seqs);
            gdn_finish_out_kernel<<<grid, 256, 0, stream>>>(
                QS, PU, cg, dst_d, D, H, n_tokens, off, len, scale, C);
        }

        // --- S = G_C S + (Gamma_C Khat)^T U ---------------------------------
        {
            dim3 grid((len * D + 255) / 256, H, n_seqs);
            gdn_scale_k_kernel<<<grid, 256, 0, stream>>>(k_e, k_s, cg, D, H, len, C);
        }
        {
            dim3 grid((D * D + 255) / 256, H, n_seqs);
            gdn_decay_state_kernel<<<grid, 256, 0, stream>>>(state_d, cg, D, H, len, C);
        }
        // S[i][c] += sum_t k_s[t][i] U[t][c]; column-major: S^T += U^T k_s
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, len,
            &one,
            k_s, CUDA_R_32F, D, C * D,
            U,   CUDA_R_32F, D, C * D,
            &one,
            state_d, CUDA_R_32F, D, D * D,
            HS, comp, CUBLAS_GEMM_DEFAULT));
    }

    CUDA_CHECK(cudaGetLastError());
    return true;
}
