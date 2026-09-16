#pragma once

#include "common.cuh"

// Chunked Gated DeltaNet prefill path (scalar-gate / non-KDA only).
// Returns false if this call is not eligible, in which case the caller must
// fall back to the sequential kernel.
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
        float scale, int64_t state_slot_stride, int K);
