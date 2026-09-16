#pragma once
#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Returns true if chunked prefill can be used; false for recurrent kernel
bool ggml_cuda_should_use_chunked_gdn(const ggml_tensor * dst);

// Shape-only part of the above, with no dependence on the current device. Used to size the chunked
// scratch at allocation time, where the eventual execution device may not be current yet.
bool ggml_cuda_gdn_chunked_shape_eligible(const ggml_tensor * dst);

// As above but ignoring the K == 1 requirement (used by the K>1 split).
bool ggml_cuda_gdn_chunked_shape_eligible_ignoring_k(const ggml_tensor * dst);

// Single-kernel chunked prefill (gdn_fused.cu): the whole chunk pipeline in one fused,
// state-resident kernel. Selected instead of the three-stage chunked path when
// ggml_cuda_gdn_use_fused_chunked() is true; honours the same cache contract as the
// three-stage entry (cache != nullptr redirects the final-state write to cache->data).
// n_bulk > 0 processes only the first n_bulk tokens and writes the carried state to
// bulk_state_out (single sequence only); 0 / nullptr keeps the normal whole-sequence behaviour.
void ggml_cuda_op_gated_delta_net_chunked_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                                const ggml_cuda_gated_delta_net_fused_cache * cache,
                                                int n_bulk = 0, float * bulk_state_out = nullptr);
bool ggml_cuda_gdn_use_fused_chunked(void);
