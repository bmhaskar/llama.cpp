# Chunked Gated Delta Net prefill for speculative decoding (K > 1), on sm_120

Two things live here:

1. **A fix**: chunked GDN prefill now works with speculative decoding. Upstream
   the chunked paths require `K == 1`, so anyone running spec decode falls back
   to the recurrent kernel and gets no prefill speedup at all. Measured
   **+10.1% prefill at K=5** on the config this was built for.
2. **Data**: sm_120 (consumer Blackwell) numbers for the existing chunked and
   fused kernels. Every measurement in the upstream discussion is RTX 3090,
   RTX 6000 Pro MaxQ, or MI210.

Related upstream work, neither of which handles `K > 1`:
- PR #26001 - CUDA: Support of GDN chunked kernel for prefill (BLSharda)
- branch `gdn-fused-v2` (lukdmine), posted to that PR

---

## 1. The K > 1 split

### Why it is blocked upstream

Speculative decoding needs a state snapshot **per token** for the last `K-1`
tokens, so a rejected draft can be rolled back. The chunked reformulation only
produces state at chunk boundaries - the intermediate per-token states never
exist - so the chunked kernels cannot emit the snapshots and correctly refuse to
run. In `gated_delta_net.cu` that is a single condition: `!kda && K == 1 && ...`.

It is not a hardware limit. Nothing about the GPU prevents it.

### The fix

Split the sequence:

- fused chunked kernel over tokens `[0, n-K)`, exporting the carried state
- recurrent kernel over the final `K` tokens, seeded with that state - it is the
  only path that writes the snapshot slots, and it already does

The tail is `K` tokens (5 in a typical MTP setup) out of a 512-token ubatch, so
the sequential remainder is negligible.

### The part that was easy to get wrong

Both kernels conflated *"tokens this launch processes"* with *"tokens per
sequence in the tensors"*. Identical for a whole-sequence run; different the
moment you run a prefix. Sequence offsets were computed from the processed
count, which silently corrupts every sequence after the first.

- `gdn_fused.cu`: added `kTstride` for per-sequence base offsets; `kT` still
  bounds the loops.
- `gated_delta_net.cu`: the recurrent kernel offset `dst` by
  `sequence * n_tokens`; added `dst_seq_tokens` so the tail launch can advance
  `dst` by `n_bulk` while still striding by the full sequence length.

### Results

RTX 5070 Ti, Qwen3.5-27B UD-IQ3_S, `c=114688`, cold ~31.6k prompt,
`--spec-type draft-mtp,ngram-map-k --spec-draft-n-max 4` (so `K = 5`):

| | |
|---|---|
| prefill | 1433.1 -> **1578.2 / 1576.0 (+10.1%)** |
| `test-backend-ops -o GATED_DELTA_NET` | **60/60** |
| `verify.py` (arith, 90k needle, tool call) | 4/4 PASS |

Test coverage added: `K = 2..5`, partial tails, `n_tokens = 133` (just above the
chunked minimum), and `n_seqs = 1..4`.

---

## 2. sm_120 data for the existing kernels

| | |
|---|---|
| GPU | RTX 5070 Ti, 16 GB, sm_120, **70 SM**, driver 610.57.04 |
| Toolchain | CUDA 13.3, g++-15, `-DCMAKE_CUDA_ARCHITECTURES=120` |
| Metric | `prompt_per_second`, unique salt per request so the prompt cache never hits |

Both branches build for sm_120 with no source changes. `gdn_fused.cu` is gated on
`__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE`; sm_120 is 1200. The fused path is chosen
by `ggml_cuda_gdn_use_fused_chunked()` = `nsm <= 84`, a bound sized for the 3090
(82 SM); this card has 70, so it is taken automatically.

### Prefill at K == 1

| GDN path | prefill tok/s | vs recurrent |
|---|---|---|
| recurrent | 1508.9 / 1507.3, repeat 1499.6 / 1498.9 | - |
| chunked (#26001) | 1622.7 / 1619.7 | +7.8% |
| fused (`gdn-fused-v2`) | 1666.6 / 1665.8 | **+10.8%** |

Correctness for the fused kernel on sm_120: **51/51** before the split work.

### How much headroom exists at all

Ablation - the recurrent kernel's token loop forced to 1 so the op is nearly
free (output is garbage, only timings read, buffer offsets left on the real
`n_tokens`):

| | prefill tok/s |
|---|---|
| normal | 1435.3 / 1435.9, repeat 1431.8 / 1430.8 |
| GDN ablated | 1607.9 / 1606.0 |

GDN is **10.8% of prefill time**, so a perfect kernel is worth about +12%. The
fused kernel captures roughly 90% of that.

Layer count misleads here: ~49 of 65 layers are GDN, but they are only ~11% of
prefill *time*, being cheap per token next to the dense FFN
(`feed_forward_length = 17408`) and the 16 full-attention layers.

---

## 3. Side finding: `ngram-map-k` is stateful and decays

Not GDN, but it distorted several benchmarks here before it was understood. The
n-gram map persists across requests. Same server, same prompt, only the request
order differs:

| reproduce-a-file | tok/s |
|---|---|
| first request on a fresh server | **642.9** |
| after one unrelated generation | **159.4** |
| again | 159.0 |

A cold map is seeded perfectly by the copy prompt; once unrelated content has
been through, it does not recover within the session. Any single-run n-gram
benchmark is measuring map state as much as the kernel. It is also why
`draft-mtp` alone is reproducible at temperature 0 while `draft-mtp,ngram-map-k`
is not.

---

## Provenance

- The K > 1 split, the multi-sequence stride fix, and all measurements here are
  from this fork.
- `gdn-fused-v2-sm120` is lukdmine's kernel, unmodified.
- `gdn-chunked-cublas` is a separate experiment of mine that did **not** work
  out (cuBLAS-orchestrated chunking, ~1.8% slower than recurrent). Kept for the
  negative results in its commit messages.
- Produced with AI assistance (Claude). Not submitted upstream; if any of it is,
  the prose needs rewriting by hand per the llama.cpp AI policy.
