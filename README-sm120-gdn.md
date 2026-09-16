# Gated Delta Net on RTX 5070 Ti (sm_120, Blackwell)

Measurements for the GDN prefill kernels on consumer Blackwell. Every number in
the upstream discussion so far is RTX 3090, RTX 6000 Pro MaxQ, or MI210, so this
fills in a gap rather than repeating existing data.

Relevant upstream work:
- PR #26001 - CUDA: Support of GDN chunked kernel for prefill (BLSharda)
- branch `gdn-fused-v2` - fused single-kernel variant (lukdmine), posted to that PR

## Setup

| | |
|---|---|
| GPU | RTX 5070 Ti, 16 GB, sm_120, **70 SM**, driver 610.57.04 |
| Toolchain | CUDA 13.3, g++-15 host compiler, `-DCMAKE_CUDA_ARCHITECTURES=120` |
| Model | Qwen3.5-27B UD-IQ3_S (`qwen35`, 65 blocks, `full_attention_interval=4`) |
| Server | `-c 114688 -fa on -ctk q4_0 -ctv q4_0 -ot token_embd.weight=CPU --parallel 1 --fit off` |
| Metric | `prompt_per_second` from llama-server, cold ~31.6k-token prompt |

Each request uses a unique salt so the prompt cache cannot hit; every prefill is
measured cold. Builds differ **only** by `-DGGML_CUDA_NO_GDN_CHUNK` /
`-DGGML_CUDA_NO_GDN_FUSED`.

## Build

Both branches compile for sm_120 with no source changes. `gdn_fused.cu` is gated
on `__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE` (800), and sm_120 is 1200.

The fused path is selected by `ggml_cuda_gdn_use_fused_chunked()`, which returns
`nsm <= 84`. This card has 70 SM, so the fused path is taken automatically.

## Correctness

| check | result |
|---|---|
| `test-backend-ops test -o GATED_DELTA_NET` | **51/51 pass** |
| `verify.py` on the real model (arith, 90k needle, mid-conversation system msg, tool call) | **4/4 pass** |

## Prefill, K == 1

| GDN path | prefill tok/s | vs recurrent |
|---|---|---|
| recurrent (baseline) | 1508.9 / 1507.3, repeat 1499.6 / 1498.9 | - |
| chunked (#26001) | 1622.7 / 1619.7 | **+7.8%** |
| fused (`gdn-fused-v2`) | 1666.6 / 1665.8 | **+10.8%** |

### How much headroom is there in total

Ablation: the recurrent kernel's token loop was forced to 1 so the op becomes
almost free (output is garbage, only timings are read; buffer offsets left on the
real `n_tokens`).

| | prefill tok/s |
|---|---|
| normal | 1435.3 / 1435.9, repeat 1431.8 / 1430.8 |
| GDN ablated to ~zero cost | 1607.9 / 1606.0 |

So GDN is `1 - 1433.5/1607` = **10.8% of prefill time**, and a perfect kernel
would be worth about +12%. The fused kernel captures roughly 90% of that.

Note that layer count is misleading here: ~49 of 65 layers are GDN, but they are
only ~11% of prefill time, being cheap per token next to the dense FFN
(`feed_forward_length = 17408`) and the 16 full-attention layers.

## Prefill, K > 1: no gain

Both the chunked and fused paths require `K == 1`. With speculative decoding
(`--spec-draft-n-max 4`, so K = 5):

| GDN path | prefill tok/s |
|---|---|
| recurrent | 1438.0 / 1437.8 |
| fused | 1433.5 / 1432.4 |

No gain, within noise. Any `--spec-draft-n-max >= 1` gives `K >= 2`, so there is
no runtime setting that reaches the chunked path while speculative decoding is
on. Closing that would need a split: chunked over `[0, n-K)`, then the recurrent
kernel over the final K tokens seeded with the carried state, since only it
writes the snapshot slots (`gated_delta_net.cu`, snapshot block).

## Provenance

`gdn-fused-v2-sm120` on this fork is lukdmine's code, unmodified; only the build
and the measurements above are mine. `gdn-chunked-cublas` is a separate,
independent experiment of my own that did **not** work out (a cuBLAS-orchestrated
chunked path, ~1.8% slower than recurrent); it is kept for its negative results,
which are recorded in that branch's commit messages.

These measurements were produced with AI assistance (Claude).
