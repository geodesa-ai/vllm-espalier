# vllm-espalier

A vLLM plugin that serves **Espalier** checkpoints: trellis-quantized models whose values are trained onto hardware
number grids. An espalier is a tree trained flat against a trellis.

The format is EXL3's 16-bit-state trellis, with every codebook value snapped onto a hardware grid: E2M1 (FP4) for
1-2-bit experts, int5 / int6 elsewhere. Weights are decoded inside the matmul kernels, so DRAM traffic is the compressed bytes.
No exllamav3 install is needed.

The first checkpoint is **Qwen3.6-35B-A3B at 2.12 bits per weight** (decoder), which runs on a 12 GB card:

| RTX 4070 Ti (12 GB), batch 1, greedy, 256 tokens | decode tok/s |
|---|---:|
| plain decoding | **181** |
| + speculative decoding with the model's own MTP head (3 draft tokens) | **304** |

Mean over three prompts (prose, code, an edit task), from a cold compile; vLLM 0.31.0, CUDA graphs, fp8 KV cache.
Prefill runs at ~1,650 tok/s from 512 to 8k tokens.

## Quality

KL divergence to the bf16 model on held-out text (prose, code and multilingual slices; same windows and probe for
every row):

| checkpoint | bits/weight | pooled KL ↓ |
|---|---:|---:|
| **Espalier Qwen3.6-35B-A3B** | **2.12** (decoder) | **0.0673** |
| published EXL3, 2.49 bpw | 2.49 (as published) | 0.0805 |
| published EXL3, 2.08 bpw (equal bytes) | 2.08 (as published) | 0.1066 |

That's 16% lower KL than the 2.49 bpw EXL3 at about 1.7 GB fewer bytes, and 37% lower at equal bytes.

**What's stored:**
- Routed experts: 1-, 2- or 3-bit trellis codes, the width chosen per expert.
- Attention, DeltaNet and the shared expert: 4-bit trellis codes.
- LM head: 6-bit trellis codes. Embedding, norms and router: bf16.
- Codebooks: E2M1 values for 1-2-bit experts, int5 / int6 grids for 3-bit experts and the always-active layers,
  ExLlamaV3's mcg for the LM head and the MTP head's experts.

How the codes are chosen is our own pipeline and isn't part of this release.

The plugin's kernels change none of this. On a parity eval they move pooled KL by under ±0.0012 and top-1 agreement
by under ±0.3 pt against the reference decode path.

## Container

`ghcr.io/geodesa-ai/vllm-espalier:0.1.0` is vLLM 0.31.0 with the plugin and its kernels prebuilt (sm80 through sm120).
With no arguments it serves `cagyirey/Qwen3.6-35B-A3B-Espalier` on port 8000 with the validated settings
(`docker/espalier-serve`):

```sh
docker run -d --gpus all --ipc host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/geodesa-ai/vllm-espalier:0.1.0
```

Extra arguments go to `vllm serve` and override the defaults (`docker/espalier-serve`); `--no-defaults` passes only
yours. `PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS` (default 2) and `KV_CACHE_MEMORY_BYTES` (default: GPU memory minus
~11.5 GiB, 512 MiB on a 12 GB card) set the common ones.

## Install

```sh
pip install vllm==0.31.0
pip install git+https://github.com/geodesa-ai/vllm-espalier
```

The CUDA extension is compiled on first use (about 3 minutes; cached under `TORCH_EXTENSIONS_DIR`).
`ESPALIER_PREBUILT_EXTENSION` can point at an already built `espalier_ext.so` instead. Requirements:
- An NVIDIA GPU with sm80 or newer. Everything is tested on sm89 (RTX 4070 Ti); the grouped expert kernels were also
  validated on sm90 (H100), sm100 (B200) and sm120 (RTX PRO 6000).
- The CUDA toolkit (nvcc) matching your PyTorch.
- Tensor parallel size 1.

## Use

```python
from vllm import LLM, SamplingParams

if __name__ == "__main__":   # vLLM starts its workers with spawn for this model
    llm = LLM(
        model="cagyirey/Qwen3.6-35B-A3B-Espalier",
        language_model_only=True,                     # no vision tower (frees ~0.8 GB)
        kv_cache_dtype="fp8",
        attention_config={"backend": "TRITON_ATTN"},  # on sm89, fp8 KV otherwise picks FlashInfer, whose workspace is large
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        kv_cache_memory_bytes=int(0.5 * 2**30),       # 12 GB card (~13k tokens); about total VRAM - 11.2 GiB
        max_model_len=-1,                             # the longest context that fits that budget
        max_num_seqs=4,
    )
    print(llm.generate("Explain why the sky is blue.", SamplingParams(max_tokens=256, temperature=0))[0].outputs[0].text)
```

### Switches (environment)

| variable | default | effect |
|---|---|---|
| `ESPALIER_FUSED_DENSE` | 1 | Dense layers: one launch per vLLM layer for up to 32 rows (all its matrices, input rotation fused, bf16 in/out). |
| `ESPALIER_GROUPED_EXPERTS` | 1 | Routed experts plus the shared expert: the grouped pipeline, all tokens of a layer in six launches, up to 32 tokens. |
| `ESPALIER_PREFILL_MODE` | chunk | Above 32 tokens (prefill): `chunk` runs the same fused kernels in 32-token windows, so a token gets the same bits in prefill as in decode; `gemm` decodes each weight and multiplies with cuBLAS fp16 (3-10x slower here). |
| `ESPALIER_FUSED_ROUTING` | 1 | Router GEMV + top-k + grouping in one launch (+1.4% plain decode, +4% with MTP-3). Its expert choices match exact routing; vLLM's bf16 router differs on near-ties in ~2% of rows. `0` uses vLLM's router. |

The plugin salts vLLM's torch.compile cache key with these switches, so
changing one never reuses another setting's compiled graph.

## Known issues

- **Don't combine speculative decoding with `kv_cache_dtype="turboquant_4bit_nc"`** (vLLM 0.31). The TurboQuant
  attention backend is fine for one-token decode, but under speculative verification it produced 98 of 601 tokens that
  the model would not have chosen. Use fp8 or bf16 KV.
- **n-gram speculation (`ngram`, `ngram_gpu`) is wrong on this hybrid DeltaNet model in vLLM 0.31**, independently
  of this plugin (same result with every Espalier kernel off), and slower than no speculation. Use `mtp`.
- **Bigger drafters don't fit a 12 GB card at bf16:** EAGLE-3 (specdrift), DFlash (z-lab), DFlash2 (incoai) and dspark
  (RedHatAI). They run out of memory at load.
- **Prefill is ~1,650 tok/s** on an RTX 4070 Ti (512 to 8k tokens): it reuses the decode kernels in 32-token windows,
  which re-read every weight once a window. A dedicated prefill GEMM is planned.
- **Accuracy notes:**
  - Prefill and decode run the same fused kernels, so a token's logits are the same either way.
  - All kernels are row-invariant: a token's logits don't depend on what else is in the batch.
- **The plugin changes two torch inductor settings for Espalier checkpoints:** combo-kernel benchmarking and
  compile-time Triton autotuning are off. Both run on random copies of a kernel's inputs, and a cold compile of the MTP
  drafter copies the bf16 embedding (0.95 GB), which runs out of memory on 12 GB cards.

## Verifying speculative output

`bench/spec_bench.py` times decode for any `speculative_config`, then scores every generated token with the plain
model. A configuration is only correct if the tokens that are not the plain model's argmax are near-ties:

```sh
python bench/spec_bench.py --ckpt <checkpoint> --tag mtp3 --spec '{"method": "mtp", "num_speculative_tokens": 3}'
```

## Tests

GPU tests check accuracy against a float32 reference (error above the bf16/fp16 rounding floor < 5e-4), bit-exact row
invariance, CUDA-graph == eager, and merged == unmerged launches:

```sh
python tests/gpu_dense.py   --ckpt <checkpoint>     # dense layers
python tests/gpu_grouped.py --ckpt <checkpoint>     # routed experts
python tests/gpu_grouped_shared.py --ckpt <checkpoint>
python tests/gpu_route.py   --ckpt <checkpoint>     # fused routing vs vLLM's router and the exact routing
python tests/gpu_native.py  --ckpt <checkpoint>     # single matrices above 32 rows
python tests/gpu_prefill.py --ckpt <checkpoint>     # prefill windows: rows bit-identical to decode
```

## Credits

- The grouped expert kernels and the dense linear kernel are derived from
  [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), itself after
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT).
- The trellis format is ExLlamaV3's. See `NOTICE`.

License: MIT or Apache-2.0, at your option, except the TensorFold-derived kernels (`csrc/experts.cpp`, `experts.cu`,
`experts_grouped.cuh`, `dense.cu`), which are Apache-2.0. See `NOTICE`.
