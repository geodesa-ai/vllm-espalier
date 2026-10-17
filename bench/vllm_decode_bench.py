"""Single-stream decode throughput of an Espalier checkpoint in vLLM (vllm-espalier), greedy.

usage: python vllm_decode_bench.py --ckpt CKPT [--tokens 128] [--graphs]
Prints prefill time, decode tokens/s and the first generated tokens (eyeball coherence).
"""
import argparse, time

from vllm import LLM, SamplingParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--graphs", action="store_true", help="allow CUDA graphs (default: eager)")
    ap.add_argument("--kv-gib", type=float, default=0, help="fixed KV cache size (small GPUs); 0: by utilization")
    ap.add_argument("--mtp", type=int, default=0, help="speculative decoding with the checkpoint's MTP head, N tokens")
    ap.add_argument("--text-only", action="store_true", help="language_model_only: no vision tower (frees ~0.8 GB)")
    ap.add_argument("--kv-dtype", default="", help="kv_cache_dtype, e.g. fp8 (turboquant_4bit_nc breaks speculation)")
    ap.add_argument("--attn", default="", help="attention backend, e.g. TRITON_ATTN")
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()

    graphs = {} if not a.graphs else {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]}}
    llm = LLM(model=a.ckpt, enforce_eager=not a.graphs, max_model_len=a.max_len, max_num_seqs=1,
              gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 0, "video": 0},
              **({"kv_cache_memory_bytes": int(a.kv_gib * 2**30)} if a.kv_gib else {}),
              **({"language_model_only": True} if a.text_only else {}),
              **({"kv_cache_dtype": a.kv_dtype} if a.kv_dtype else {}),
              **({"attention_config": {"backend": a.attn}} if a.attn else {}),
              **({"speculative_config": {"method": "mtp", "num_speculative_tokens": a.mtp}} if a.mtp else {}),
              **graphs)
    prompt = "Explain, step by step, why the sky appears blue during the day and red at sunset."
    msgs = [{"role": "user", "content": prompt}]
    tok = llm.get_tokenizer()
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    warm = SamplingParams(max_tokens=8, temperature=0.0)
    llm.generate([{"prompt_token_ids": ids}], warm, use_tqdm=False)

    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    t1 = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=a.tokens, temperature=0.0,
                                                                   ignore_eos=True), use_tqdm=False)[0]
    t2 = time.perf_counter()
    n = len(out.outputs[0].token_ids)
    prefill = t1 - t0
    print(f"prompt {len(ids)} tok, prefill+1 {prefill * 1000:.0f} ms, "
          f"decode {(n - 1) / (t2 - t1 - prefill):.2f} tok/s over {n} tokens")
    print("TEXT:", out.outputs[0].text[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
