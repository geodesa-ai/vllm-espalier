"""Speculative decoding schemes on an Espalier checkpoint: speed and correctness from the same generations.

Stage gen (with --spec): for each prompt, greedy-generate --tokens tokens, timing decode as vllm_decode_bench does
(prefill + 1 token timed separately, with the prefix cache in the same state, and subtracted). Stage score (plain model, chunked prefill): every generated token up
to the answer's end of turn is scored with the plain model's prompt log-probs; a token that is not the plain model's argmax must be a near-tie
(small gap), or the speculative path computes something else (see the README's note on TurboQuant).

Prompts: prose, code, and an edit task that repeats much of its prompt (where n-gram / suffix drafting can help).

usage: python spec_bench.py --ckpt CKPT --spec '{"method": "ngram", "num_speculative_tokens": 3,
         "prompt_lookup_max": 4, "prompt_lookup_min": 2}' --tag ngram3 [--kv-dtype fp8 --attn TRITON_ATTN --kv-gib 0.5]
"""
import argparse
import json
import os
import subprocess
import sys
import time

CODE = '''def merge(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
'''
PROMPTS = {
    "prose": "Explain, step by step, why the sky appears blue during the day and red at sunset.",
    "code": "Write a Python function that merges two sorted lists, then explain its complexity.",
    "edit": "Rewrite this function with type hints and a docstring, keeping the body unchanged:\n\n" + CODE,
}


def build(a, spec):
    from vllm import LLM
    kw = dict(model=a.ckpt, max_model_len=a.max_len, max_num_seqs=1, kv_cache_memory_bytes=int(a.kv_gib * 2**30),
              language_model_only=True, limit_mm_per_prompt={"image": 0, "video": 0}, max_logprobs=5,
              compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]}, disable_log_stats=False)
    if a.kv_dtype:
        kw["kv_cache_dtype"] = a.kv_dtype
    if a.no_prefix_cache:
        kw["enable_prefix_caching"] = False
    if a.attn:
        kw["attention_config"] = {"backend": a.attn}
    if spec:
        kw["speculative_config"] = spec
    else:
        kw.update(enable_chunked_prefill=True, max_num_batched_tokens=64)
    return LLM(**kw)


def ids_of(tok, p):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False), add_special_tokens=False)


def spec_metrics(llm):
    """Accepted / drafted tokens from vLLM's metrics, when it exposes them."""
    try:
        out = {}
        for m in llm.get_metrics():
            if "spec_decode" in m.name and hasattr(m, "value"):
                out[m.name.split(":")[-1]] = m.value
        return out
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", default="", help="speculative_config as JSON ('' = no speculation)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--attn", default="TRITON_ATTN")
    ap.add_argument("--kv-gib", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default="results/specdec")
    ap.add_argument("--stage", default="")
    ap.add_argument("--no-prefix-cache", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    io = os.path.join(a.out, f"{a.tag}.json")
    if not a.stage:
        base = [sys.executable, __file__] + [x for x in sys.argv[1:]]
        subprocess.run(base + ["--stage", "gen"], check=True)
        # score on the known-good plain path whatever the generation used (ESPALIER_FUSED_DENSE=0 would reconstruct the head)
        env = dict(os.environ, ESPALIER_FUSED_DENSE="1", ESPALIER_GROUPED_EXPERTS="1", ESPALIER_FUSED_ROUTING="0")
        subprocess.run(base + ["--stage", "score"], check=True, env=env)
        r = json.load(open(io))
        print(f"\n{a.tag}: " + "  ".join(f"{k} {v['tok_s']:.1f} tok/s" for k, v in r["prompts"].items())
              + f"  | mean {r['mean_tok_s']:.1f} tok/s | not plain argmax {r['score']['off']}/{r['score']['n']}, "
              f"largest gap {r['score']['max_gap']:.3f}")
        return
    from vllm import SamplingParams
    if a.stage == "gen":
        spec = json.loads(a.spec) if a.spec else None
        llm = build(a, spec)
        tok = llm.get_tokenizer()
        res = {"spec": spec, "prompts": {}}
        warm = ids_of(tok, "Hello")
        llm.generate([{"prompt_token_ids": warm}], SamplingParams(max_tokens=16, temperature=0.0), use_tqdm=False)
        for name, p in PROMPTS.items():
            ids = ids_of(tok, p)
            # prefill + 1 token, timed in the state the generation runs in: the first call fills the prefix cache,
            # the second (timed) hits it as the generation will (subtracting an uncached prefill overstates tok/s by
            # the prompt's prefill time)
            llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
            t0 = time.perf_counter()
            llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
            t1 = time.perf_counter()
            g = llm.generate([{"prompt_token_ids": ids}],
                             SamplingParams(max_tokens=a.tokens, temperature=0.0, ignore_eos=True), use_tqdm=False)
            t2 = time.perf_counter()
            gen = list(g[0].outputs[0].token_ids)
            dec = (t2 - t1) - (t1 - t0)
            res["prompts"][name] = {"prompt": ids, "gen": gen, "tok_s": (len(gen) - 1) / dec}
        res["mean_tok_s"] = sum(v["tok_s"] for v in res["prompts"].values()) / len(PROMPTS)
        res["metrics"] = spec_metrics(llm)
        json.dump(res, open(io, "w"))
        return
    res = json.load(open(io))
    llm = build(a, None)
    tok = llm.get_tokenizer()
    # tokens after the answer's end of turn (generated only because of ignore_eos) are not scored: there the model
    # wants to stop, its distribution over template fragments is flat, and every scoring path disagrees with decode
    stops = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} | {tok.eos_token_id}
    gaps, n = [], 0
    for name, d in res["prompts"].items():
        ids = d["prompt"] + d["gen"]
        end = len(ids)
        for j in range(len(d["prompt"]), len(ids)):
            if ids[j] in stops:
                end = j + 1
                break
        d["scored"] = end - len(d["prompt"])
        r = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=5),
                         use_tqdm=False)[0]
        pl = r.prompt_logprobs
        for j in range(len(d["prompt"]), end):
            n += 1
            cand = pl[j]
            best = max(cand.values(), key=lambda v: v.logprob)
            chosen = cand.get(ids[j])
            if chosen is None or chosen.rank != 1:
                gaps.append(best.logprob - (chosen.logprob if chosen else -99.0))
    gaps.sort(reverse=True)
    res["score"] = {"n": n, "off": len(gaps), "max_gap": gaps[0] if gaps else 0.0, "gaps": gaps[:20]}
    json.dump(res, open(io, "w"))


if __name__ == "__main__":
    main()
