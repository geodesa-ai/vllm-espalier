"""vLLM plugin for Espalier checkpoints (quant_method "espalier").

Registered through the vllm.general_plugins entry point; importing the config module registers
the quantization method with vLLM.

The ESPALIER_* switches change the compiled graph (which custom ops it calls) but are not vLLM config, so vLLM's
torch.compile cache would hand one setting's graph to another (a graph calling espalier::dense with no layers registered).
register() mixes them into VllmConfig.compute_hash, which keys that cache.
"""
import os

SWITCHES = ("ESPALIER_FUSED_DENSE", "ESPALIER_GROUPED_EXPERTS", "ESPALIER_FUSED_ROUTING", "ESPALIER_PREFILL_MODE")


def _salted(compute_hash):
    def compute_hash_esp(self, *args, **kwargs):
        import hashlib
        h = compute_hash(self, *args, **kwargs)
        salt = ",".join(f"{k}={os.environ.get(k, '')}" for k in SWITCHES)
        return hashlib.sha256(f"{h}|{salt}".encode()).hexdigest()[:len(h)]
    compute_hash_esp._esp = True
    return compute_hash_esp


def _patch_mtp():
    """vLLM shares the target's lm_head and embed_tokens with an MTP drafter, but only after the drafter has
    allocated its own embedding (0.95 GB here) and loaded its own head (our 384 MB K6 head). On a 12 GB card that
    transient copy does not fit, so for Espalier checkpoints the drafter frees its embedding right after construction and
    skips loading both; the proposer then installs the target's."""
    _after_import("vllm.model_executor.models.qwen3_5_mtp", _patch_mtp_module)


def _after_import(name, fn):
    """fn(module) once `name` is imported, without importing it now: importing a vLLM model module initialises CUDA,
    which in the API process forces vLLM's workers onto the spawn start method (and holds a CUDA context there)."""
    import sys
    if name in sys.modules:
        fn(sys.modules[name])
        return
    import importlib.abc
    import importlib.util

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def exec_and_patch(module):
                exec_module(module)
                fn(module)
            spec.loader.exec_module = exec_and_patch
            return spec

    sys.meta_path.insert(0, Hook())


def _patch_mtp_module(m):
    import torch
    from .config import NAME

    pred = m.Qwen3_5MultiTokenPredictor
    if getattr(pred.__init__, "_esp", False):
        return
    orig_init, orig_load = pred.__init__, m.Qwen3_5MTP.load_weights

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        qc = kwargs["vllm_config"].quant_config
        if qc is not None and qc.get_name() == NAME and hasattr(self, "embed_tokens"):
            w = self.embed_tokens.weight
            self.embed_tokens.weight = torch.nn.Parameter(w.new_empty((0, w.shape[1])), requires_grad=False)

    def load_weights(self, weights):
        if self.quant_config is not None and self.quant_config.get_name() == NAME:
            weights = ((n, w) for n, w in weights if "lm_head" not in n and "embed_tokens" not in n)
        return orig_load(self, weights)

    init._esp = True
    pred.__init__ = init
    m.Qwen3_5MTP.load_weights = load_weights


def _no_combo_benchmark(post_init):
    """vLLM's inductor settings benchmark combo kernels, and autotune Triton kernels at compile time, on random copies
    of a kernel's inputs. A cold compile of the MTP drafter does this for a kernel that reads the bf16 embedding
    (0.95 GB), and on a 12 GB card that copy runs out of memory (a warm compile cache hides it). For Espalier
    checkpoints combo kernels stay on, unbenchmarked, and Triton kernels autotune at first run on the real tensors."""
    def post_init_esp(self, *args, **kwargs):
        post_init(self, *args, **kwargs)
        qc = getattr(self, "quant_config", None)
        from .config import NAME
        if qc is not None and qc.get_name() == NAME:
            cfg = self.compilation_config.inductor_compile_config
            cfg["benchmark_combo_kernel"] = False
            cfg["triton.autotune_at_compile_time"] = False     # also benchmarks on random input copies
    post_init_esp._esp = True
    return post_init_esp


def register():
    from . import config, ops  # noqa: F401
    _patch_mtp()
    from vllm.config import VllmConfig
    if not getattr(VllmConfig.__post_init__, "_esp", False):
        VllmConfig.__post_init__ = _no_combo_benchmark(VllmConfig.__post_init__)
    if not getattr(VllmConfig.compute_hash, "_esp", False):
        VllmConfig.compute_hash = _salted(VllmConfig.compute_hash)
