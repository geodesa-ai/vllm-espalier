# ghcr.io/geodesa-ai/vllm-espalier: vLLM 0.31.0 with the vllm-espalier plugin and its CUDA kernels prebuilt.
# Built and pushed by .github/workflows/image.yml. Local build: docker build -f docker/Dockerfile -t vllm-espalier .
FROM vllm/vllm-openai:v0.31.0

LABEL org.opencontainers.image.source="https://github.com/geodesa-ai/vllm-espalier" \
      org.opencontainers.image.description="vLLM 0.31.0 + vllm-espalier: serves Espalier trellis checkpoints" \
      org.opencontainers.image.licenses="(MIT OR Apache-2.0) AND Apache-2.0"

# sm80 (A100), sm86 (RTX 30), sm89 (RTX 40, L4, L40), sm90 (H100), sm100 (B200), sm120 (RTX 50, RTX PRO 6000)
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"
ARG MAX_JOBS=4

# Kernels first, from csrc alone, so Python-only changes reuse this layer. The image's CUDA toolkit lacks some
# library headers that torch's CUDA headers include; take only those from the pip CUDA packages (their other headers
# are a newer CUDA than the image's nvcc).
COPY vllm_espalier/csrc /opt/espalier/csrc
RUN mkdir -p /opt/espalier/cuda-include /opt/espalier/ext \
 && for f in cusparse cusparse_v2 cusolverDn cusolverSp cusolverRf cusolverMg cusolver_common cufft cufftXt cudalibxt nvJitLink; do \
      ln -s /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include/$f.h /opt/espalier/cuda-include/$f.h; done \
 && cd /opt/espalier/csrc \
 && CPATH=/opt/espalier/cuda-include TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" python3 -c \
    "from torch.utils.cpp_extension import load; \
     load(name='espalier_ext', sources=['experts.cpp', 'experts.cu', 'dense.cu', 'route.cu'], \
          extra_cuda_cflags=['-O3', '-lineinfo'], build_directory='/opt/espalier/ext')" \
 && test -f /opt/espalier/ext/espalier_ext.so \
 && rm -f /opt/espalier/ext/*.o

ENV ESPALIER_PREBUILT_EXTENSION=/opt/espalier/ext/espalier_ext.so \
    ESPALIER_MODEL=cagyirey/Qwen3.6-35B-A3B-Espalier

# install, and check that the plugin imports and loads the prebuilt kernels (no GPU needed)
COPY pyproject.toml README.md LICENSE-MIT LICENSE-APACHE NOTICE /opt/vllm-espalier/
COPY vllm_espalier /opt/vllm-espalier/vllm_espalier
RUN python3 -m pip install --no-cache-dir --no-deps --root-user-action=ignore /opt/vllm-espalier \
 && python3 -c "import torch, vllm_espalier; from vllm_espalier.experts import ext; m = ext(); assert hasattr(m, 'dense'), m"

COPY docker/espalier-serve /usr/local/bin/espalier-serve
ENTRYPOINT ["/usr/local/bin/espalier-serve"]
