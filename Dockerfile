ARG BASE_IMAGE=tensorfold-qwen38:v0.6.1
FROM ${BASE_IMAGE}

ENV HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HF_HUB_OFFLINE=1

COPY patches/0003-cors-and-multimodal-roles.patch /tmp/0003.patch
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    patch -p0 --forward < /tmp/0003.patch && rm /tmp/0003.patch

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["/app/entrypoint.sh"]
