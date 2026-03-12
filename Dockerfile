ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install tt-metal debs (runtime + dev headers + JIT compiler)
# The base image has tt-metal as a Python package but not as system debs,
# so we install all four to get the C++ headers and system libs needed to build.
ARG TT_METAL_VERSION=0.66.0
RUN TMP=$(mktemp -d) && \
    BASE="https://github.com/tenstorrent/tt-metal/releases/download/v${TT_METAL_VERSION}" && \
    wget -q -P "$TMP" \
        "$BASE/tt-metalium_${TT_METAL_VERSION}.ubuntu22.04_amd64.deb" \
        "$BASE/tt-metalium-dev_${TT_METAL_VERSION}.ubuntu22.04_amd64.deb" \
        "$BASE/tt-metalium-jit_${TT_METAL_VERSION}.ubuntu22.04_amd64.deb" \
        "$BASE/tt-nn_${TT_METAL_VERSION}.ubuntu22.04_amd64.deb" && \
    apt-get update && \
    apt-get install -y "$TMP"/*.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* "$TMP"

WORKDIR /app

# Copy project source
COPY src/ src/
COPY tools/ tools/
COPY third_party/json.hpp third_party/json.hpp
COPY third_party/httplib.h third_party/httplib.h
COPY third_party/blockfloat_common.hpp third_party/blockfloat_common.hpp
COPY Makefile .

# Build the project (target x86-64-v4 for AVX-512 since -march=native
# won't work in cross-build environments like Docker BuildKit)
RUN make -j$(nproc) MARCH=x86-64-v4

# Install SSH server (after build to preserve cache on source changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-server && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    ssh-keygen -A && \
    sed -i 's/#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Embed build version (git SHA) — must be last ARG to avoid cache busting earlier layers
ARG GIT_SHA=unknown
ENV BUILD_VERSION=${GIT_SHA}

# Model is downloaded at runtime via the binary's built-in HuggingFace resolver.
# To bake the model into the image for faster cold starts, uncomment:
# RUN mkdir -p /models && \
#     curl -L -o /models/Qwen3.5-9B-BF16.gguf \
#       "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-BF16.gguf"
# ENV MODEL_PATH=/models/Qwen3.5-9B-BF16.gguf

ENV MODEL_PATH=vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled
ENV TT_METAL_RUNTIME_ROOT=/usr/libexec/tt-metalium
ENV QUIET=1
ENV SERVER_PORT=8888

EXPOSE 8888 22

CMD ["/entrypoint.sh"]
