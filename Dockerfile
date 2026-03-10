ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies (base image already has clang-20)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Discover the pre-installed tt-metalium SDK by searching for exact files,
# then create a symlink tree matching our Makefile's expected layout.
RUN set -e && TT_SDK=/app/tt-sdk && \
    mkdir -p $TT_SDK/build_Release/lib $TT_SDK/build_Release/include \
             $TT_SDK/build_Release/libexec $TT_SDK/tt_metal/impl/data_format && \
    # --- Headers --- \
    # Find tt-metalium/host_api.hpp and derive the include root \
    HOST_API=$(find / -path "*/tt-metalium/host_api.hpp" -type f 2>/dev/null | head -1) && \
    echo "host_api.hpp: $HOST_API" && \
    if [ -n "$HOST_API" ]; then \
        TT_METALIUM_DIR=$(dirname "$HOST_API") && \
        INCLUDE_ROOT=$(dirname "$TT_METALIUM_DIR") && \
        ln -sf "$TT_METALIUM_DIR" $TT_SDK/build_Release/include/tt-metalium && \
        ln -sf "$TT_METALIUM_DIR" $TT_SDK/build_Release/libexec/tt-metalium && \
        # metalium-thirdparty (sibling to tt-metalium or up a level) \
        for tp in "$INCLUDE_ROOT/metalium-thirdparty" \
                  "$(dirname $INCLUDE_ROOT)/metalium-thirdparty"; do \
            if [ -d "$tp" ]; then \
                ln -sf "$tp" $TT_SDK/build_Release/include/metalium-thirdparty; \
                break; \
            fi; \
        done; \
    fi && \
    # If metalium-thirdparty not found yet, search for it \
    if [ ! -e $TT_SDK/build_Release/include/metalium-thirdparty ]; then \
        tp=$(find / -path "*/metalium-thirdparty/fmt" -type d 2>/dev/null | head -1) && \
        if [ -n "$tp" ]; then \
            ln -sf "$(dirname $tp)" $TT_SDK/build_Release/include/metalium-thirdparty; \
        fi; \
    fi && \
    # --- Libraries --- \
    for lib in libtt_metal.so libdevice.so libtt_stl.so; do \
        found=$(find / -name "$lib" \( -type f -o -type l \) 2>/dev/null | head -1) && \
        if [ -n "$found" ]; then ln -sf "$found" $TT_SDK/build_Release/lib/$lib; fi; \
    done && \
    # Tracy (versioned) \
    tracy=$(find / -name "libtracy*.so*" \( -type f -o -type l \) 2>/dev/null | head -1) && \
    if [ -n "$tracy" ]; then \
        ln -sf "$tracy" $TT_SDK/build_Release/lib/libtracy.so.0.10.0; \
    fi && \
    # spdlog \
    spdlog=$(find / \( -name "libspdlog.a" -o -name "libspdlog.so" -o -name "libspdlog*.so*" \) 2>/dev/null | head -1) && \
    if [ -n "$spdlog" ]; then \
        ln -sf "$spdlog" $TT_SDK/build_Release/lib/libspdlog.a; \
    fi && \
    # Debug output \
    echo "=== libs ===" && ls -la $TT_SDK/build_Release/lib/ && \
    echo "=== include ===" && ls -la $TT_SDK/build_Release/include/ && \
    echo "=== libexec ===" && ls -la $TT_SDK/build_Release/libexec/

# Copy project source
COPY src/ src/
COPY third_party/json.hpp third_party/json.hpp
COPY Makefile .

# Build using the pre-installed SDK.
# Touch libtt_metal.so so Make skips the auto-setup rule.
RUN touch /app/tt-sdk/build_Release/lib/libtt_metal.so && \
    make -j$(nproc) \
    TT_METAL_HOME=/app/tt-sdk \
    TT_METAL_BUILD=/app/tt-sdk/build_Release

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model is downloaded at runtime via the binary's built-in HuggingFace resolver.
# To bake the model into the image for faster cold starts, uncomment:
# RUN mkdir -p /models && \
#     curl -L -o /models/Qwen3.5-9B-BF16.gguf \
#       "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-BF16.gguf"
# ENV MODEL_PATH=/models/Qwen3.5-9B-BF16.gguf

ENV MODEL_PATH=unsloth/Qwen3.5-9B-GGUF:BF16
ENV TT_METAL_RUNTIME_ROOT=/usr/libexec/tt-metalium
ENV QUIET=1
ENV PORT=8888

EXPOSE 8888

CMD ["/entrypoint.sh"]
