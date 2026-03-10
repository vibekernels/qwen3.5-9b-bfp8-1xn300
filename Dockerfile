ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies (base image already has clang-20)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Discover the pre-installed tt-metalium SDK and create a symlink tree
# matching the layout our Makefile expects (TT_METAL_BUILD/{include,lib,libexec}).
RUN set -e && TT_SDK=/app/tt-sdk && \
    mkdir -p $TT_SDK/build_Release/lib $TT_SDK/build_Release/libexec \
             $TT_SDK/tt_metal/impl/data_format && \
    echo "=== Discovering tt-metalium SDK ===" && \
    echo "--- libtt_metal.so ---" && find / -name "libtt_metal.so" 2>/dev/null || true && \
    echo "--- tt-metalium dirs ---" && find / -path "*/tt-metalium" -type d 2>/dev/null | head -10 || true && \
    echo "--- tt_metal includes ---" && find / -path "*/include/tt-metalium" -type d 2>/dev/null | head -5 || true && \
    echo "--- spdlog ---" && find / -name "libspdlog*" 2>/dev/null | head -5 || true && \
    echo "--- tracy ---" && find / -name "libtracy*" 2>/dev/null | head -5 || true && \
    # Symlink include dir \
    INC_PARENT=$(dirname $(find / -path "*/include/tt-metalium" -type d 2>/dev/null | head -1) 2>/dev/null) && \
    if [ -n "$INC_PARENT" ] && [ -d "$INC_PARENT" ]; then \
        ln -sf "$INC_PARENT" $TT_SDK/build_Release/include; \
    fi && \
    # Symlink libexec/tt-metalium \
    LIBEXEC=$(find / -path "*/libexec/tt-metalium" -type d 2>/dev/null | head -1) && \
    if [ -n "$LIBEXEC" ] && [ -d "$LIBEXEC" ]; then \
        ln -sf "$LIBEXEC" $TT_SDK/build_Release/libexec/tt-metalium; \
    fi && \
    # Symlink each required library (search everywhere) \
    for lib in libtt_metal.so libdevice.so libtt_stl.so; do \
        src=$(find / -name "$lib" \( -type f -o -type l \) 2>/dev/null | head -1); \
        if [ -n "$src" ]; then ln -sf "$src" $TT_SDK/build_Release/lib/$lib; fi; \
    done && \
    # Tracy (version-specific soname) \
    tracy=$(find / -name "libtracy.so*" \( -type f -o -type l \) 2>/dev/null | head -1); \
    if [ -n "$tracy" ]; then \
        ln -sf "$tracy" $TT_SDK/build_Release/lib/libtracy.so; \
        ln -sf "$tracy" $TT_SDK/build_Release/lib/$(basename "$tracy"); \
    fi && \
    # spdlog (static or shared) \
    spdlog=$(find / -name "libspdlog.a" -o -name "libspdlog.so" 2>/dev/null | head -1); \
    if [ -n "$spdlog" ]; then \
        ln -sf "$spdlog" $TT_SDK/build_Release/lib/$(basename "$spdlog"); \
    fi && \
    echo "=== SDK shim result ===" && ls -la $TT_SDK/build_Release/lib/ && \
    ls -la $TT_SDK/build_Release/include/ 2>/dev/null || true

# Copy project source
COPY src/ src/
COPY third_party/json.hpp third_party/json.hpp
COPY Makefile .

# Build using the pre-installed SDK.
# Touch the libtt_metal.so target first so Make skips the auto-setup rule
# (which tries git submodule update).
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
