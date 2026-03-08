# qwen3.5-9b-bf16-1xn300d

Custom inference engine for Qwen3.5-9B on a single Tenstorrent N300 card (two Wormhole chips), built from scratch with tt-metal/ttnn APIs.

## Architecture

Qwen3.5-9B is a hybrid architecture with 32 layers: 8 full attention layers (every 4th) and 24 SSM delta-net recurrent layers.

- **Matrix multiplications** run on-device via `ttnn::matmul` on Tensix cores
- **Chip 0** holds all layer weights + output projections as BFLOAT8_B (~6.75 GB)
- **Chip 1** holds the LM head as BFLOAT8_B (~1 GB)
- **On-device RMSNorm** via `ttnn::rms_norm` (eliminates PCIe round-trip for norm)
- **On-device FFN chain**: output projection + residual add + norm + gate/up/down matmuls + residual add, all as a single traced operation
- **SSM recurrence** and **attention** (softmax + KV cache) run on host CPU in f32
- **Metal Traces** capture and replay device op sequences with zero dispatch overhead

## Performance

| Metric | Value |
|--------|-------|
| Decode latency | ~115-130 ms/tok |
| Decode throughput | ~8-9 tok/s |
| Model size (device) | ~7.75 GB BFLOAT8_B |
| Theoretical floor | ~88 ms/tok (DRAM bandwidth limited) |

Performance measured on Tenstorrent N300 (2x Wormhole chips, 1000 MHz AI clock, 12 Gbps DRAM). The first token is slower (~160ms) while traces are captured; subsequent tokens stabilize at ~115-130ms.

### Decode time breakdown (steady state, token 90+)

| Component | Time | Notes |
|-----------|------|-------|
| norm + matmul traces | 84 ms | Includes waiting for previous FFN chain; DRAM bandwidth limited |
| Host SSM/attention | 15-18 ms | Delta-net recurrence (12ms) + conv1d (1.3ms) + attention (2-3ms) |
| LM head (chip 1) | 15 ms | 248K x 4096 BFP8 matmul, bandwidth limited |
| Residual writes | 13 ms | 32 PCIe writes per token (tilize + enqueue) |
| FFN chain dispatch | ~0 ms | Non-blocking, overlaps with next layer's norm wait |

### Why 30 tok/s is not achievable

The fundamental bottleneck is DRAM bandwidth:
- **6.75 GB** of BFP8 weights must be read per token
- At ~115 GB/s effective DRAM bandwidth: **59 ms minimum** just for weight reads
- Plus PCIe overhead (13ms), LM head (15ms), host compute (15ms)
- **Theoretical floor: ~88 ms/tok = 11.4 tok/s**

For reference, Qwen 2.5 7B (smaller, pure attention) achieves ~22 tok/s on N300.

## Building

Requires tt-metal built from source (with `_ttnncpp.so`) and clang-20.

```sh
cd tt_metal
mkdir -p build && cd build
cmake .. -DTT_METAL_BUILD=/path/to/tt-metal/build_Release -DCMAKE_CXX_COMPILER=clang++-20
make -j$(nproc)
```

This produces test binaries: `test_device`, `test_matmul`, `test_load_weights`, `test_forward`.

## Running inference

Requires `sudo` (for device access) and `TT_METAL_RUNTIME_ROOT` pointing to your tt-metal source tree:

```sh
sudo TT_METAL_RUNTIME_ROOT=/path/to/tt-metal \
  ./build/test_forward /path/to/Qwen3.5-9B-BF16.gguf "What is the capital of France?" 128
```

Pass `--raw` as a 4th argument to skip the chat template and send the prompt directly.

## Tests

All tests require `sudo` and `TT_METAL_RUNTIME_ROOT`:

```sh
sudo TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_device        # validate N300 device opens
sudo TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_matmul        # basic ttnn::matmul
sudo TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_load_weights  # GGUF weight loading
sudo TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_forward       # full generation test
```

## Project structure

```
tt_metal/
  CMakeLists.txt              # build system
  host/
    engine.cpp                # inference engine (forward pass, generate loop)
    engine.h                  # public API: load_model_and_tokenizer(), generate(), etc.
    gguf_loader.cpp           # GGUF weight loading into device DRAM MeshBuffers
    gguf_loader.h             # loader interface
    model_config.h            # Qwen3.5-9B hyperparameters and tile dimensions
  kernels/
    compute/                  # Tensix compute kernels (unused -- matmuls via ttnn)
    dataflow/                 # data movement kernels (unused -- using ttnn API)
  tests/
    test_device.cpp           # device validation
    test_matmul.cpp           # basic matmul test
    test_load_weights.cpp     # weight loading test
    test_forward.cpp          # end-to-end generation test
src/
  tokenizer.{h,cpp}          # BPE tokenizer (GPT-2 byte-level)
  download.{h,cpp}           # HuggingFace model download
```

## Key optimizations

- **BFLOAT8_B weights** on device (halves DRAM reads vs BF16, native HW decompression)
- **Metal Traces** for norm+matmul and FFN chain ops (zero dispatch overhead on replay)
- **On-device RMSNorm + FFN chain**: outproj + residual add + norm + gate(SiLU)/up/down matmuls + residual add, all in one traced sequence per layer
- **Two-chip split**: chip 0 for 32 layer weights, chip 1 for LM head (avoids DRAM contention)
- **Pre-allocated device buffers** for zero-allocation GEMV dispatch
- **Fast approximate exp()** (Schraudolph's method) for SiLU/sigmoid in conv1d and gated norm
- **Vectorized conv1d**: separated convolution from state update for auto-vectorization
- **Pre-computed RoPE tables** (avoids trig calls per token)
- **Raw bf16 bit operations** for f32<->bf16 conversion (avoids bfloat16 class overhead)
- **Static scratch buffers** for all forward pass intermediates (no heap allocation per token)
- **Program cache enabled** for faster repeated matmul dispatch
