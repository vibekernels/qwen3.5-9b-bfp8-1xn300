# Project structure

- `src/engine.cu` — inference engine: CUDA kernels, forward pass, `generate()`, `load_model_and_tokenizer()`
- `src/cli.cu` — CLI entry point (`qwen-inference`)
- `src/server.cu` — HTTP server entry point (`qwen-server`)
- `src/kernels/` — individual CUDA kernel files (attention, ffn, rmsnorm, rope, embedding, mamba)
- `src/model.h` — model config and weight structs
- `src/inference.h` — public API: `generate()`, `load_model_and_tokenizer()`, `reset_state()`, `get_tokenizer()`
- `src/tokenizer.{h,cpp}` — BPE tokenizer (GPT-2 byte-level)
- `src/sampling.{h,cu}` — token sampling (top-k, top-p, temperature)
- `src/download.{h,cpp}` — HuggingFace model download
- `tests/` — test suite

## Build & test

```sh
make -j$(nproc)    # build CLI + server
make test          # run all tests (CPU + GPU)
make test-cpu      # UTF-8 streaming buffer + tokenizer tests
make test-gpu      # inference integration tests (small + full context)
```

## Test inference

Run a one-shot completion (non-interactive) with:

```sh
./qwen-inference -m unsloth/Qwen3.5-9B-GGUF:BF16 -p "Your prompt here" -n 128
```

## Reference model (llama.cpp)

For comparison against llama.cpp:

```sh
llama-completion -hf unsloth/Qwen3.5-9B-GGUF:BF16 -p "Your prompt here" -n 128 -ngl 99
```

# currentDate
Today's date is 2026-03-07.
