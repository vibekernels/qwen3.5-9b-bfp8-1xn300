# qwen3.5-9b-bfp8b-1xn300

Custom inference engine for [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) on a single Tenstorrent N300 card (two Wormhole chips). All device operations use hand-written Tensix kernels via the tt-metalium C++ API — no ttnn dependency. Supports prefix caching for fast multi-turn conversations.

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?name=qwen3-5-9b-n300&type=docker&image=ghcr.io%2Fvibekernels%2Fqwen3.5-9b-bfp8b-1xn300:latest&instance_type=gpu-tenstorrent-n300s&regions=na&instances_min=1&hc_grace_period%5B8888%5D=900&ports=8888;http;/&ports=22;tcp;;true;tcp&env%5BPUBLIC_KEY%5D=REPLACE_ME)

https://github.com/user-attachments/assets/f9da983a-4178-4ab0-b22a-7cbaa2b7cf8e

## Performance

Measured on a Tenstorrent N300 (2x Wormhole) in a 4-CPU container. The first few tokens are slower while Metal Traces are captured; subsequent tokens run at steady state.

| Metric | Value |
|--------|-------|
| Decode throughput | ~8.2 tok/s (~122 ms/tok) |
| Prefill throughput | ~87 tok/s |
| Model size on device | ~9.5 GB (BFP8_B across 2 chips) |

## Requirements

- Tenstorrent N300 card
- clang-20
- tt-metalium SDK v0.66 (installed via debs)

## Setup

Install the tt-metalium debs (one-time):

```sh
~/install-tt-metal-debs.sh
```

Build everything:

```sh
make -j$(nproc)
```

The default model (`vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled`) is automatically downloaded from HuggingFace on first run and cached in `~/.cache/qwen-models/`.

Run the smoke test to verify everything works:

```sh
make quicktest    # "The capital of France is" -> Paris
```

Run the full integration test suite (~60s, 8 tests):

```sh
make test
```

## Running

```sh
make chat               # interactive chat CLI
make serve              # HTTP server on port 8888
make quicktest          # single-prompt smoke test
```

The HTTP server provides an OpenAI-compatible API and a built-in chat UI:

- `GET /` — Chat UI with streaming responses
- `POST /v1/chat/completions` — OpenAI-compatible chat API
- `GET /v1/models` — Model list
- `GET /health` — Health check

```sh
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## Docker

Build and run locally:

```sh
docker build -t qwen-n300 .
docker run --rm --device /dev/tenstorrent -p 8888:8888 qwen-n300
```

The container image is published to `ghcr.io/vibekernels/qwen3.5-9b-bfp8b-1xn300:latest` on every push to `main`. It can be deployed to [Koyeb](https://www.koyeb.com/) with a Tenstorrent N300 GPU instance — use the deploy button at the top of this page.

## Environment variables

All `make` targets and the Docker container use sensible defaults. Override as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled` | GGUF model file or HuggingFace tag |
| `CTX_SIZE` | `4096` | Maximum context length |
| `PORT` | `8888` | HTTP server port |
| `PUBLIC_KEY` | (none) | SSH public key for container remote access |
| `QUIET` | `0` | Set `1` to suppress per-token debug output |
| `NO_TRACES` | `0` | Set `1` to disable Metal Trace replay (useful for debugging) |
