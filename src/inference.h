#pragma once

#include "tokenizer.h"
#include <vector>
#include <string>
#include <functional>

// Token callback for streaming. Return false to stop generation.
using TokenCallback = std::function<bool(int token_id, const std::string& text)>;

// Load model weights + tokenizer from GGUF, allocate GPU buffers.
// max_ctx: maximum context length (prompt + generation).
bool load_model_and_tokenizer(const char* model_path, int max_ctx);

enum StopReason { STOP_EOS, STOP_LENGTH, STOP_CALLBACK };

// Run generation: prefill prompt_tokens, then decode up to max_tokens.
// Calls cb for each generated token. Returns total generated count.
// temperature <= 0 means greedy.
int generate(const std::vector<int>& prompt_tokens, int max_tokens,
             float temperature, TokenCallback cb, StopReason* stop_reason = nullptr);

// Reset all inference state (KV caches, SSM states, CUDA graph) between requests.
void reset_state();

// Access the loaded tokenizer.
Tokenizer& get_tokenizer();
