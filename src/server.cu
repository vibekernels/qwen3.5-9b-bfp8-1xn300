#include "inference.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <random>
#include <sstream>

// Suppress warnings from third-party headers under nvcc
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20208
#pragma nv_diag_suppress 611

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

static std::mutex g_inference_mutex;

static std::string generate_id() {
    static std::mt19937 rng(std::random_device{}());
    static const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::string id = "chatcmpl-";
    for (int i = 0; i < 16; i++) {
        id += chars[rng() % (sizeof(chars) - 1)];
    }
    return id;
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Apply Qwen ChatML template to OpenAI-format messages
static std::string apply_chat_template(const json& messages) {
    std::string prompt;
    for (const auto& msg : messages) {
        std::string role = msg.value("role", "user");
        std::string content = msg.value("content", "");
        prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    // Add generation prompt with thinking disabled
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return prompt;
}

static void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"Invalid JSON","type":"invalid_request_error"}})", "application/json");
        return;
    }

    auto messages = body.value("messages", json::array());
    if (messages.empty()) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"messages is required","type":"invalid_request_error"}})", "application/json");
        return;
    }

    int max_tokens = body.value("max_tokens", 1024);
    float temperature = body.value("temperature", 0.7f);
    bool stream = body.value("stream", false);
    std::string model_name = body.value("model", "qwen3.5-9b");

    // Apply chat template and tokenize
    std::string prompt_text = apply_chat_template(messages);
    auto& tokenizer = get_tokenizer();
    std::vector<int> prompt_tokens = tokenizer.encode(prompt_text);

    std::string completion_id = generate_id();
    int64_t created = unix_timestamp();

    if (stream) {
        // SSE streaming response
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        // Send initial role delta
        res.set_chunked_content_provider("text/event-stream",
            [&, prompt_tokens, max_tokens, temperature, model_name, completion_id, created]
            (size_t offset, httplib::DataSink& sink) -> bool {

                // Send role delta first
                json role_chunk = {
                    {"id", completion_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", model_name},
                    {"choices", json::array({
                        {{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}
                    })}
                };
                std::string role_data = "data: " + role_chunk.dump() + "\n\n";
                sink.write(role_data.c_str(), role_data.size());

                StopReason reason;

                {
                    std::lock_guard<std::mutex> lock(g_inference_mutex);
                    reset_state();
                    generate(prompt_tokens, max_tokens, temperature,
                        [&](int token_id, const std::string& text) -> bool {
                            json chunk = {
                                {"id", completion_id},
                                {"object", "chat.completion.chunk"},
                                {"created", created},
                                {"model", model_name},
                                {"choices", json::array({
                                    {{"index", 0}, {"delta", {{"content", text}}}, {"finish_reason", nullptr}}
                                })}
                            };
                            std::string data = "data: " + chunk.dump() + "\n\n";
                            return sink.write(data.c_str(), data.size());
                        }, &reason);
                }

                std::string finish_reason = (reason == STOP_EOS) ? "stop" : "length";

                // Send final chunk with finish_reason
                json final_chunk = {
                    {"id", completion_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", model_name},
                    {"choices", json::array({
                        {{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}
                    })}
                };
                std::string final_data = "data: " + final_chunk.dump() + "\n\n";
                sink.write(final_data.c_str(), final_data.size());

                // Send [DONE]
                std::string done = "data: [DONE]\n\n";
                sink.write(done.c_str(), done.size());
                sink.done();
                return true;
            }
        );
    } else {
        // Non-streaming response
        std::string full_response;
        int total_tokens = 0;
        StopReason reason;

        {
            std::lock_guard<std::mutex> lock(g_inference_mutex);
            reset_state();
            total_tokens = generate(prompt_tokens, max_tokens, temperature,
                [&](int token_id, const std::string& text) -> bool {
                    full_response += text;
                    return true;
                }, &reason);
        }

        std::string finish_reason = (reason == STOP_EOS) ? "stop" : "length";

        json response = {
            {"id", completion_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", model_name},
            {"choices", json::array({
                {{"index", 0},
                 {"message", {{"role", "assistant"}, {"content", full_response}}},
                 {"finish_reason", finish_reason}}
            })},
            {"usage", {
                {"prompt_tokens", (int)prompt_tokens.size()},
                {"completion_tokens", total_tokens},
                {"total_tokens", (int)prompt_tokens.size() + total_tokens}
            }}
        };

        res.set_content(response.dump(), "application/json");
    }
}

static void handle_models(const httplib::Request&, httplib::Response& res) {
    json response = {
        {"object", "list"},
        {"data", json::array({
            {{"id", "qwen3.5-9b"}, {"object", "model"}, {"owned_by", "local"}}
        })}
    };
    res.set_content(response.dump(), "application/json");
}

static void handle_health(const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8080;
    int ctx_size = 4096;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
            ctx_size = atoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s -m <model_path> [--host <addr>] [--port <port>] [--ctx-size <n>]\n", argv[0]);
        return 1;
    }

    printf("Loading model: %s\n", model_path.c_str());
    if (!load_model_and_tokenizer(model_path.c_str(), ctx_size)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded. Context size: %d\n", ctx_size);

    httplib::Server svr;

    // CORS headers for browser clients
    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Post("/v1/chat/completions", handle_chat_completions);
    svr.Get("/v1/models", handle_models);
    svr.Get("/health", handle_health);

    printf("Server listening on %s:%d\n", host.c_str(), port);
    printf("  POST /v1/chat/completions\n");
    printf("  GET  /v1/models\n");
    printf("  GET  /health\n");

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "Failed to bind to %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
