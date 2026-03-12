# Qwen 3.5-9B inference on Tenstorrent N300
# Usage:
#   make                  — build everything
#   make quicktest        — smoke test: "The capital of France is"
#   make test             — run full integration tests
#   make clean            — remove build artifacts
#
# Environment:
#   TT_METAL_DEB          — deb-installed SDK prefix (default: /usr)
#                           requires: tt-metalium tt-metalium-dev tt-metalium-jit tt-nn debs
#   MODEL_PATH            — path to .gguf model     (default: vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled)

TT_METAL_DEB   ?= /usr
CXX            := clang++-20
BUILD          := build

_TT_LIBEXEC := $(TT_METAL_DEB)/libexec/tt-metalium

TT_INCLUDES := \
	-isystem $(_TT_LIBEXEC)/tt_metal/hostdevcommon/api \
	-isystem $(_TT_LIBEXEC) \
	-isystem $(TT_METAL_DEB)/include/metalium-thirdparty

TT_LIBS := \
	$(TT_METAL_DEB)/lib/libtt_metal.so \
	$(TT_METAL_DEB)/lib/libtracy.so.0.10.0 \
	-ldl \
	$(TT_METAL_DEB)/lib/libdevice.so \
	$(TT_METAL_DEB)/lib/libtt_stl.so \
	-lspdlog -lfmt

TT_DEFINES := \
	-DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 \
	-DFMT_HEADER_ONLY=1 \
	-DNTEST \
	-DSPDLOG_COMPILED_LIB \
	-DSPDLOG_FMT_EXTERNAL \
	-DTRACY_ENABLE \
	-DTRACY_IMPORTS \
	-DTRACY_TIMER_FALLBACK

CXXFLAGS := -O3 -std=gnu++20 -Wno-int-to-pointer-cast -fno-omit-frame-pointer \
	$(TT_DEFINES) $(TT_INCLUDES) -I src -I third_party \
	-DKERNEL_DIR=\"$(CURDIR)/src/kernels\"

MARCH ?= native
ENGINE_CXXFLAGS := $(CXXFLAGS) -march=$(MARCH) -ffast-math
LDFLAGS := -rdynamic -Wl,-rpath,$(TT_METAL_DEB)/lib

# Source files
ENGINE_SRCS := src/engine.cpp src/gguf_loader.cpp src/tokenizer.cpp
ENGINE_OBJS := $(ENGINE_SRCS:%.cpp=$(BUILD)/%.o)

# Targets
.PHONY: all clean test quicktest chat serve

all: $(BUILD)/test_forward $(BUILD)/test_inference $(BUILD)/qwen-chat $(BUILD)/qwen-server $(BUILD)/make_bfp8_gguf

# Engine static library
$(BUILD)/libqwen_engine.a: $(ENGINE_OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

$(BUILD)/src/engine.o: src/engine.cpp
	@mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(BUILD)/src/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main test targets (link against engine)
$(BUILD)/test_forward: src/tests/test_forward.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/tests/test_forward.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

$(BUILD)/test_inference: src/tests/test_inference.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/tests/test_inference.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

# Interactive chat CLI
$(BUILD)/qwen-chat: src/chat.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/chat.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS)

chat: $(BUILD)/qwen-chat
	@env TT_METAL_RUNTIME_ROOT=$(_TT_LIBEXEC) QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/qwen-chat \
		$${MODEL_PATH:-vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled} 2>/dev/null

# HTTP server with chat UI
$(BUILD)/qwen-server: src/server.cpp src/download.cpp $(BUILD)/libqwen_engine.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) src/server.cpp src/download.cpp -o $@ $(LDFLAGS) $(BUILD)/libqwen_engine.a $(TT_LIBS) -lpthread

serve: $(BUILD)/qwen-server
	@env TT_METAL_RUNTIME_ROOT=$(_TT_LIBEXEC) QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/qwen-server \
		-m $${MODEL_PATH:-vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled} \
		--port $${PORT:-8888} 2>/dev/null

# BFP8_B GGUF converter — downloads BF16, converts to BFP8_B tiled GGUF
$(BUILD)/make_bfp8_gguf: tools/make_bfp8_gguf.cpp src/download.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -march=$(MARCH) $< src/download.cpp -o $@ $(LDFLAGS) $(TT_LIBS)

# Standalone test targets (link directly against tt-metal)
$(BUILD)/test_matmul: src/tests/test_matmul.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TT_LIBS)

$(BUILD)/test_dram_bw: src/tests/test_dram_bw.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TT_LIBS)

$(BUILD)/test_mesh_overhead: src/tests/test_mesh_overhead.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TT_LIBS)

# Quick smoke test: "The capital of France is" → should output "Paris"
quicktest: $(BUILD)/test_forward
	@env TT_METAL_RUNTIME_ROOT=$(_TT_LIBEXEC) QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/test_forward \
		$${MODEL_PATH:-vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled} \
		"The capital of France is" 16 --raw 2>/dev/null

# Run integration test suite
test: $(BUILD)/test_inference
	@env TT_METAL_RUNTIME_ROOT=$(_TT_LIBEXEC) \
		MODEL_PATH=$${MODEL_PATH:-vibekernels/Qwen3.5-9B-GGUF:BFP8B-tiled} \
		QUIET=1 \
		TT_METAL_OPERATION_TIMEOUT_SECONDS=$${TT_METAL_OPERATION_TIMEOUT_SECONDS:-5} \
		$(BUILD)/test_inference 2>/dev/null

clean:
	rm -rf $(BUILD)
