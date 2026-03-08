// SPDX-License-Identifier: Apache-2.0
// Single-kernel RMSNorm: reads input + weight, computes RMSNorm in scalar, writes output.
// No compute kernel needed — runs entirely on the dataflow RISC-V.
// For [1, 4096] vectors this takes ~10us.
//
// Uses a single CB as L1 scratch (reserve, read, process, release pattern).
//
// Compile-time args: [n_tiles, acc_in_config, acc_weight_config, acc_out_config]
// Runtime args: [in_addr, weight_addr, out_addr, n_elements]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

void kernel_main() {
    uint32_t in_addr     = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t out_addr    = get_arg_val<uint32_t>(2);
    uint32_t n_elements  = get_arg_val<uint32_t>(3);

    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_scratch = tt::CBIndex::c_0;
    uint32_t tile_size = get_tile_size(cb_scratch);

    constexpr auto acc_in_args = TensorAccessorArgs<1>();
    const auto acc_in = TensorAccessor(acc_in_args, in_addr, tile_size);
    constexpr auto acc_w_args = TensorAccessorArgs<acc_in_args.next_compile_time_args_offset()>();
    const auto acc_w = TensorAccessor(acc_w_args, weight_addr, tile_size);
    constexpr auto acc_out_args = TensorAccessorArgs<acc_w_args.next_compile_time_args_offset()>();
    const auto acc_out = TensorAccessor(acc_out_args, out_addr, tile_size);

    // Pass 1: compute sum of squares
    // Data layout in bf16 tiles: row 0 = elements [0..15], row 16 = elements [16..31]
    float sum_sq = 0.0f;

    for (uint32_t t = 0; t < n_tiles; t++) {
        cb_reserve_back(cb_scratch, 1);
        uint32_t l1_addr = get_write_ptr(cb_scratch);
        noc_async_read_tile(t, acc_in, l1_addr);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
        uint32_t base = t * 32;

        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float v = bf16_to_f32(d[j]);
            sum_sq += v * v;
        }
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float v = bf16_to_f32(d[256 + j]);
            sum_sq += v * v;
        }

        // Release CB slot so we can reuse it
        cb_push_back(cb_scratch, 1);
        cb_wait_front(cb_scratch, 1);
        cb_pop_front(cb_scratch, 1);
    }

    // Compute 1/sqrt(mean(x^2) + eps) using fast inverse sqrt
    float mean_sq = sum_sq / (float)n_elements;
    float val = mean_sq + 1e-6f;
    float x2 = val * 0.5f;
    uint32_t i;
    __builtin_memcpy(&i, &val, 4);
    i = 0x5f3759df - (i >> 1);
    float norm_factor;
    __builtin_memcpy(&norm_factor, &i, 4);
    norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);
    norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);

    // Pass 2: normalize and multiply by weight
    // For each tile: read input, read weight, compute out = x * norm_factor * w, write output
    for (uint32_t t = 0; t < n_tiles; t++) {
        // Read input tile
        cb_reserve_back(cb_scratch, 1);
        uint32_t l1_addr = get_write_ptr(cb_scratch);
        noc_async_read_tile(t, acc_in, l1_addr);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
        uint32_t base = t * 32;

        // Extract normalized input values
        float norm_vals[32];
        for (uint32_t j = 0; j < 16; j++) {
            norm_vals[j] = (base + j < n_elements) ? bf16_to_f32(d[j]) * norm_factor : 0.0f;
        }
        for (uint32_t j = 0; j < 16; j++) {
            norm_vals[16 + j] = (base + 16 + j < n_elements) ? bf16_to_f32(d[256 + j]) * norm_factor : 0.0f;
        }

        // Read weight tile into same L1 slot
        noc_async_read_tile(t, acc_w, l1_addr);
        noc_async_read_barrier();

        // Multiply by weight and write back
        for (uint32_t j = 0; j < 16; j++) {
            if (base + j < n_elements) {
                float result = norm_vals[j] * bf16_to_f32(d[j]);
                d[j] = f32_to_bf16(result);
            }
        }
        for (uint32_t j = 0; j < 16; j++) {
            if (base + 16 + j < n_elements) {
                float result = norm_vals[16 + j] * bf16_to_f32(d[256 + j]);
                d[256 + j] = f32_to_bf16(result);
            }
        }

        // Write output tile
        noc_async_write_tile(t, acc_out, l1_addr);
        noc_async_write_barrier();

        cb_push_back(cb_scratch, 1);
        cb_wait_front(cb_scratch, 1);
        cb_pop_front(cb_scratch, 1);
    }
}
