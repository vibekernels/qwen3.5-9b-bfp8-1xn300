// SPDX-License-Identifier: Apache-2.0
// Multi-core GEMV reader: each core reads its slice of weight rows.
// Activation tiles loaded ONCE into cb_act (stays in L1 for all output rows).
// Weight tiles double-buffered: prefetch next tile while compute processes current.
//
// Compile-time args: [cb_act, cb_weight, Kt, acc_act_config, acc_weight_config]
// Runtime args: [act_addr, weight_addr, Mt_per_core, weight_start_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t act_addr         = get_arg_val<uint32_t>(0);
    uint32_t weight_addr      = get_arg_val<uint32_t>(1);
    uint32_t Mt_per_core      = get_arg_val<uint32_t>(2);
    uint32_t weight_start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_act    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt        = get_compile_time_arg_val(2);

    uint32_t act_tile_size    = get_tile_size(cb_act);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    constexpr auto acc_act_args = TensorAccessorArgs<3>();
    const auto acc_act = TensorAccessor(acc_act_args, act_addr, act_tile_size);

    constexpr auto acc_weight_args = TensorAccessorArgs<acc_act_args.next_compile_time_args_offset()>();
    const auto acc_weight = TensorAccessor(acc_weight_args, weight_addr, weight_tile_size);

    // Phase 1: Load ALL activation tiles into cb_act (stays in L1)
    for (uint32_t kt = 0; kt < Kt; kt++) {
        cb_reserve_back(cb_act, 1);
        uint32_t l1_act = get_write_ptr(cb_act);
        noc_async_read_tile(kt, acc_act, l1_act);
        noc_async_read_barrier();
        cb_push_back(cb_act, 1);
    }

    // Phase 2: For each output row, stream weight tiles (double-buffered via 2-tile CB)
    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t weight_row = weight_start_tile + mt;
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_reserve_back(cb_weight, 1);
            uint32_t l1_weight = get_write_ptr(cb_weight);
            noc_async_read_tile(weight_row * Kt + kt, acc_weight, l1_weight);
            noc_async_read_barrier();
            cb_push_back(cb_weight, 1);
        }
    }
}
