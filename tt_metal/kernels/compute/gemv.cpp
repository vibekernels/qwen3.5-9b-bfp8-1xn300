// SPDX-License-Identifier: Apache-2.0
// Compute kernel for GEMV: accumulates matmul_tiles over K dimension.
// For each of Mt output tiles, accumulates Kt matmul_tiles then packs result.
// Compile-time args: [Mt, Kt]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_act    = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    constexpr uint32_t cb_out    = tt::CBIndex::c_16;

    mm_init(cb_act, cb_weight, cb_out);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        acquire_dst();

        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_act, 1);
            cb_wait_front(cb_weight, 1);

            // Accumulate: dst[0] += act[0] @ weight[0]^T
            matmul_tiles(cb_act, cb_weight, 0, 0, 0);

            cb_pop_front(cb_act, 1);
            cb_pop_front(cb_weight, 1);
        }

        // Pack accumulated result to output CB
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
