#pragma once
#include <stdint.h>
#include "pvz_wgc.h"

inline bool pvz_wgc_compute_alpha_diag(
    const uint8_t* bgra,
    uint32_t width,
    uint32_t height,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t* out_nonopaque,
    PvzWgcAlphaDiag* out_diag
) {
    if (!bgra || !out_nonopaque || !out_diag) {
        return false;
    }
    const bool is_800 = (width == 800u && height == 600u);
    const bool is_1200 = (width == 1200u && height == 900u);
    if (!is_800 && !is_1200) {
        return false;
    }

    const uint32_t b = is_800 ? 8u : 12u;
    const uint32_t sun_x1 = is_800 ? 24u : 36u;
    const uint32_t sun_x2 = is_800 ? 68u : 102u;
    const uint32_t sun_y1 = is_800 ? 55u : 82u;
    const uint32_t sun_y2 = is_800 ? 72u : 108u;

    const uint32_t cards_x1 = is_800 ? 90u : 135u;
    const uint32_t cards_x2 = is_800 ? 597u : 896u;
    const uint32_t cards_y1 = is_800 ? 8u : 12u;
    const uint32_t cards_y2 = is_800 ? 78u : 117u;

    const uint32_t hud_x1 = is_800 ? 14u : 21u;
    const uint32_t hud_x2 = is_800 ? 598u : 897u;
    const uint32_t hud_y1 = is_800 ? 8u : 12u;
    const uint32_t hud_y2 = is_800 ? 78u : 117u;

    const uint32_t board_x1 = is_800 ? 40u : 60u;
    const uint32_t board_x2 = is_800 ? 760u : 1140u;
    const uint32_t board_y1 = is_800 ? 80u : 120u;
    const uint32_t board_y2 = is_800 ? 590u : 885u;

    PvzWgcAlphaDiag diag{};
    diag.crop_x = crop_x;
    diag.crop_y = crop_y;
    diag.alpha_min = 255u;
    diag.min_x = width;
    diag.min_y = height;
    diag.max_x = 0u;
    diag.max_y = 0u;

    uint32_t nonopaque = 0u;
    const uint32_t right_edge_start = width - b;
    const uint32_t bottom_edge_start = height - b;
    const uint32_t last_x = width - 1u;
    const uint32_t last_y = height - 1u;

    for (uint32_t row = 0u; row < height; ++row) {
        const uint8_t* row_ptr = bgra + static_cast<uint64_t>(row) * width * 4u;
        const bool is_top = (row < b);
        const bool is_bottom = (row >= bottom_edge_start);
        const bool is_outer_y = (row == 0u || row == last_y);
        const bool in_sun_y = (row >= sun_y1 && row < sun_y2);
        const bool in_cards_y = (row >= cards_y1 && row < cards_y2);
        const bool in_hud_y = (row >= hud_y1 && row < hud_y2);
        const bool in_board_y = (row >= board_y1 && row < board_y2);

        for (uint32_t col = 0u; col < width; ++col) {
            const uint32_t a = static_cast<uint32_t>(row_ptr[static_cast<uint64_t>(col) * 4u + 3u]);
            if (a < diag.alpha_min) {
                diag.alpha_min = a;
            }
            if (a == 255u) {
                ++diag.alpha_255;
                continue;
            }

            ++nonopaque;
            if (a == 0u) {
                ++diag.alpha_0;
            } else if (a <= 250u) {
                ++diag.alpha_1_250;
            } else {
                ++diag.alpha_251_254;
            }

            if (col < diag.min_x) diag.min_x = col;
            if (col > diag.max_x) diag.max_x = col;
            if (row < diag.min_y) diag.min_y = row;
            if (row > diag.max_y) diag.max_y = row;

            const bool is_left = (col < b);
            const bool is_right = (col >= right_edge_start);
            if (is_top) {
                if (is_left) {
                    ++diag.corner_tl;
                } else if (is_right) {
                    ++diag.corner_tr;
                } else {
                    ++diag.edge_top;
                }
            } else if (is_bottom) {
                if (is_left) {
                    ++diag.corner_bl;
                } else if (is_right) {
                    ++diag.corner_br;
                } else {
                    ++diag.edge_bottom;
                }
            } else if (is_left) {
                ++diag.edge_left;
            } else if (is_right) {
                ++diag.edge_right;
            } else {
                ++diag.interior;
            }

            if (is_outer_y || col == 0u || col == last_x) {
                ++diag.outer_1px;
            }

            if (in_sun_y && col >= sun_x1 && col < sun_x2) {
                ++diag.hit_sun;
            }
            if (in_cards_y && col >= cards_x1 && col < cards_x2) {
                ++diag.hit_cards;
            }
            if (in_hud_y && col >= hud_x1 && col < hud_x2) {
                ++diag.hit_hud_header;
            }
            if (in_board_y && col >= board_x1 && col < board_x2) {
                ++diag.hit_board;
            }
        }
    }

    if (nonopaque == 0u) {
        diag.min_x = 65535u;
        diag.min_y = 65535u;
        diag.max_x = 65535u;
        diag.max_y = 65535u;
    }

    *out_nonopaque = nonopaque;
    *out_diag = diag;
    return true;
}
