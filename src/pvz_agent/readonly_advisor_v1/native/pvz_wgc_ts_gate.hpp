#pragma once
#include <stdint.h>
#include "pvz_wgc.h"

constexpr uint32_t PVZ_WGC_MAX_STALE_PREOPEN_DROPS = 8u;
constexpr uint64_t PVZ_WGC_MAX_FRAME_AGE_100NS = 20000000ULL;
constexpr uint64_t PVZ_WGC_MAX_FUTURE_LEAD_100NS = 500000ULL;

inline uint32_t pvz_wgc_compute_lead_us(int64_t timestamp, uint64_t now) {
    if (timestamp <= 0 || now == 0u || static_cast<uint64_t>(timestamp) <= now) {
        return 0u;
    }
    const uint64_t diff = static_cast<uint64_t>(timestamp) - now;
    const uint64_t us = (diff / 10ULL) + ((diff % 10ULL) ? 1ULL : 0ULL);
    return us > 0xFFFFFFFFULL ? 0xFFFFFFFFu : static_cast<uint32_t>(us);
}

inline uint32_t pvz_wgc_classify_frame_ts(
    int64_t timestamp,
    uint64_t opened_qpc,
    uint64_t last_qpc,
    uint64_t now
) {
    if (timestamp <= 0) {
        return PVZ_TS_NONPOSITIVE;
    }
    const uint64_t ts = static_cast<uint64_t>(timestamp);
    if (last_qpc > 0u && ts <= last_qpc) {
        return PVZ_TS_NONMONOTONIC;
    }
    if (ts <= opened_qpc) {
        return PVZ_TS_PREOPEN;
    }
    if (now == 0u) {
        return PVZ_TS_PRE_QPC_ZERO;
    }
    if (ts > now && (ts - now) > PVZ_WGC_MAX_FUTURE_LEAD_100NS) {
        return PVZ_TS_PRE_AHEAD;
    }
    if (now >= ts && (now - ts) > PVZ_WGC_MAX_FRAME_AGE_100NS) {
        return PVZ_TS_PRE_TOO_OLD;
    }
    return PVZ_TS_OK;
}

inline bool pvz_wgc_should_drop_preopen(
    uint32_t accepted_frames,
    uint32_t dropped_so_far,
    uint32_t ts_code
) {
    return accepted_frames == 0u
        && dropped_so_far < PVZ_WGC_MAX_STALE_PREOPEN_DROPS
        && ts_code == PVZ_TS_PREOPEN;
}

inline uint32_t pvz_wgc_classify_end_ts(
    uint64_t timestamp,
    uint64_t end_qpc
) {
    if (end_qpc == 0u) {
        return PVZ_TS_POST_QPC_ZERO;
    }
    if (timestamp > end_qpc && (timestamp - end_qpc) > PVZ_WGC_MAX_FUTURE_LEAD_100NS) {
        return PVZ_TS_POST_AHEAD;
    }
    if (end_qpc >= timestamp && (end_qpc - timestamp) > PVZ_WGC_MAX_FRAME_AGE_100NS) {
        return PVZ_TS_POST_TOO_OLD;
    }
    return PVZ_TS_OK;
}

constexpr uint64_t PVZ_WGC_MAX_BUFFERED_POOL_AGE_100NS = 1500000ULL;

inline bool pvz_wgc_should_drop_buffered_frame(
    uint32_t accepted_frames,
    uint32_t dropped_so_far,
    uint32_t ts_code,
    int64_t timestamp,
    uint64_t now
) {
    if (dropped_so_far >= PVZ_WGC_MAX_STALE_PREOPEN_DROPS) {
        return false;
    }
    if (pvz_wgc_should_drop_preopen(accepted_frames, dropped_so_far, ts_code)) {
        return true;
    }
    if (ts_code == PVZ_TS_PRE_TOO_OLD) {
        return true;
    }
    if (ts_code == PVZ_TS_OK && timestamp > 0 && now > static_cast<uint64_t>(timestamp)
        && (now - static_cast<uint64_t>(timestamp)) > PVZ_WGC_MAX_BUFFERED_POOL_AGE_100NS) {
        return true;
    }
    return false;
}
