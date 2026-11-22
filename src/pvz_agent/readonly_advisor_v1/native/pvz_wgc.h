#pragma once
#include <stdint.h>
#include <wchar.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PvzWgcRequest {
    uint32_t size, version, pid, tid;
    uint64_t hwnd, creation_time_100ns, deadline_tick_ms;
    int32_t client_left, client_top, client_right, client_bottom;
    int32_t dwm_left, dwm_top, dwm_right, dwm_bottom;
    wchar_t class_name[128];
} PvzWgcRequest;
typedef struct PvzWgcAlphaDiag {
    uint32_t crop_x, crop_y;
    uint32_t alpha_0, alpha_1_250, alpha_251_254, alpha_255, alpha_min;
    uint32_t min_x, min_y, max_x, max_y;
    uint32_t corner_tl, corner_tr, corner_bl, corner_br;
    uint32_t edge_top, edge_bottom, edge_left, edge_right, interior, outer_1px;
    uint32_t hit_sun, hit_cards, hit_hud_header, hit_board;
} PvzWgcAlphaDiag;
typedef struct PvzWgcResult {
    uint32_t size,status,identity_before,identity_after,content_w,content_h,texture_w,texture_h;
    uint32_t client_w,client_h,stride,frame_index;
    uint64_t frame_qpc_100ns,bytes_written;
    int32_t client_left,client_top,client_right,client_bottom;
    int32_t dwm_left,dwm_top,dwm_right,dwm_bottom;
    uint32_t hresult,cleanup_state,alpha_nonopaque;
    uint32_t open_stage,apartment_hr,apartment_type,apartment_qualifier;
    uint32_t ts_reject_code,stale_preopen_dropped;
    uint32_t ts_lead_us,reserved0;
    PvzWgcAlphaDiag alpha_diag;
} PvzWgcResult;
/* Timestamp freshness gate reject codes (ts_reject_code):
 * 0: no timestamp rejection
 * 1: timestamp <= 0
 * 2: timestamp <= opened_qpc (after exceeding 8-frame pre-open drop cap or non-first frame)
 * 3: timestamp <= last_qpc (non-monotonic or repeated timestamp; never silently dropped)
 * 4: now == 0 (pre-read qpc100() failed)
 * 5: timestamp > now && (timestamp - now) > 500000ULL (pre-read frame timestamp >50ms ahead of now)
 * 6: now >= timestamp && (now - timestamp) > 20000000ULL (pre-read frame age > 2s)
 * 7: end_qpc == 0 (post-read qpc100() failed)
 * 8: timestamp > end_qpc && (timestamp - end_qpc) > 500000ULL (post-read timestamp >50ms ahead)
 * 9: end_qpc >= timestamp && (end_qpc - timestamp) > 20000000ULL (post-read frame age > 2s)
 */
enum PvzWgcTsRejectCode {
    PVZ_TS_OK = 0,
    PVZ_TS_NONPOSITIVE = 1,
    PVZ_TS_PREOPEN = 2,
    PVZ_TS_NONMONOTONIC = 3,
    PVZ_TS_PRE_QPC_ZERO = 4,
    PVZ_TS_PRE_AHEAD = 5,
    PVZ_TS_PRE_TOO_OLD = 6,
    PVZ_TS_POST_QPC_ZERO = 7,
    PVZ_TS_POST_AHEAD = 8,
    PVZ_TS_POST_TOO_OLD = 9
};
enum PvzWgcStatus { PVZ_OK=0, PVZ_NO_FRAME=1, PVZ_DENIED=2, PVZ_GEOMETRY=3, PVZ_API_ERROR=4, PVZ_UNSAFE_PENDING=5, PVZ_FRAME_LIMIT=6, PVZ_WRONG_THREAD=7, PVZ_CLEANUP_UNVERIFIED=8 };
__declspec(dllexport) int32_t __stdcall pvz_wgc_open(const PvzWgcRequest*,void**,PvzWgcResult*);
__declspec(dllexport) int32_t __stdcall pvz_wgc_read(void*,uint8_t*,uint64_t,PvzWgcResult*);
__declspec(dllexport) int32_t __stdcall pvz_wgc_close(void*);
#ifdef __cplusplus
}
#endif
