"""Comprehensive offline test suite for PVZagent Read-Only Real-Time Advice Assistant v1.0."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pvz_agent.product_startup import verify_product_startup
from pvz_agent.readonly_advisor_v1.advisor_runtime import (
    AdvisorController,
    ContinuousAdviceEngine,
)
from pvz_agent.readonly_advisor_v1.async_llm_coach import AsyncLlmCoach
from pvz_agent.readonly_advisor_v1.fast_perception import (
    NO_EVIDENCE_TEXT,
    RGB_BYTES,
    FastGeometryNormalizer,
    FastSinglePassPerception,
    PhaseTracker,
    evaluate_three_channels,
)
from pvz_agent.readonly_advisor_v1.local_settings import LocalSettingsStore
from pvz_agent.readonly_advisor_v1.sidebar_app import ReadonlyAdvisorSidebarApp
from pvz_agent.readonly_advisor_v1.wgc_capture import (
    ALLOWED_EXE_NAMES,
    ALPHA_DIAG_FIELDS,
    CapturedFrame,
    DLL_PATH,
    WindowLease,
    matches_pvz_exe_path,
    perimeter_alpha_ok,
)

_CANDIDATE_REAL_FRAMES = [
    REPO_ROOT / "evidence/day06/goal-live-v1/live-01/captures/longrun-2026-10-24T12-08-06-630780+00-00-d8a0e9.png",
    REPO_ROOT / "evidence/day06/goal-live-v1/live-01/captures/longrun-2026-10-24T12-08-33-621940+00-00-a49852.png",
    REPO_ROOT / "evidence/day06/bucket-angle-v1/live-01/captures/longrun-2026-10-24T13-23-31-056824+00-00-793f95.png",
]
_FALLBACK_REF_FRAME = REPO_ROOT / "evidence/day06/new-material-a-20261107-v1/canonical/frame-0000.png"
REAL_FRAMES = _CANDIDATE_REAL_FRAMES if all(p.is_file() for p in _CANDIDATE_REAL_FRAMES) else [_FALLBACK_REF_FRAME] * 3


def _load_bgra_1200(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"missing test frame {path}"
    bgr1200 = cv2.resize(bgr, (1200, 900), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr1200, cv2.COLOR_BGR2BGRA)


class _FakeClock:
    def __init__(self, start: float = 100.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


class _FakePerception:
    def __init__(self, frames_spec: list[dict]):
        self.specs = list(frames_spec)
        self.calls = 0

    def observe(self, _img, *, run_id, state_id, capture_time, frame_time_seconds, evidence, tracking_epoch):
        idx = min(self.calls, len(self.specs) - 1)
        spec = self.specs[idx]
        self.calls += 1
        row = spec.get("row", 3)
        hx = spec.get("head_x", 500)
        heads_n = spec.get("heads", 1)
        sun = spec.get("sun", 150)
        unc_edge = spec.get("uncertain_edge", False)
        occ = spec.get("occ", 12)
        cands = [{"row": row, "head_xy": [float(hx + i * 25), float(row * 85)]} for i in range(heads_n)]
        cells = []
        for i in range(54):
            r = (i // 9) + 1
            c = (i % 9) + 1
            cells.append({"row": r, "column": c, "occupancy": "ground_occupied" if i < occ else "empty"})
        return {
            "run_id": run_id,
            "state_id": state_id,
            "capture_time": capture_time,
            "frame_time_seconds": frame_time_seconds,
            "evidence": dict(evidence),
            "scene": {"value": spec.get("scene", "playing")},
            "input_sent": False,
            "action_eligible": False,
            "fields": {
                "state_id": state_id,
                "capture_time": capture_time,
                "scene": {"value": spec.get("scene", "playing")},
                "sun": {"value": sun},
                "cards": [{"slot": s, "visual_state": "bright", "raw_visual_state": "bright"} for s in range(1, 11)],
                "cells": cells,
            },
            "zombies": {
                "candidates": cands,
                "uncertain_edge_presence": [{"x": 795}] if unc_edge else [],
                "uncertain_water_candidates": [],
            },
            "vehicles": {"candidates": [], "tracks": [], "uncertain_candidates": []},
            "resources": {
                "sun_appearance": {
                    "schema": "pvz-sun-appearance/1",
                    "status": "observed",
                    "state_id": state_id,
                    "capture_time": capture_time,
                    "evidence": dict(evidence),
                    "candidates": [],
                }
            },
            "loadout": {
                "schema": "pvz-card-identity/1",
                "status": "observed",
                "state_id": state_id,
                "capture_time": capture_time,
                "source_run_id": run_id,
                "evidence": dict(evidence),
                "slots": [
                    {"slot": s, "status": "observed", "plant_id": "sunflower", "display_cost": 50}
                    for s in range(1, 11)
                ],
            },
        }


class ReadonlyAdvisorV1Tests(unittest.TestCase):
    def test_1_native_dll_and_self_test_and_alpha_gate(self):
        import os
        from unittest import mock

        self.assertEqual(verify_product_startup(REPO_ROOT)["artifacts_checked"], 59)
        self.assertTrue(DLL_PATH.is_file())
        dll_bytes = DLL_PATH.read_bytes()
        for name in ALLOWED_EXE_NAMES:
            self.assertIn(name.encode("utf-16-le"), dll_bytes)
        self.assertNotIn("PVZ原版".encode("utf-16-le"), dll_bytes)

        # Portable exe-path matching + optional PVZ_GAME_EXE exact pin
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PVZ_GAME_EXE", None)
            self.assertTrue(matches_pvz_exe_path(r"C:\Games\PVZ\PVZ-Windowed.exe"))
            self.assertTrue(matches_pvz_exe_path(r"D:\Steam\PlantsVsZombies.exe"))
            self.assertFalse(matches_pvz_exe_path(r"C:\Tools\Magpie.exe"))
        with mock.patch.dict(os.environ, {"PVZ_GAME_EXE": r"E:\Custom\MyPVZ.exe"}):
            self.assertTrue(matches_pvz_exe_path(r"E:\Custom\MyPVZ.exe"))
            self.assertFalse(matches_pvz_exe_path(r"C:\Games\PVZ\PVZ-Windowed.exe"))

        diag = {k: 0 for k in ALPHA_DIAG_FIELDS}
        diag.update({
            "crop_x": 2, "crop_y": 39, "alpha_1_250": 962, "alpha_255": 1200 * 900 - 962,
            "alpha_min": 47, "min_x": 0, "min_y": 0, "max_x": 1199, "max_y": 899,
            "corner_tr": 12, "edge_right": 876, "corner_bl": 31, "corner_br": 43, "outer_1px": 929,
        })
        self.assertTrue(perimeter_alpha_ok(962, diag, (1200, 900)))
        bad = dict(diag, interior=1, edge_right=875)
        self.assertFalse(perimeter_alpha_ok(962, bad, (1200, 900)))

        res = subprocess.run(
            [sys.executable, "-B", str(REPO_ROOT / "scripts/pvz_readonly_advisor.py"), "--self-test"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        payload = json.loads(res.stdout.strip())
        self.assertEqual(payload["status"], "SELF_TEST_PASS")
        self.assertEqual(payload["artifacts_checked"], 59)
        self.assertFalse(payload["input_sent"])
        self.assertTrue(payload["stop_latched"])

    def test_2_fast_geometry_cache_and_single_pass_speedup_on_real_frames(self):
        norm = FastGeometryNormalizer(REPO_ROOT)
        perc = FastSinglePassPerception(REPO_ROOT)
        rect_sig = (102, 139, 1302, 1039, 100, 100, 1303, 1041)

        cached_totals_ms = []
        for idx, fpath in enumerate(REAL_FRAMES, 1):
            bgra = _load_bgra_1200(fpath)
            t0 = time.perf_counter()
            rgb, reason, diag = norm.normalize(bgra, (1200, 900), rect_sig)
            self.assertIsNone(reason, f"Frame {idx} norm failed: {reason}")
            self.assertEqual(rgb.shape, (600, 800, 3))
            if idx == 1:
                self.assertEqual(diag["mode"], "sift")
            else:
                self.assertEqual(diag["mode"], "cached_affine")
                self.assertLess(diag["elapsed_ms"], 60.0)

            engine = ContinuousAdviceEngine(clock=time.monotonic, perception=perc, root=REPO_ROOT, threaded_expiry=False)
            try:
                published, ch, advice = engine.ingest_canonical_rgb(rgb.tobytes(), (1200, 900))
                total_ms = (time.perf_counter() - t0) * 1000.0
                self.assertIsNotNone(published)
                self.assertIsNotNone(ch)
                self.assertIsNotNone(advice)
                self.assertGreaterEqual(ch["zombie_heads"], 1)
                if idx >= 2:
                    cached_totals_ms.append(total_ms)
            finally:
                engine.close()

        self.assertEqual(norm.sift_runs, 1)
        self.assertEqual(norm.fast_hits, 2)
        # Cached frames should be dramatically faster than v1r7's ~1620 ms (< 1100 ms even under load)
        self.assertTrue(all(ms < 1100.0 for ms in cached_totals_ms), f"Unexpected slow frames: {cached_totals_ms}")

        # Negative check: uniform black frame rejected
        black = np.zeros((900, 1200, 4), dtype=np.uint8)
        rgb_b, reason_b, _ = norm.normalize(black, (1200, 900), rect_sig)
        self.assertIsNone(rgb_b)
        self.assertEqual(reason_b, "uniform_or_black")

    def test_3_watchdog_fail_closed_dedup_and_phase_timeline_restart(self):
        fake_perc = _FakePerception([
            {"row": 2, "head_x": 500, "heads": 1, "occ": 4},
            {"row": 2, "head_x": 490, "heads": 1, "occ": 5},   # duplicate advice -> coalesced
            {"row": 2, "head_x": 480, "heads": 1, "occ": 14, "uncertain_edge": True},  # withdraws current
            {"row": 5, "head_x": 180, "heads": 2, "occ": 15},  # left warning + new advice
            {"row": 1, "head_x": 600, "heads": 1, "occ": 2},   # sharp drop 15->2 triggers new run_id
        ])
        clock = _FakeClock(100.0)
        engine = ContinuousAdviceEngine(clock=clock, perception=fake_perc, root=REPO_ROOT, threaded_expiry=False)
        self.addCleanup(engine.close)
        tracker = PhaseTracker()
        first_run = tracker.run_id
        rgb = bytes([42]) * RGB_BYTES

        _, ch1, adv1 = engine.ingest_canonical_rgb(rgb, (1200, 900), rel_time_text="00:00")
        self.assertEqual(adv1.focus_row, 2)
        tracker.update(ch1, clock(), advice_added=True)

        clock.now = 100.8
        _, ch2, adv2 = engine.ingest_canonical_rgb(rgb, (1200, 900), rel_time_text="00:01")
        self.assertEqual(adv2.focus_row, 2)
        self.assertEqual(engine.sidebar.duplicate_coalesced_count, 1)
        self.assertEqual(len(engine.sidebar.read().history), 1)
        tracker.update(ch2, clock(), advice_added=False)

        clock.now = 101.5
        _, ch3, adv3 = engine.ingest_canonical_rgb(rgb, (1200, 900), rel_time_text="00:02")
        self.assertIsNone(adv3)
        self.assertIsNone(engine.sidebar.read().current)
        self.assertEqual(len(engine.sidebar.read().history), 1)
        self.assertGreaterEqual(engine.sidebar.withdrawn_count, 1)

        clock.now = 102.2
        _, ch4, adv4 = engine.ingest_canonical_rgb(rgb, (1200, 900), rel_time_text="00:02")
        self.assertEqual(adv4.focus_row, 5)
        self.assertEqual(len(engine.sidebar.read().history), 2)
        tracker.update(ch4, clock(), advice_added=True)
        snap_tl = tracker.snapshot(clock())
        self.assertTrue(any("靠左告警" in e for s in snap_tl["segments"] for e in s["events"]))

        # Restart detection
        clock.now = 103.0
        _, ch5, _ = engine.ingest_canonical_rgb(rgb, (1200, 900), rel_time_text="00:03")
        tracker.update(ch5, clock(), advice_added=True)
        self.assertNotEqual(tracker.run_id, first_run)

        # Live OS 2.0s watchdog test
        live_engine = ContinuousAdviceEngine(
            clock=time.monotonic, perception=_FakePerception([{"row": 3, "head_x": 450, "heads": 1}]),
            root=REPO_ROOT, threaded_expiry=True,
        )
        self.addCleanup(live_engine.close)
        _, _, adv_live = live_engine.ingest_canonical_rgb(rgb, (1200, 900))
        self.assertIsNotNone(adv_live)
        deadline = time.monotonic() + 2.5
        while live_engine.expiry.retire_count == 0 and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertGreaterEqual(live_engine.expiry.retire_count, 1)
        self.assertIsNone(live_engine.sidebar.read().current)
        self.assertEqual(len(live_engine.sidebar.read().history), 1)

    def test_4_async_llm_coach_default_off_and_opt_in_recheck(self):
        calls = []
        gate = threading.Event()

        class _FakeProvider:
            def complete(self, prompt, *, image_data_url, json_schema):
                calls.append((prompt, image_data_url[:22]))
                gate.wait(timeout=2.0)
                sid = prompt.split("source_state_id=")[-1].strip()
                return {
                    "text": json.dumps({
                        "source_state_id": sid,
                        "goal": "defend_lane",
                        "focus_row": 3,
                        "reason": "第3路有可见僵尸，建议先补该路输出再兼顾对称布局。",
                        "uncertainties": ["未核实铁桶血量"],
                    }, ensure_ascii=False),
                    "model_returned": "gpt-6.1-sol",
                }

        coach = AsyncLlmCoach(
            clock=time.monotonic,
            provider_factory=lambda _c: _FakeProvider(),
            enabled=False,
            min_interval_s=10.0,
            vision_mode="high_png",
        )
        self.addCleanup(coach.close)
        fake_perc = _FakePerception([{"row": 3, "head_x": 460, "heads": 1}])
        engine = ContinuousAdviceEngine(clock=time.monotonic, perception=fake_perc, root=REPO_ROOT, threaded_expiry=False)
        self.addCleanup(engine.close)
        rgb = bytes([10]) * RGB_BYTES
        pub, ch, _ = engine.ingest_canonical_rgb(rgb, (1200, 900))
        phase_snap = PhaseTracker().snapshot(time.monotonic())

        # Default OFF -> 0 calls
        self.assertFalse(coach.maybe_dispatch(state=pub, rgb_bytes=rgb, phase_info=phase_snap, channel_info=ch))
        self.assertEqual(len(calls), 0)

        # Opt-in ON -> non-blocking dispatch, waiting status, then verified on fresh frame
        coach.set_enabled(True)
        coach._provider = _FakeProvider()
        self.assertTrue(coach.maybe_dispatch(state=pub, rgb_bytes=rgb, phase_info=phase_snap, channel_info=ch))
        self.assertTrue(coach.status_snapshot()["inflight"])
        gate.set()
        for _ in range(50):
            if not coach.status_snapshot()["inflight"]:
                break
            time.sleep(0.02)
        verified = coach.recheck_against_current(engine.current_published_state())
        self.assertIsNotNone(verified)
        self.assertEqual(verified.goal, "defend_lane")
        self.assertEqual(verified.focus_row, 3)
        self.assertEqual(calls[0][1], "data:image/png;base64,")

        # Epoch invalidation clears verified advice
        coach.invalidate("game_paused")
        self.assertIsNone(coach.recheck_against_current(engine.current_published_state()))

    def test_5_controller_pause_resume_and_single_collapsible_tk_sidebar(self):
        class _FakeSelector:
            def __init__(self):
                self.fg = True

            def is_lease_alive(self, lease):
                return True, self.fg, "ok" if self.fg else "not_foreground"

            def release(self, lease):
                lease.closed = True
                return True

        class _FakeWgc:
            def __init__(self):
                self.open_count = 0
                self.close_count = 0
                self._open = False
                self._idx = 0

            @property
            def is_open(self):
                return self._open

            def open(self, _lease):
                self._open = True
                self.open_count += 1
                return (1200, 900), (0, 0, 1200, 900, 0, 0, 1200, 900)

            def read_frame(self, timeout=3.0):
                self._idx += 1
                return CapturedFrame(
                    bgra=np.full((900, 1200, 4), 120, dtype=np.uint8),
                    client_size=(1200, 900),
                    rect_signature=(0, 0, 1200, 900, 0, 0, 1200, 900),
                    frame_index=self._idx,
                    frame_qpc_100ns=self._idx * 1000,
                    captured_monotonic=time.monotonic(),
                    stale_dropped=0,
                    alpha_nonopaque=0,
                )

            def close(self):
                if self._open:
                    self._open = False
                    self.close_count += 1
                return True

        class _FakeNorm:
            def invalidate_cache(self):
                pass

            def normalize(self, _bgra, _size, _sig):
                return np.full((600, 800, 3), 80, dtype=np.uint8), None, {"mode": "cached_affine", "elapsed_ms": 5.0}

        sel = _FakeSelector()
        wgc = _FakeWgc()
        engine = ContinuousAdviceEngine(
            clock=time.monotonic, perception=_FakePerception([{"row": 4, "head_x": 400, "heads": 1}]),
            root=REPO_ROOT, threaded_expiry=True,
        )
        ctrl = AdvisorController(
            root=REPO_ROOT, selector=sel, wgc_source=wgc, normalizer=_FakeNorm(), engine=engine,
        )
        self.addCleanup(ctrl.close)
        ctrl.bind_verified_lease(WindowLease(hwnd=1234, pid=999, tid=888, process_handle=777, birth_100ns=12345))
        ok, _ = ctrl.start_monitoring()
        self.assertTrue(ok)

        # Wait until at least 1 frame is processed
        for _ in range(40):
            if ctrl.ui_snapshot()["frames_processed"] >= 1:
                break
            time.sleep(0.05)
        s1 = ctrl.ui_snapshot()
        self.assertEqual(s1["mode_state"], "monitoring")
        self.assertEqual(s1["current_advice_title"], "关注第 4 路防守")

        # Simulate alt-tab away -> auto-withdraws advice and closes WGC while keeping history
        sel.fg = False
        for _ in range(40):
            if ctrl.ui_snapshot()["mode_state"] == "waiting_game" and not wgc.is_open:
                break
            time.sleep(0.05)
        s2 = ctrl.ui_snapshot()
        self.assertEqual(s2["current_advice_title"], "暂无当前建议")
        self.assertEqual(s2["evidence_text"], NO_EVIDENCE_TEXT)
        self.assertGreaterEqual(len(s2["rich_history"]), 1)
        self.assertGreaterEqual(wgc.close_count, 1)

        # Simulate returning to foreground -> auto-reopens WGC and resumes!
        sel.fg = True
        prev_frames = s2["frames_processed"]
        for _ in range(40):
            if ctrl.ui_snapshot()["frames_processed"] > prev_frames:
                break
            time.sleep(0.05)
        s3 = ctrl.ui_snapshot()
        self.assertEqual(s3["mode_state"], "monitoring")
        self.assertGreaterEqual(wgc.open_count, 2)

        # Single right-hand collapsible Tk sidebar verification
        import tkinter as tk
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = tk.Tk()
        root.withdraw()
        app = ReadonlyAdvisorSidebarApp(root, ctrl, settings_store=LocalSettingsStore(Path(tmp_dir.name) / "s.json"))
        try:
            self.assertFalse(any(isinstance(c, tk.Toplevel) for c in root.winfo_children()))
            self.assertEqual(app._active_tab, "advice")
            self.assertTrue(app._llm_enabled_var.get())  # Default LLM mode ON per user preference
            app.render_snapshot(s3)
            self.assertEqual(app._current_title_lbl.cget("text"), "关注第 4 路防守")
            app.show_tab("timeline")
            self.assertEqual(app._active_tab, "timeline")
            app.show_tab("settings")
            self.assertEqual(app._active_tab, "settings")
            app.toggle_collapse()
            self.assertTrue(app._collapsed)
            app.toggle_collapse()
            self.assertFalse(app._collapsed)
            app.show_tab("advice")
        finally:
            app.close()

    def test_6_manual_base_url_and_api_key_in_settings(self):
        sent = []

        def fake_transport(url, body, key, timeout):
            sent.append((url, body, key, timeout))
            if url.endswith("/chat/completions"):
                return {
                    "model": body["model"],
                    "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}],
                }
            return {
                "status": "completed",
                "model": body["model"],
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
            }

        coach = AsyncLlmCoach(clock=time.monotonic, transport=fake_transport, enabled=False)
        self.addCleanup(coach.close)

        # Reject non-HTTPS remote URL and embedded credentials
        ok_bad1, _ = coach.configure_connection(base_url="http://api.example.com/v1", api_key="sk-test-123")
        self.assertFalse(ok_bad1)
        ok_bad2, _ = coach.configure_connection(base_url="https://user:pass@api.example.com/v1", api_key="sk-test-123")
        self.assertFalse(ok_bad2)

        # Accept valid custom HTTPS base_url + in-memory key + chat_completions protocol
        ok_good, msg = coach.configure_connection(
            base_url="https://custom.example.com/v1/",
            api_key="sk-memory-only-secret",
            model="custom-model-v1",
            protocol="chat_completions",
        )
        self.assertTrue(ok_good, msg)
        snap = coach.status_snapshot()
        self.assertEqual(snap["base_url"], "https://custom.example.com/v1")
        self.assertEqual(snap["model"], "custom-model-v1")
        self.assertEqual(snap["protocol"], "chat_completions")
        self.assertEqual(snap["api_key_source"], "manual_memory")
        # Never expose raw secret in status_snapshot or messages
        self.assertNotIn("sk-memory-only-secret", json.dumps({k: v for k, v in snap.items() if k != "advice"}))
        self.assertNotIn("sk-memory-only-secret", msg)

        # Test async connection ping
        done = threading.Event()
        ping_res = []
        coach.test_connection_async(lambda ok, text: (ping_res.append((ok, text)), done.set()))
        self.assertTrue(done.wait(timeout=3.0))
        self.assertTrue(ping_res[0][0], ping_res[0][1])
        self.assertEqual(sent[0][0], "https://custom.example.com/v1/chat/completions")
        self.assertEqual(sent[0][2], "sk-memory-only-secret")

        # Verify UI Settings widgets mask key by default, apply settings, and clear key from memory
        import tkinter as tk
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        ctrl = AdvisorController(root=REPO_ROOT, llm_coach=coach)
        self.addCleanup(ctrl.close)
        root = tk.Tk()
        root.withdraw()
        app = ReadonlyAdvisorSidebarApp(root, ctrl, settings_store=LocalSettingsStore(Path(tmp_dir.name) / "s.json"))
        try:
            self.assertEqual(app._api_key_entry.cget("show"), "*")
            app._show_key_var.set(True)
            app._toggle_key_mask()
            self.assertEqual(app._api_key_entry.cget("show"), "")
            app._base_url_var.set("https://another.example.com/v1")
            app._api_key_var.set("sk-ui-temp")
            app._model_var.set("gpt-6.1-terra")
            app._protocol_var.set("responses")
            ok_ui, _ = app._apply_connection_settings()
            self.assertTrue(ok_ui)
            self.assertEqual(coach.status_snapshot()["base_url"], "https://another.example.com/v1")
            self.assertEqual(coach.status_snapshot()["api_key_source"], "manual_memory")
            app._clear_manual_key()
            self.assertEqual(app._api_key_var.get(), "")
            self.assertEqual(coach._override_api_key, "")
        finally:
            app.close()

    def test_7_auto_fetch_model_list_on_base_url_and_key(self):
        fetched_calls = []

        def fake_models_transport(url, key, timeout):
            fetched_calls.append((url, key, timeout))
            return {
                "object": "list",
                "data": [
                    {"id": "gpt-4o-mini"},
                    {"id": "gpt-6.1-terra"},
                    {"id": "deepseek-chat"},
                    {"id": "gpt-6.1-terra"},  # duplicate should be deduplicated
                ],
            }

        coach = AsyncLlmCoach(clock=time.monotonic, models_transport=fake_models_transport, enabled=False)
        self.addCleanup(coach.close)
        ctrl = AdvisorController(root=REPO_ROOT, llm_coach=coach)
        self.addCleanup(ctrl.close)

        import tkinter as tk
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = tk.Tk()
        root.withdraw()
        app = ReadonlyAdvisorSidebarApp(root, ctrl, settings_store=LocalSettingsStore(Path(tmp_dir.name) / "s.json"))
        try:
            app._model_var.set("")
            app._base_url_var.set("https://models.example.com/v1/")
            app._api_key_var.set("sk-autofetch-secret")
            # Immediate trigger (as on FocusOut / Return)
            triggered = app._run_auto_fetch_if_ready(force=False)
            self.assertTrue(triggered)
            # Duplicate call with same URL+key must be deduplicated
            self.assertFalse(app._run_auto_fetch_if_ready(force=False))

            # Pump Tk mainloop briefly until background thread callback applies values
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                root.update()
                vals = tuple(app._model_combo.cget("values"))
                if len(vals) == 3:
                    break
                time.sleep(0.02)

            self.assertEqual(len(fetched_calls), 1)
            self.assertEqual(fetched_calls[0][0], "https://models.example.com/v1/models")
            self.assertEqual(fetched_calls[0][1], "sk-autofetch-secret")
            self.assertEqual(tuple(app._model_combo.cget("values")), ("deepseek-chat", "gpt-4o-mini", "gpt-6.1-terra"))
            self.assertEqual(app._model_var.get(), "gpt-6.1-terra")
            # Raw key never leaked into feedback label
            self.assertNotIn("sk-autofetch-secret", app._conn_feedback_lbl.cget("text"))
            self.assertIn("3 个可用模型", app._conn_feedback_lbl.cget("text"))
        finally:
            app.close()

    def test_8_stable_advice_dwell_and_zero_jump_fixed_layout(self):
        from pvz_agent.resource_sidebar import Advice
        from pvz_agent.readonly_advisor_v1.async_llm_coach import VerifiedLlmAdvice

        clock = _FakeClock(200.0)
        ctrl = AdvisorController(root=REPO_ROOT, clock=clock, min_dwell_s=3.5, display_stale_s=4.5)
        self.addCleanup(ctrl.close)

        # 1. Advice appears on row 2 at t=200.0
        ch_row2 = {"summary_text": "ev-row2", "leftmost_head_x": 650, "per_lane": {2: [650.0], 5: [660.0]}}
        adv2 = Advice("defend_lane", 2, "s1", "s1", 202.0)
        ctrl._update_display_from_frame(ch_info=ch_row2, advice=adv2, reason_text="第2路防守", rel_time_text="00:01", now=200.0)
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 2 路防守")

        # 2. 1.0s later, row 5 is slightly closer (640 vs 650) but NOT urgent -> holds row 2 stably (no ping-pong!)
        clock.now = 201.0
        ch_row5_far = {"summary_text": "ev-row5-far", "leftmost_head_x": 640, "per_lane": {2: [650.0], 5: [640.0]}}
        adv5 = Advice("defend_lane", 5, "s2", "s2", 203.0)
        ctrl._update_display_from_frame(ch_info=ch_row5_far, advice=adv5, reason_text="第5路防守", rel_time_text="00:02", now=201.0)
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 2 路防守")

        # 3. 1 single transient uncertain frame (advice=None) does NOT flicker/blank the current advice
        clock.now = 201.8
        ctrl._update_display_from_frame(ch_info=ch_row2, advice=None, reason_text="", rel_time_text="00:02", now=201.8)
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 2 路防守")

        # 4. Urgent left-warning (x=175 <= 200) on row 5 overrides IMMEDIATELY (ignores dwell!)
        clock.now = 202.3
        ch_urgent = {"summary_text": "ev-urgent", "leftmost_head_x": 175, "per_lane": {2: [650.0], 5: [175.0]}}
        ctrl._update_display_from_frame(ch_info=ch_urgent, advice=adv5, reason_text="第5路靠左紧急告警", rel_time_text="00:03", now=202.3)
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 5 路防守")

        # 5. Genuine 2 consecutive uncertain frames (or >= display_stale_s) withdraws fail-closed
        clock.now = 203.0
        ctrl._update_display_from_frame(ch_info=ch_urgent, advice=None, reason_text="", rel_time_text="00:04", now=203.0)
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 5 路防守")
        clock.now = 203.8
        ctrl._update_display_from_frame(ch_info=ch_urgent, advice=None, reason_text="", rel_time_text="00:05", now=203.8)
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "暂无当前建议")

        # 6. Verify 0px vertical jump on Advice History box between withdrawn, short local, and long local+LLM states
        import tkinter as tk
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root = tk.Tk()
        root.withdraw()
        app = ReadonlyAdvisorSidebarApp(root, ctrl, settings_store=LocalSettingsStore(Path(tmp_dir.name) / "s.json"))
        try:
            root.update_idletasks()
            base_snap = ctrl.ui_snapshot()
            app.render_snapshot(base_snap)
            root.update_idletasks()
            y_withdrawn = app._hist_box.winfo_y()

            long_snap = dict(
                base_snap,
                has_active_advice=True,
                current_advice_title="关注第 5 路防守",
                current_advice_meta="持续有效 · 已展示 4.2s（最近确认 0.2s 前）",
                current_reason_text="第 5 路观测到 3 个僵尸头，最左 x=175（靠左紧急告警），优先关注该路防线并补足输出。",
                evidence_text="【卡片】阳光=250 | 亮=7, 暗=2, 冷却/未知=1 | 身份一致=8/10, 够买=7 | 可买参考：sunflower(50), peashooter(100)\n【威胁】确认僵尸头=3（2路×1(最左x=650)、5路×2(最左x=175)）| 水路/边缘/车辆不确定=0\n【位置】54格：已占=20, 空=30, 未知=4 | 地面阳光候选=1",
                llm={
                    **base_snap["llm"],
                    "enabled": True,
                    "status": "verified",
                    "advice": VerifiedLlmAdvice(
                        "s1", "s2", "gpt-6.1-terra", "中期转型期", "defend_lane", 5,
                        "第5路压力骤增，建议先补高伤害输出或阻挡植物稳住战线，待威胁降低后再恢复上下对称布局。",
                        ("未核实铁桶剩余血量", "未核实樱桃炸弹冷却"), 9.2, 202.0, 1,
                    ),
                },
            )
            app.render_snapshot(long_snap)
            root.update_idletasks()
            y_full = app._hist_box.winfo_y()

            # Back to withdrawn state
            app.render_snapshot(base_snap)
            root.update_idletasks()
            y_back = app._hist_box.winfo_y()

            self.assertEqual(y_withdrawn, y_full, "Advice history box shifted vertically when advice expanded!")
            self.assertEqual(y_full, y_back, "Advice history box shifted vertically when advice withdrew!")
        finally:
            app.close()

    def test_9_dpapi_remember_and_clear_settings(self):
        # 1. Must reject saving inside the project repo
        with self.assertRaises(ValueError):
            LocalSettingsStore(REPO_ROOT / "bad_secret.json")

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        settings_path = Path(tmp_dir.name) / "advisor_settings.json"
        store = LocalSettingsStore(settings_path)

        net_calls = []

        def forbid_models(url, key, timeout):
            net_calls.append(("models", url))
            return {"data": [{"id": "gpt-6.1-terra"}]}

        def forbid_post(url, body, key, timeout):
            net_calls.append(("post", url))
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}

        import tkinter as tk

        # 2. Session 1: enter Base URL, API key, model, protocol, interval -> Apply -> persists encrypted
        coach1 = AsyncLlmCoach(clock=time.monotonic, transport=forbid_post, models_transport=forbid_models, enabled=False)
        self.addCleanup(coach1.close)
        ctrl1 = AdvisorController(root=REPO_ROOT, llm_coach=coach1)
        self.addCleanup(ctrl1.close)
        root1 = tk.Tk()
        root1.withdraw()
        app1 = ReadonlyAdvisorSidebarApp(root1, ctrl1, settings_store=store)
        try:
            self.assertTrue(app1._remember_var.get())
            app1._base_url_var.set("https://remember.example.com/v1/")
            app1._api_key_var.set("sk-super-secret-dpapi-9999")
            app1._model_var.set("gpt-6.1-sol")
            app1._protocol_var.set("chat_completions")
            app1._llm_interval_var.set("25")
            ok_apply, msg_apply = app1._apply_connection_settings()
            self.assertTrue(ok_apply, msg_apply)
            self.assertNotIn("sk-super-secret-dpapi-9999", msg_apply)
        finally:
            app1.close()

        # Verify file on disk has NO plaintext key, only DPAPI ciphertext
        self.assertTrue(settings_path.exists())
        raw_disk = settings_path.read_text(encoding="utf-8")
        self.assertNotIn("sk-super-secret-dpapi-9999", raw_disk)
        disk_obj = json.loads(raw_disk)
        self.assertEqual(disk_obj["base_url"], "https://remember.example.com/v1")
        self.assertEqual(disk_obj["model"], "gpt-6.1-sol")
        self.assertEqual(disk_obj["protocol"], "chat_completions")
        self.assertEqual(disk_obj["min_interval_s"], 25.0)
        self.assertTrue(len(disk_obj["api_key_dpapi_b64"]) > 20)

        # 3. Session 2: fresh app automatically restores Base URL, API key, model, protocol, interval with 0 network calls
        net_calls.clear()
        coach2 = AsyncLlmCoach(clock=time.monotonic, transport=forbid_post, models_transport=forbid_models, enabled=False)
        self.addCleanup(coach2.close)
        ctrl2 = AdvisorController(root=REPO_ROOT, llm_coach=coach2)
        self.addCleanup(ctrl2.close)
        root2 = tk.Tk()
        root2.withdraw()
        app2 = ReadonlyAdvisorSidebarApp(root2, ctrl2, settings_store=store)
        try:
            root2.update()
            self.assertEqual(app2._base_url_var.get(), "https://remember.example.com/v1")
            self.assertEqual(app2._api_key_var.get(), "sk-super-secret-dpapi-9999")
            self.assertEqual(app2._api_key_entry.cget("show"), "*")
            self.assertEqual(app2._model_var.get(), "gpt-6.1-sol")
            self.assertEqual(app2._protocol_var.get(), "chat_completions")
            self.assertEqual(app2._llm_interval_var.get(), "25")
            self.assertTrue(app2._llm_enabled_var.get())  # Default LLM mode ON per user preference
            self.assertEqual(coach2.status_snapshot()["base_url"], "https://remember.example.com/v1")
            self.assertEqual(coach2.status_snapshot()["api_key_source"], "manual_memory")
            self.assertEqual(len(net_calls), 0, "Startup restore must not make unsolicited network calls")

            # 4. Clear key removes ciphertext from disk AND clears memory
            app2._clear_manual_key()
            self.assertEqual(app2._api_key_var.get(), "")
            self.assertEqual(coach2._override_api_key, "")
            self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8"))["api_key_dpapi_b64"], "")

            # 5. Unchecking remember deletes the file completely
            app2._remember_var.set(False)
            app2._on_remember_toggle()
            self.assertFalse(settings_path.exists())
        finally:
            app2.close()

    def test_10_reviewer_blocking_fixes_regression(self):
        from unittest import mock

        # 1. Blocking #1: Retained verified LLM advice must be re-validated on subsequent frames and withdrawn when lane clears
        class _FakeProvider:
            def complete(self, prompt, *, image_data_url, json_schema):
                sid = prompt.split("source_state_id=")[-1].strip()
                return {
                    "text": json.dumps({
                        "source_state_id": sid,
                        "goal": "defend_lane",
                        "focus_row": 3,
                        "reason": "第3路有僵尸，优先防守。",
                        "uncertainties": [],
                    }, ensure_ascii=False),
                    "model_returned": "gpt-6.1-sol",
                }

        coach = AsyncLlmCoach(clock=time.monotonic, provider_factory=lambda _c: _FakeProvider(), enabled=True, min_interval_s=10.0)
        self.addCleanup(coach.close)
        fake_perc = _FakePerception([
            {"row": 3, "head_x": 450, "heads": 1},  # Frame 1: threat on row 3
            {"row": 3, "head_x": 440, "heads": 1},  # Frame 2: same row 3 -> retained still valid
            {"row": 5, "head_x": 180, "heads": 1},  # Frame 3: row 3 cleared & urgent row 5 -> retained MUST withdraw!
        ])
        engine = ContinuousAdviceEngine(clock=time.monotonic, perception=fake_perc, root=REPO_ROOT, threaded_expiry=False)
        self.addCleanup(engine.close)
        rgb = bytes([12]) * RGB_BYTES
        pub1, ch1, _ = engine.ingest_canonical_rgb(rgb, (1200, 900))
        self.assertTrue(coach.maybe_dispatch(state=pub1, rgb_bytes=rgb, phase_info=PhaseTracker().snapshot(time.monotonic()), channel_info=ch1))
        for _ in range(50):
            if not coach.status_snapshot()["inflight"]:
                break
            time.sleep(0.02)
        self.assertIsNotNone(coach.recheck_against_current(engine.current_published_state()))

        # Frame 2: row 3 still threatened -> retained advice still fresh (possibly_stale=False)
        engine.ingest_canonical_rgb(rgb, (1200, 900))
        ret2 = coach.recheck_against_current(engine.current_published_state())
        self.assertIsNotNone(ret2)
        self.assertFalse(ret2.possibly_stale)

        # Frame 3: row 3 cleared -> retained advice stays visible per user preference, tagged possibly_stale=True
        engine.ingest_canonical_rgb(rgb, (1200, 900))
        ret3 = coach.recheck_against_current(engine.current_published_state())
        self.assertIsNotNone(ret3)
        self.assertTrue(ret3.possibly_stale)
        self.assertEqual(coach.status_snapshot()["status"], "verified_stale")
        # Pause / epoch invalidation still withdraws cleanly
        coach.invalidate("game_paused")
        self.assertIsNone(coach.status_snapshot()["advice"])

        # 2. Blocking #2: PhaseTracker concurrent start_new_run / update / snapshot / elapsed_text must be thread-safe
        tracker = PhaseTracker()
        errors = []
        stop = threading.Event()
        sample_ch = {"cells_occupied": 10, "cells_unknown": 2, "zombie_heads": 2, "leftmost_head_x": 190}

        def _writer():
            try:
                for i in range(150):
                    tracker.update(sample_ch, time.monotonic(), advice_added=True)
                    tracker.record_gap_event(time.monotonic(), "test")
                    if i % 10 == 0:
                        tracker.start_new_run(time.monotonic(), reason="concurrent")
            except Exception as exc:
                errors.append(exc)
            finally:
                stop.set()

        def _reader():
            try:
                while not stop.is_set():
                    snap = tracker.snapshot(time.monotonic())
                    _ = tracker.elapsed_text(time.monotonic())
                    self.assertIn("segments", snap)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_writer)
        t2 = threading.Thread(target=_reader)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        self.assertEqual(errors, [], f"PhaseTracker thread-safety errors: {errors}")

        # 3. Blocking #3 + startup no-write: In-flight GET /models must NEVER resurrect a cleared key
        gate = threading.Event()
        entered = threading.Event()

        def slow_models_transport(url, key, timeout):
            entered.set()
            gate.wait(timeout=3.0)
            return {"data": [{"id": "gpt-6.1-terra"}]}

        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        settings_path = Path(tmp_dir.name) / "no_resurrect.json"
        store = LocalSettingsStore(settings_path)
        coach_m = AsyncLlmCoach(clock=time.monotonic, models_transport=slow_models_transport, enabled=False)
        self.addCleanup(coach_m.close)
        ctrl_m = AdvisorController(root=REPO_ROOT, llm_coach=coach_m)
        self.addCleanup(ctrl_m.close)

        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        app = ReadonlyAdvisorSidebarApp(root, ctrl_m, settings_store=store)
        try:
            # Startup must NOT silently create the settings file
            self.assertFalse(settings_path.exists())

            app._base_url_var.set("https://race.example.com/v1")
            app._api_key_var.set("sk-must-not-resurrect")
            self.assertTrue(app._run_auto_fetch_if_ready(force=False))
            self.assertTrue(entered.wait(timeout=2.0))

            # User clicks Clear Key while GET /models is still in flight
            app._clear_manual_key()
            gate.set()

            # Pump Tk mainloop so the queued callback drains
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                root.update()
                time.sleep(0.02)

            self.assertEqual(app._api_key_var.get(), "")
            self.assertEqual(coach_m._override_api_key, "")
            self.assertFalse(settings_path.exists())
        finally:
            app.close()

        # 4. DPAPI protect failure must fail closed and never write plaintext
        with mock.patch("pvz_agent.readonly_advisor_v1.local_settings.dpapi_protect", side_effect=RuntimeError("boom")):
            ok_dp, msg_dp = store.save(
                base_url="https://fail.example.com/v1",
                api_key="sk-never-plaintext",
                model="gpt-6.1-sol",
                protocol="responses",
                min_interval_s=18.0,
                remember_enabled=True,
            )
            self.assertFalse(ok_dp)
            self.assertIn("拒绝明文保存", msg_dp)
            self.assertFalse(settings_path.exists())

    def test_11_speed_optimizations_text_jpeg_reasoning_and_macro_recheck(self):
        from pvz_agent.llm_provider import ProviderError
        from pvz_agent.readonly_advisor_v1.async_llm_coach import (
            encode_rgb_jpeg_data_url,
            encode_rgb_png_data_url,
        )

        # 1. JPEG 512x384 payload must be dramatically smaller than 800x600 PNG on real frame bytes
        ref_bgr = cv2.imread(str(REAL_FRAMES[0]), cv2.IMREAD_COLOR)
        ref_rgb = cv2.cvtColor(cv2.resize(ref_bgr, (800, 600)), cv2.COLOR_BGR2RGB).tobytes()
        png_url = encode_rgb_png_data_url(ref_rgb)
        jpg_url = encode_rgb_jpeg_data_url(ref_rgb)
        self.assertTrue(jpg_url.startswith("data:image/jpeg;base64,"))
        self.assertLess(len(jpg_url), len(png_url) * 0.25, "Downscaled JPEG should be <25% of PNG size")

        # 2. Default text_only sends NO image + reasoning.effort='low'; low_jpeg sends detail='low' JPEG;
        #    HTTP 400 on reasoning triggers automatic 1-time fallback retry without reasoning.
        sent_bodies = []
        fail_reasoning_once = {"flag": True}

        def fake_transport(url, body, key, timeout):
            sent_bodies.append((url, body))
            if fail_reasoning_once["flag"] and ("reasoning" in body or "reasoning_effort" in body):
                fail_reasoning_once["flag"] = False
                raise ProviderError("provider_http_error", 400)
            # Extract state_id from prompt
            if "input" in body:
                txt = body["input"][0]["content"][0]["text"]
            else:
                txt = body["messages"][0]["content"][0]["text"]
            sid = txt.split("source_state_id=")[-1].strip()
            payload = json.dumps({
                "source_state_id": sid,
                "goal": "develop_economy",
                "focus_row": None,
                "reason": "前排稳固，继续补向日葵提升阳光产速。",
                "uncertainties": [],
            }, ensure_ascii=False)
            if url.endswith("/chat/completions"):
                return {"model": body["model"], "choices": [{"finish_reason": "stop", "message": {"content": payload}}]}
            return {"status": "completed", "model": body["model"], "output": [{"type": "message", "content": [{"type": "output_text", "text": payload}]}]}

        coach = AsyncLlmCoach(clock=time.monotonic, transport=fake_transport, enabled=True, min_interval_s=10.0)
        self.addCleanup(coach.close)
        ok_cfg, _ = coach.configure_connection(
            base_url="https://fast.example.com/v1",
            api_key="sk-fast-test",
            model="gpt-6.2-sol",
            protocol="responses",
            vision_mode="text_only",
            reasoning_effort="low",
        )
        self.assertTrue(ok_cfg)

        # Frame 1: calm board (no zombies), Frame 2: single far-right spawn (x=720), Frame 3: urgent left threat (x=180)
        fake_perc = _FakePerception([
            {"row": 1, "head_x": 750, "heads": 0, "sun": 100},
            {"row": 2, "head_x": 720, "heads": 1, "sun": 100},
            {"row": 2, "head_x": 180, "heads": 1, "sun": 100},
        ])
        engine = ContinuousAdviceEngine(clock=time.monotonic, perception=fake_perc, root=REPO_ROOT, threaded_expiry=False)
        self.addCleanup(engine.close)
        pub1, ch1, _ = engine.ingest_canonical_rgb(ref_rgb, (1200, 900))
        self.assertTrue(coach.maybe_dispatch(state=pub1, rgb_bytes=ref_rgb, phase_info=PhaseTracker().snapshot(time.monotonic()), channel_info=ch1))
        for _ in range(60):
            if not coach.status_snapshot()["inflight"]:
                break
            time.sleep(0.02)

        # Verify: first attempt sent reasoning={"effort":"low"} with text_only (1 content item, no image),
        # got HTTP 400, and auto-retried cleanly without reasoning!
        self.assertEqual(len(sent_bodies), 2)
        self.assertEqual(sent_bodies[0][1].get("reasoning"), {"effort": "low"})
        self.assertEqual(len(sent_bodies[0][1]["input"][0]["content"]), 1)
        self.assertNotIn("reasoning", sent_bodies[1][1])
        self.assertIsNotNone(coach.recheck_against_current(engine.current_published_state()))

        # Frame 2 (routine far-right zombie at x=720): macro develop_economy advice SURVIVES fresh
        engine.ingest_canonical_rgb(ref_rgb, (1200, 900))
        m2 = coach.recheck_against_current(engine.current_published_state())
        self.assertIsNotNone(m2)
        self.assertFalse(m2.possibly_stale)

        # Frame 3 (urgent left threat x=180 <= 200): stays visible but tagged possibly_stale=True
        engine.ingest_canonical_rgb(ref_rgb, (1200, 900))
        m3 = coach.recheck_against_current(engine.current_published_state())
        self.assertIsNotNone(m3)
        self.assertTrue(m3.possibly_stale)

        # 3. Verify low_jpeg mode sends JPEG with detail='low' on chat_completions + reasoning_effort='low'
        sent_bodies.clear()
        fail_reasoning_once["flag"] = False
        coach.configure_connection(
            base_url="https://fast.example.com/v1",
            api_key="sk-fast-test",
            model="gpt-6.2-sol",
            protocol="chat_completions",
            vision_mode="low_jpeg",
            reasoning_effort="low",
        )
        coach._next_allowed_at = 0.0
        pub_now = engine.current_published_state()
        self.assertTrue(coach.maybe_dispatch(state=pub_now, rgb_bytes=ref_rgb, phase_info=PhaseTracker().snapshot(time.monotonic()), channel_info=ch1))
        for _ in range(60):
            if not coach.status_snapshot()["inflight"]:
                break
            time.sleep(0.02)
        self.assertEqual(len(sent_bodies), 1)
        body_cc = sent_bodies[0][1]
        self.assertEqual(body_cc.get("reasoning_effort"), "low")
        img_part = body_cc["messages"][0]["content"][1]
        self.assertEqual(img_part["image_url"]["detail"], "low")
        self.assertTrue(img_part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

        # 4. Verify UI + DPAPI LocalSettingsStore persist & restore vision_mode + reasoning_effort
        import tkinter as tk
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        store = LocalSettingsStore(Path(tmp_dir.name) / "speed_prefs.json")
        ctrl1 = AdvisorController(root=REPO_ROOT, llm_coach=coach)
        self.addCleanup(ctrl1.close)
        r1 = tk.Tk()
        r1.withdraw()
        app1 = ReadonlyAdvisorSidebarApp(r1, ctrl1, settings_store=store)
        try:
            app1._base_url_var.set("https://fast.example.com/v1")
            app1._api_key_var.set("sk-speed-persist")
            app1._model_var.set("gpt-6.2-sol")
            app1._vision_mode_var.set("轻量看图(512px·low)")
            app1._reasoning_effort_var.set("auto(模型默认)")
            ok_u, _ = app1._apply_connection_settings()
            self.assertTrue(ok_u)
        finally:
            app1.close()

        coach2 = AsyncLlmCoach(clock=time.monotonic, transport=fake_transport, enabled=False)
        self.addCleanup(coach2.close)
        ctrl2 = AdvisorController(root=REPO_ROOT, llm_coach=coach2)
        self.addCleanup(ctrl2.close)
        r2 = tk.Tk()
        r2.withdraw()
        app2 = ReadonlyAdvisorSidebarApp(r2, ctrl2, settings_store=store)
        try:
            self.assertEqual(app2._selected_vision_mode(), "low_jpeg")
            self.assertEqual(app2._selected_reasoning_effort(), "auto")
            self.assertEqual(coach2.status_snapshot()["vision_mode"], "low_jpeg")
            self.assertEqual(coach2.status_snapshot()["reasoning_effort"], "auto")
        finally:
            app2.close()

    def test_12_llm_only_display_and_possibly_stale_tag(self):
        from pvz_agent.resource_sidebar import Advice
        from pvz_agent.readonly_advisor_v1.async_llm_coach import VerifiedLlmAdvice

        clock = _FakeClock(300.0)
        coach = AsyncLlmCoach(clock=clock, enabled=True, min_interval_s=10.0)
        self.addCleanup(coach.close)
        ctrl = AdvisorController(root=REPO_ROOT, clock=clock, llm_coach=coach)
        self.addCleanup(ctrl.close)

        # Feed a local frame while LLM is enabled -> local advice MUST NOT show or enter rich_history!
        ch = {"summary_text": "ev-live", "leftmost_head_x": 450, "per_lane": {2: [450.0]}}
        adv_local = Advice("defend_lane", 2, "s1", "s1", 302.0)
        ctrl._update_display_from_frame(ch_info=ch, advice=adv_local, reason_text="本地第2路防守", rel_time_text="00:05", now=300.0)
        snap_wait = ctrl.ui_snapshot()
        self.assertNotIn("本地即时", " ".join(snap_wait["rich_history"]))
        self.assertNotEqual(snap_wait["current_advice_title"], "关注第 2 路防守")
        self.assertEqual(snap_wait["current_advice_title"], "暂无 LLM 建议")

        # Inject a possibly_stale LLM advice -> stays displayed with （可能过时） right after ]
        with coach._lock:
            coach._verified_advice = VerifiedLlmAdvice(
                source_state_id="s1",
                reviewed_state_id="s2",
                model="gpt-6.2-sol",
                phase="中期转型期",
                goal="defend_lane",
                focus_row=4,
                reason="水路4路补双发射手稳住输出。",
                uncertainties=(),
                latency_s=3.8,
                verified_at=300.0,
                epoch=coach._epoch,
                possibly_stale=True,
                stale_reason="focus_lane_cleared",
            )
            coach._status = "verified_stale"
        with ctrl._lock:
            ctrl._llm_history.append("[00:05] [gpt-6.2-sol · 3.8s]（可能过时） 关注第 4 路防守 — 水路4路补双发射手稳住输出。")

        snap_llm = ctrl.ui_snapshot()
        self.assertTrue(snap_llm["has_active_advice"])
        self.assertEqual(snap_llm["current_advice_title"], "关注第 4 路防守")
        self.assertIn("[gpt-6.2-sol · 3.8s]（可能过时）", snap_llm["current_advice_meta"])
        self.assertIn("水路4路补双发射手", snap_llm["current_reason_text"])
        self.assertEqual(len(snap_llm["rich_history"]), 1)
        self.assertIn("[gpt-6.2-sol · 3.8s]（可能过时）", snap_llm["rich_history"][0])
        self.assertNotIn("本地即时", snap_llm["rich_history"][0])

        # Turning LLM OFF cleanly restores pure-local mode
        coach.set_enabled(False)
        snap_local = ctrl.ui_snapshot()
        self.assertEqual(snap_local["current_advice_title"], "关注第 2 路防守")


if __name__ == "__main__":
    unittest.main(verbosity=2)
