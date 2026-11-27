"""Comprehensive offline test suite for PVZagent Read-Only Real-Time Advice Assistant v1.0."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pvz_agent.goal_review import ReviewBlocked
from pvz_agent.llm_provider import ProviderError
from pvz_agent.product_startup import verify_product_startup
from pvz_agent.readonly_advisor_v1.advisor_runtime import (
    AdvisorController,
    ContinuousAdviceEngine,
)
from pvz_agent.readonly_advisor_v1 import async_llm_coach as llm
from pvz_agent.readonly_advisor_v1.async_llm_coach import AsyncLlmCoach
from pvz_agent.readonly_advisor_v1.fast_perception import (
    NO_EVIDENCE_TEXT,
    RGB_BYTES,
    FastGeometryNormalizer,
    FastSinglePassPerception,
    PhaseTracker,
    evaluate_local_strategy,
    evaluate_three_channels,
)
from pvz_agent.readonly_advisor_v1.local_settings import LocalSettingsStore
from pvz_agent.readonly_advisor_v1.sidebar_app import ReadonlyAdvisorSidebarApp
from pvz_agent.readonly_advisor_v1.zombie_intel import (
    PLANT_CATALOG,
    ZOMBIE_SPECIES_DB,
    ZombieSpeciesClassifier,
    classify_state_metadata,
    extended_species_prompt_hint,
    get_zombie_profile,
    recommend_counter_plan,
)
from pvz_agent.readonly_advisor_v1.wgc_capture import (
    ALLOWED_EXE_NAMES,
    ALPHA_DIAG_FIELDS,
    CapturedFrame,
    DLL_PATH,
    WindowLease,
    matches_pvz_exe_path,
    perimeter_alpha_ok,
)
from pvz_agent.strategy_review import validate_strategy

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


def _completion_payload(protocol, text):
    if protocol == "responses":
        return {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ]}
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


def _http_response(body, *, status=200, content_type="application/json"):
    if not isinstance(body, bytes):
        body = json.dumps(body, ensure_ascii=False).encode()
    headers = (
        f"HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    response = http.client.HTTPResponse(mock.Mock(makefile=mock.Mock(return_value=io.BytesIO(headers + body))))
    response.begin()
    return response


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
            "vehicles": {
                "candidates": spec.get("vehicles", []),
                "tracks": [],
                "uncertain_candidates": spec.get("uncertain_vehicles", []),
            },
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
                    {
                        "slot": s, "status": "observed",
                        "plant_id": "spikeweed" if spec.get("spikeweed") and s == 1 else "sunflower",
                        "display_cost": 100 if spec.get("spikeweed") and s == 1 else 50,
                    }
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
            payload = json.dumps({
                "goal": "develop_economy",
                "focus_row": None,
                "reason": "前排稳固，继续补向日葵提升阳光产速。",
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
        self.assertEqual([message["role"] for message in body_cc["messages"]], ["system", "user"])
        img_part = body_cc["messages"][1]["content"][1]
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


    def test_13_zombie_encyclopedia_and_counter_plant_planner(self):
        core = {
            "zomboni": 1350, "buckethead_zombie": 1370, "snorkel_zombie": 270,
            "conehead_zombie": 640, "flag_zombie": 270, "basic_zombie": 270,
        }
        extended = {
            "pole_vaulter": 500, "football_zombie": 1670, "screen_door_zombie": 1370,
            "dolphin_rider": 500, "newspaper_zombie": 420, "gargantuar": 3000, "bobsled_team": 270 * 4,
        }
        self.assertEqual(set(ZOMBIE_SPECIES_DB), set(core) | set(extended))
        self.assertEqual(len(PLANT_CATALOG), 10)
        for plant in PLANT_CATALOG.values():
            self.assertTrue(plant["name_cn"] and plant["cost"] > 0 and plant["role"])
        prompt = extended_species_prompt_hint()
        self.assertNotIn("\n", prompt)
        for species, hp in (core | extended).items():
            with self.subTest(species=species):
                profile = get_zombie_profile(species)
                self.assertEqual(profile.species_id, species)
                self.assertEqual(profile.cv_supported, species in core)
                self.assertEqual(profile.hp_total, hp)
                self.assertEqual(profile.hp_body + profile.hp_armor, hp)
                self.assertTrue(profile.name_cn and profile.traits and profile.counter_hint)
                counters = profile.land_counters + profile.water_counters
                self.assertTrue(counters)
                self.assertLessEqual(set(counters), set(PLANT_CATALOG))
                self.assertEqual(profile.name_cn in prompt, species in extended)
        self.assertIs(get_zombie_profile("unrecognized"), ZOMBIE_SPECIES_DB["basic_zombie"])

        def plan_for(species, row, sun, observations, use_visuals=True):
            intel = classify_state_metadata({"zombies": {"candidates": [
                {"row": row, "head_xy": (600, 200), "species_hint": species, "helmet_support": {"area": 600}},
            ]}})
            slots = [
                {"slot": i, "status": "observed", "plant_id": plant, "printed_cost": PLANT_CATALOG[plant]["cost"]}
                for i, (plant, _) in enumerate(observations, 1)
            ]
            visuals = [
                {"slot": i, "visual_state": visual} for i, (_, visual) in enumerate(observations, 1)
            ]
            return recommend_counter_plan(intel, slots, sun, raw_cards=visuals if use_visuals else None)

        for species, row, sun, cards, plants, text, power in (
            ("zomboni", 1, 100, [("spikeweed", "bright")], ["spikeweed"], "放地刺(100)爆胎秒杀", "anti_vehicle"),
            ("zomboni", 1, 50, [("spikeweed", "bright"), ("squash", "bright")], ["squash"],
             "放窝瓜(50)紧急单解冰车", "anti_vehicle"),
            ("zomboni", 1, 50, [("spikeweed", "bright")], ["spikeweed"], "攒阳到100放地刺爆胎", "anti_vehicle"),
            ("buckethead_zombie", 3, 25, [("tangle_kelp", "bright")], ["tangle_kelp"],
             "水路放缠绕海草(25)秒杀", "heavy_burst"),
            ("buckethead_zombie", 2, 50, [("squash", "bright")], ["squash"], "放窝瓜(50)单杀高甲", "heavy_burst"),
            ("buckethead_zombie", 2, 250, [("squash", "dim"), ("wallnut", "bright"), ("repeater", "bright")],
             ["wallnut", "repeater"], "坚果(50)挡住+补双发射手(200)集火", "heavy_burst"),
            ("buckethead_zombie", 2, 200, [("wallnut", "bright"), ("repeater", "bright")],
             ["wallnut"], "放坚果(50)拖住铁桶", "heavy_burst"),
            ("snorkel_zombie", 4, 25, [("tangle_kelp", "bright")], ["tangle_kelp"],
             "水路放缠绕海草(25)秒杀", "water_counter"),
            ("conehead_zombie", 2, 200, [("repeater", "bright")], ["repeater"], "压制路障", "medium_dps"),
            ("basic_zombie", 5, 100, [("peashooter", "bright")], ["peashooter"], "豌豆射手", "standard_dps"),
        ):
            with self.subTest(species=species, row=row, sun=sun, cards=cards):
                plan = plan_for(species, row, sun, cards)
                self.assertEqual(plan["focus_row"], row)
                self.assertEqual(plan["target_species_id"], species)
                self.assertEqual(plan["target_hp_est"], core[species])
                self.assertEqual(plan["firepower_level"], power)
                self.assertEqual(plan["recommended_plants"], plants)
                self.assertIn(text, plan["action_text"])
                self.assertLessEqual(len(plan["action_text"]), 46)
                self.assertIn(plan["action_text"], plan["llm_counter_line"])
                if ("squash", "dim") in cards:
                    self.assertNotIn("squash", plan["ready_counters"])
        self.assertEqual(plan_for("zomboni", 1, 50, [("spikeweed", "bright")])["ready_counters"], [])
        for sun, visual in ((None, "bright"), (100, "unknown"), (100, "dim")):
            self.assertEqual(plan_for("zomboni", 1, sun, [("spikeweed", visual)])["ready_counters"], [])
        self.assertEqual(plan_for("zomboni", 1, 100, [("spikeweed", "unknown")], False)["ready_counters"], ["spikeweed"])
        self.assertEqual(plan_for("buckethead_zombie", 2, 25, [("tangle_kelp", "bright")])["recommended_plants"], [])
        empty_plan = recommend_counter_plan(classify_state_metadata({}), [], None)
        self.assertEqual((empty_plan["focus_row"], empty_plan["action_text"], empty_plan["firepower_level"]), (None, "", "none"))

        basic = {"row": 2, "head_xy": (680, 180)}
        bucket = {"row": 3, "head_xy": (700, 285), "hat_hypothesis": "bucket", "helmet_support": {"area": 560}}
        truck = {"row": 1, "front_xy": (704, 123)}
        state = {"zombies": {"candidates": [basic, bucket]}, "vehicles": {"candidates": [truck]}}
        intel = classify_state_metadata(state)
        self.assertEqual(intel["priority_row"], 1)
        self.assertEqual(intel["species_counts"], {"zomboni": 1, "basic_zombie": 1, "buckethead_zombie": 1})
        lane = intel["by_lane"][1]
        self.assertEqual(lane["lane_urgency"], 6.972)
        self.assertEqual((lane["count"], lane["leftmost_x"], lane["has_vehicle"]), (1, 704.0, True))
        state["vehicles"]["candidates"] = []
        self.assertEqual(classify_state_metadata(state)["priority_row"], 3)
        basic["head_xy"] = (150, 180)
        self.assertEqual(classify_state_metadata(state)["priority_row"], 2)
        self.assertEqual(classify_state_metadata(state, left_warning_x=100)["priority_row"], 3)

    def test_14_real_frame_zombie_species_classification(self):
        from unittest import mock

        perception = FastSinglePassPerception(REPO_ROOT)
        classifier = perception.zombie_classifier
        if _FALLBACK_REF_FRAME.is_file():
            with Image.open(_FALLBACK_REF_FRAME) as frame:
                reference = np.array(frame.convert("RGB"))
            np.testing.assert_array_equal(classifier._ref_rgb, reference)
            self.assertEqual(classifier._ref_rgb.dtype, np.uint8)
        else:
            reference = np.zeros((600, 800, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            no_reference = ZombieSpeciesClassifier(Path(root))
            self.assertIsNone(no_reference._ref_rgb)
            self.assertEqual(no_reference.classify_frame(None, {})["summary_short"], "无确认威胁")

        def synthetic_head(rgb, row, point, **metadata):
            state = {"zombies": {"candidates": [{"row": row, "head_xy": point, **metadata}]}}
            return classifier.classify_frame(rgb, state)["by_lane"][row]["primary_entity"]

        rgb = reference.copy()
        rgb[180:192, 351:372] = (190, 12, 25)
        self.assertEqual(synthetic_head(rgb, 2, (400, 200))["species_id"], "flag_zombie")
        self.assertEqual(synthetic_head(rgb, 3, (400, 200))["species_id"], "basic_zombie")
        self.assertEqual(synthetic_head(rgb, 2, (400, 200), species_hint="basic_zombie")["species_id"], "basic_zombie")
        rgb = reference.copy()
        rgb[60:79, 351:372] = (190, 12, 25)
        self.assertEqual(synthetic_head(rgb, 1, (400, 82))["species_id"], "basic_zombie")
        self.assertEqual(synthetic_head(reference, 2, (400, 200))["species_id"], "basic_zombie")

        for width, height, color, species, armor, hp in (
            (31, 21, (210, 209, 211), "buckethead_zombie", "满甲", 1370),
            (20, 18, (210, 209, 211), "buckethead_zombie", "破损", 820),
            (22, 20, (230, 136, 0), "conehead_zombie", "满甲", 640),
            (20, 10, (230, 136, 0), "conehead_zombie", "受损", 440),
        ):
            with self.subTest(species=species, armor=armor):
                rgb = reference.copy()
                rgb[156:156 + height, 384:384 + width] = color
                entity = synthetic_head(rgb, 2, (400, 200))
                self.assertEqual((entity["species_id"], entity["armor_stage"], entity["hp_est"]), (species, armor, hp))
        eater = synthetic_head(None, 4, (690, 390), water_pose="water_eating", hat_hypothesis="bucket")
        self.assertEqual((eater["species_id"], eater["armor_stage"], eater["hp_est"]), ("snorkel_zombie", "露头啃咬", 270))
        self.assertEqual(eater["traits"], ("已露头可直伤", "正在啃咬植物"))

        state = _FakePerception([{"heads": 0, "sun": 100}]).observe(
            None, run_id="synthetic", state_id="s", capture_time="t",
            frame_time_seconds=0.0, evidence={}, tracking_epoch=0,
        )
        no_threat = evaluate_three_channels(REPO_ROOT, state)
        self.assertIn("确认威胁=0（无确认僵尸/载具）", no_threat["summary_text"])
        self.assertIsNone(no_threat["leftmost_threat_x"])
        self.assertEqual(evaluate_local_strategy(state)["strategy"]["goal"], "develop_economy")
        state["vehicles"]["candidates"] = [{"row": 1, "front_xy": [704, 125]}]
        state["loadout"]["slots"] = [{"slot": 1, "status": "observed", "plant_id": "spikeweed", "display_cost": 100}]
        channels = evaluate_three_channels(REPO_ROOT, state)
        self.assertEqual((channels["zombie_heads"], channels["confirmed_vehicles"], channels["uncertain_threats"]), (0, 1, 0))
        self.assertEqual((channels["priority_row"], channels["leftmost_threat_x"]), (1, 704))
        self.assertIn("放地刺(100)爆胎秒杀", channels["summary_text"])
        strategy = validate_strategy(evaluate_local_strategy(state)["strategy"], "s")
        self.assertEqual(strategy["focus_row"], 1)
        self.assertIn("第1路冰车僵尸(约1350血·碾压,x=704)：放地刺", strategy["reason"])
        for section, key in (
            ("zombies", "uncertain_edge_presence"), ("zombies", "uncertain_water_candidates"),
            ("vehicles", "uncertain_candidates"),
        ):
            with self.subTest(channel=key), mock.patch.dict(state[section], {key: [{}]}):
                with self.assertRaises(ReviewBlocked):
                    evaluate_local_strategy(state)
        with mock.patch.dict(state["vehicles"], {"candidates": [{"row": None, "front_xy": [704, 125]}]}):
            self.assertEqual(evaluate_three_channels(REPO_ROOT, state)["uncertain_threats"], 1)
            with self.assertRaises(ReviewBlocked):
                evaluate_local_strategy(state)
        with mock.patch.dict(state["scene"], {"value": "unknown"}):
            with self.assertRaises(ReviewBlocked):
                evaluate_local_strategy(state)

        rgb = reference.copy()
        rgb[180:192, 351:372] = (190, 12, 25)
        state["zombies"]["candidates"] = [{"row": 2, "head_xy": [400, 200]}]
        with (
            Image.fromarray(rgb) as image,
            mock.patch.object(perception.base, "observe", return_value=state),
            mock.patch.object(perception, "_observe_cards_in_memory", return_value=state["loadout"]),
            mock.patch.object(Image.Image, "save", side_effect=AssertionError("frame write")),
            mock.patch.object(cv2, "imwrite", side_effect=AssertionError("frame write")),
        ):
            elapsed_ms = [perception.observe(image)["processing_ms"] for _ in range(11)]
        self.assertLess(median(elapsed_ms), 40.0, elapsed_ms)
        self.assertEqual(state["zombie_intel"]["species_counts"], {"zomboni": 1, "flag_zombie": 1})
        self.assertEqual(evaluate_three_channels(REPO_ROOT, state)["leftmost_threat_x"], 400)

        real_cases = [
            (REPO_ROOT / "data/day06/crossgame-v1/a/images/060.000.png", {2: ("basic_zombie", 270)}),
            (REPO_ROOT / "data/day06/vehicle-v1/a-448.png",
             {1: ("zomboni", 1350), 3: ("conehead_zombie", 640), 4: ("conehead_zombie", 640)}),
            (REPO_ROOT / "data/day06/water-v1/a-291.5.png", {1: ("conehead_zombie", 640), 4: ("snorkel_zombie", 270)}),
            (REPO_ROOT / "data/day06/water-v1/a-293.25.png", {1: ("conehead_zombie", 640), 4: ("snorkel_zombie", 270)}),
            (REPO_ROOT / "evidence/day06/goal-live-v1/live-01/captures/longrun-2026-10-24T12-08-03-641390+00-00-7423c8.png",
             {3: ("buckethead_zombie", 1370)}),
            (_CANDIDATE_REAL_FRAMES[0],
             {2: ("basic_zombie", 270), 3: ("buckethead_zombie", 1370), 4: ("conehead_zombie", 440), 6: ("basic_zombie", 270)}),
        ]
        for index, (path, expected) in enumerate(real_cases):
            with self.subTest(frame=path.name):
                if not path.is_file():
                    self.skipTest("optional replay frame is absent")
                with Image.open(path) as frame, frame.convert("RGB") as image:
                    observed = perception.observe(
                        image, run_id=f"species-replay-{index}", state_id=f"s{index}", capture_time=f"replay-{index}",
                        frame_time_seconds=0.0, tracking_epoch=index,
                        evidence={"image": path.relative_to(REPO_ROOT).as_posix(),
                                  "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                    )
                lanes = observed["zombie_intel"]["by_lane"]
                for row, (species, hp) in expected.items():
                    entity = lanes[row]["primary_entity"]
                    self.assertEqual((entity["species_id"], entity["hp_est"]), (species, hp))
                    if species == "buckethead_zombie":
                        self.assertEqual(entity["armor_stage"], "满甲")
                    if species == "snorkel_zombie":
                        expected_stance = "diving" if path.name == "a-291.5.png" else "submerged"
                        self.assertEqual(entity["stance"], expected_stance)
                if 1 in expected and expected[1][0] == "zomboni":
                    strategy = validate_strategy(evaluate_local_strategy(observed)["strategy"], observed["state_id"])
                    self.assertEqual(observed["zombie_intel"]["priority_row"], 1)
                    self.assertEqual(observed["lane_defense"]["by_row"][1]["verdict"], "unknown")
                    self.assertEqual(observed["lane_defense"]["by_row"][3]["verdict"], "tight")
                    self.assertEqual(strategy["focus_row"], 3)
                    self.assertIn("路障僵尸", strategy["reason"])
        self.assertEqual(verify_product_startup(REPO_ROOT)["artifacts_checked"], 59)


    def test_15_vehicle_lane_runtime_publication_and_watchdog_expiry(self):
        from unittest import mock

        from pvz_agent.resource_sidebar import Advice

        clock = _FakeClock()
        vehicle = {"heads": 0, "vehicles": [{"row": 1, "front_xy": [704, 125]}], "spikeweed": True}
        perception = _FakePerception([
            vehicle, vehicle, dict(vehicle, uncertain_vehicles=[{}]), vehicle,
            dict(vehicle, scene="unknown"),
        ])
        engine = ContinuousAdviceEngine(clock=clock, perception=perception, threaded_expiry=False)
        ctrl = AdvisorController(clock=clock, engine=engine)
        self.addCleanup(ctrl.close)
        rgb = bytes(RGB_BYTES)
        size = (1200, 900)

        ctrl._update_display_from_frame(
            ch_info={"summary_text": "", "per_lane": {2: [650]}},
            advice=Advice("defend_lane", 2, "initial", "initial", clock() + 2.0),
            reason_text="", rel_time_text="00:00", now=clock(),
        )
        clock.now += 0.25
        first, channels, advice = engine.ingest_canonical_rgb(rgb, size)
        self.assertIsNotNone(advice)
        self.assertEqual((advice.goal, advice.focus_row), ("defend_lane", 1))
        self.assertEqual(engine.sidebar.read().current, advice)
        self.assertIn("冰车僵尸", engine.sidebar.current_reason_text)
        self.assertIn("地刺", engine.sidebar.current_reason_text)
        # Keep the old lane present to isolate the urgent switch from the lane-cleared override.
        ctrl._update_display_from_frame(
            ch_info=dict(channels, per_lane={2: [650]}), advice=advice,
            reason_text=engine.sidebar.current_reason_text, rel_time_text="00:01", now=clock(),
        )
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 1 路防守")
        # Isolate lane membership from the urgency flag to exercise the dwell rule.
        ctrl._update_display_from_frame(
            ch_info=dict(channels, confirmed_vehicles=0, leftmost_threat_x=600,
                         priority_row=2, per_lane={2: [600]}),
            advice=Advice("defend_lane", 2, "next", "next", clock() + 2.0),
            reason_text="", rel_time_text="00:01", now=clock(),
        )
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 1 路防守")
        ctrl._update_display_from_frame(
            ch_info={"summary_text": "", "per_lane": {2: [560]}, "priority_row": 2,
                     "zombie_intel": {"by_lane": {
                         **channels["zombie_intel"]["by_lane"],
                         2: {"primary_entity": {"species_id": "buckethead_zombie", "x": 560}},
                     }}},
            advice=Advice("defend_lane", 2, "bucket", "bucket", clock() + 2.0),
            reason_text="铁桶僵尸", rel_time_text="00:01", now=clock(),
        )
        self.assertEqual(ctrl.ui_snapshot()["current_advice_title"], "关注第 2 路防守")

        clock.now += 0.25
        _, _, duplicate = engine.ingest_canonical_rgb(rgb, size)
        self.assertEqual(duplicate.focus_row, 1)
        self.assertGreaterEqual(engine.sidebar.duplicate_coalesced_count, 1)
        self.assertEqual(len(engine.sidebar.read().history), 1)
        with engine.registry._condition:
            self.assertFalse(engine.sidebar.present_direct_locked(evaluate_local_strategy(first)["strategy"], first))
            prior = engine.sidebar._current
            self.assertEqual(engine.expiry.armed.deadline, duplicate.expires_monotonic)

        clock.now += 0.25
        _, uncertain, withdrawn = engine.ingest_canonical_rgb(rgb, size)
        self.assertEqual(uncertain["uncertain_threats"], 1)
        self.assertIsNone(withdrawn)
        self.assertIsNone(engine.sidebar.read().current)

        clock.now += 0.25
        published, channels, latest = engine.ingest_canonical_rgb(rgb, size)
        ctrl._update_display_from_frame(
            ch_info=channels, advice=latest, reason_text=engine.sidebar.current_reason_text,
            rel_time_text="00:02", now=clock(),
        )
        self.assertEqual(latest.expires_monotonic, clock() + 2.0)
        clock.now = latest.expires_monotonic - 0.001
        with engine.registry._condition:
            self.assertFalse(engine.expiry.enforce_locked())
            self.assertFalse(engine.sidebar._is_current_locked(prior, clock()))
        self.assertEqual(engine.sidebar.read().current, latest)
        clock.now = latest.expires_monotonic
        with engine.registry._condition:
            self.assertTrue(engine.expiry.enforce_locked())
            self.assertGreater(engine.registry._epoch, published["continuity"]["epoch"])
            self.assertFalse(engine.sidebar.present_direct_locked(
                evaluate_local_strategy(published)["strategy"], published,
            ))
        self.assertEqual(engine.expiry.retire_count, 1)
        self.assertIsNone(engine.sidebar.read().current)
        self.assertFalse(ctrl.ui_snapshot()["has_active_advice"])
        clock.now += 0.25
        _, _, advice = engine.ingest_canonical_rgb(rgb, size)
        self.assertIsNone(advice)

        # Exercise the real perception wrapper with a vehicle-only frame and missing sun OCR.
        fast_perception = FastSinglePassPerception(REPO_ROOT)
        occluded = _FakePerception([dict(vehicle, sun=None, occ=0)])
        replay_engine = ContinuousAdviceEngine(clock=clock, perception=fast_perception, threaded_expiry=False)
        self.addCleanup(replay_engine.close)
        with (
            mock.patch.object(fast_perception.base, "observe", side_effect=occluded.observe),
            mock.patch.object(fast_perception, "_observe_cards_in_memory", side_effect=lambda _, state: state["loadout"]),
        ):
            observed, channels, advice = replay_engine.ingest_canonical_rgb(rgb, size)
            self.assertEqual(observed["fields"]["sun"], {"value": 0, "raw_value": None})
            self.assertIsNone(channels["sun_estimate"])
            self.assertEqual(advice.focus_row, 1)

        replay_path = REPO_ROOT / "data/day06/vehicle-v1/a-448.png"
        if replay_path.is_file():
            clock.now += 0.25
            with Image.open(replay_path) as frame, frame.convert("RGB") as image:
                _, channels, advice = replay_engine.ingest_canonical_rgb(image.tobytes(), size)
            self.assertEqual(channels["confirmed_vehicles"], 1)
            self.assertEqual(channels["lane_defense"]["by_row"][1]["verdict"], "unknown")
            self.assertEqual((advice.goal, advice.focus_row), ("defend_lane", 3))
            self.assertIn("路障僵尸", replay_engine.sidebar.current_reason_text)

    def test_16_llm_coach_species_prompt_vision_hint_and_vehicle_recheck(self):
        from unittest import mock

        clock = _FakeClock()
        vehicle = {"heads": 0, "vehicles": [{"row": 1, "front_xy": [650, 125]}], "spikeweed": True}
        clear = {"heads": 0, "sun": 100}
        perception = _FakePerception([vehicle, vehicle, clear, vehicle])
        engine = ContinuousAdviceEngine(clock=clock, perception=perception, threaded_expiry=False)
        self.addCleanup(engine.close)
        provider = mock.Mock()
        coach = AsyncLlmCoach(clock=clock, enabled=True, provider_factory=lambda _: provider)
        self.addCleanup(coach.close)
        jpeg_coach = AsyncLlmCoach(clock=clock, vision_mode="low_jpeg")
        self.addCleanup(jpeg_coach.close)
        rgb = bytes(RGB_BYTES)
        size = (1200, 900)
        phase = {"current_phase": "中期转型期"}
        source, channels, _ = engine.ingest_canonical_rgb(rgb, size)

        text_prompt = coach._build_prompt(source, phase, channels)
        self.assertIn(channels["zombie_intel"]["summary_short"].replace("1路:", "1路(陆):"), text_prompt)
        self.assertIn(channels["counter_plan"]["llm_counter_line"], text_prompt)
        self.assertIn("确认载具=1", text_prompt)
        self.assertIn("亮且够买：spikeweed(100)", text_prompt)
        self.assertNotIn("橄榄球僵尸", text_prompt)
        self.assertNotIn("巨人僵尸", text_prompt)
        suffix = f"原样回填 source_state_id={source['state_id']}"
        self.assertTrue(text_prompt.endswith(suffix))
        jpeg_prompt = jpeg_coach._build_prompt(source, phase, channels)
        for expected in ("核实护甲磨损", "橄榄球僵尸", "巨人僵尸"):
            self.assertIn(expected, jpeg_prompt)
        self.assertTrue(jpeg_prompt.endswith(suffix))

        strategy = evaluate_local_strategy(source)["strategy"]
        provider.complete.return_value = {"text": json.dumps(strategy)}
        coach._worker(source, b"", phase, channels, coach._epoch, clock())
        clock.now += 0.25
        current, _, _ = engine.ingest_canonical_rgb(rgb, size)
        verified = coach.recheck_against_current(current)
        self.assertFalse(verified.possibly_stale)
        self.assertEqual(coach.status_snapshot()["status"], "verified")
        for change, stale_reason in (
            ({"vehicles": dict(current["vehicles"], candidates=[{"row": 1, "front_xy": [200, 125]}])},
             "focus_warning_boundary_crossed"),
            ({"zombies": dict(current["zombies"], candidates=[{"row": 3, "head_xy": [400, 300]}])},
             "new_other_lane_warning"),
            ({"vehicles": dict(current["vehicles"], uncertain_candidates=[{}])}, "unlocalized_threat_context"),
        ):
            with self.subTest(stale_reason=stale_reason):
                self.assertEqual(coach._validate_hypothesis(source, {**current, **change}, strategy),
                                 (False, stale_reason))

        clock.now += 0.25
        cleared, channels, _ = engine.ingest_canonical_rgb(rgb, size)
        verified = coach.recheck_against_current(cleared)
        self.assertTrue(verified.possibly_stale)
        self.assertEqual(verified.stale_reason, "focus_lane_unconfirmed")

        macro = evaluate_local_strategy(cleared)["strategy"]
        self.assertEqual(macro["goal"], "develop_economy")
        provider.complete.return_value = {"text": json.dumps(macro)}
        coach._worker(cleared, b"", phase, channels, coach._epoch, clock())
        self.assertFalse(coach.recheck_against_current(cleared).possibly_stale)
        clock.now += 0.25
        spawned, _, _ = engine.ingest_canonical_rgb(rgb, size)
        verified = coach.recheck_against_current(spawned)
        self.assertTrue(verified.possibly_stale)
        self.assertEqual(verified.stale_reason, "vehicle_threat_appeared")


    def test_17_snorkel_accuracy_stances_and_diversified_counters(self):
        from unittest import mock

        perception = FastSinglePassPerception(REPO_ROOT)
        water = REPO_ROOT / "data/day06/water-v1"
        crossgame = REPO_ROOT / "data/day06/crossgame-v1/a/images"
        vehicle = REPO_ROOT / "data/day06/vehicle-v1"
        captures = REPO_ROOT / "evidence/day06/goal-live-v1/live-01/captures"

        def capture(second):
            return next(
                (path for path in sorted(captures.glob(f"*T12-08-{second}-*.png"))
                 if not path.name.endswith("-display.png")),
                captures / f"12-08-{second}.png",
            )

        replay_cases = [
            (crossgame / "450.000.png", [(4, "diving")], {3: "conehead_zombie"}),
            (capture("15"), [(3, "diving")], {4: "conehead_zombie"}),
            (capture("26"), [(3, "eating")], {}),
            (capture("27"), [(3, "eating")], {}),
            (water / "a-290.png", [(3, "eating")], {}),
            (water / "a-291.5.png", [(4, "diving")], {}),
            (water / "a-293.png", [(4, "submerged")], {}),
            (water / "a-293.25.png", [(4, "submerged")], {}),
            (water / "a-294.png", [(4, "submerged")], {}),
            (crossgame / "060.000.png", [], {}),
            (crossgame / "240.000.png", [], {3: "basic_zombie"}),
            (vehicle / "a-448.png", [], {3: "conehead_zombie", 4: "conehead_zombie"}),
            (capture("03"), [], {3: "buckethead_zombie"}),
            (capture("06"), [], {3: "buckethead_zombie", 4: "conehead_zombie"}),
            (crossgame / "000.000.png", [], {}),
            (_FALLBACK_REF_FRAME, [], {}),
        ]
        stages = {"diving": "跃入下潜", "eating": "露头啃咬", "submerged": "潜伏水中"}
        for index, (path, expected, other_species) in enumerate(replay_cases):
            with self.subTest(frame=path.name):
                if not path.is_file():
                    self.skipTest("optional replay frame is absent")
                with Image.open(path) as frame, frame.convert("RGB") as image:
                    state = perception.observe(
                        image, run_id=f"snorkel-replay-{index}", state_id=f"s{index}",
                        capture_time=f"replay-{index}", frame_time_seconds=0.0, tracking_epoch=index,
                        evidence={"image": path.relative_to(REPO_ROOT).as_posix(),
                                  "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                    )
                intel = state.get("zombie_intel") or classify_state_metadata(state)
                lanes = intel["by_lane"]
                snorkels = [
                    (row, entity["stance"], entity["armor_stage"])
                    for row, lane in lanes.items()
                    for entity in lane["entities"] if entity["species_id"] == "snorkel_zombie"
                ]
                self.assertEqual(snorkels, [(row, stance, stages[stance]) for row, stance in expected])
                for row, species in other_species.items():
                    self.assertEqual(lanes[row]["primary_entity"]["species_id"], species)
                    self.assertEqual(lanes[row]["primary_entity"]["stance"], "normal")
                if path == crossgame / "000.000.png":
                    self.assertEqual(lanes, {})
                if path == _FALLBACK_REF_FRAME:
                    self.assertTrue({3, 4}.isdisjoint(lanes))
                if path in (capture("26"), capture("27")):
                    self.assertFalse(any(head["row"] == 3 for head in state["zombies"]["candidates"]))
                    self.assertIn(3, evaluate_three_channels(REPO_ROOT, state)["per_lane"])
                    self.assertEqual(state["lane_defense"]["by_row"][3]["verdict"], "sufficient")
                    self.assertEqual(evaluate_local_strategy(state)["strategy"]["goal"], "hold_formation")

        rgb = np.zeros((600, 800, 3), dtype=np.uint8)
        rgb[302:307, 395:415] = (75, 25, 35)
        rgb[315:319, 389:411] = (192, 172, 37)
        ducky = rgb.copy()
        ducky[346:370, 385:421] = (246, 208, 35)
        blank = np.zeros_like(rgb)
        candidate = {"row": 4, "head_xy": [400, 324], "pose": "ordinary", "track_id": "track-1"}

        with tempfile.TemporaryDirectory() as root:
            classifier = ZombieSpeciesClassifier(Path(root))

        def classify(pixels, stamp, *, run="run-a", epoch=0, **metadata):
            state = {
                "scene": {"value": "playing"}, "run_id": run, "frame_time_seconds": stamp,
                "tracking_context": {"epoch": epoch}, "zombies": {"candidates": [{**candidate, **metadata}]},
            }
            return classifier.classify_frame(pixels, state)["by_lane"][4]["primary_entity"]

        entity = classify(rgb, 0.0)
        self.assertEqual((entity["species_id"], entity["stance"], entity["source"]),
                         ("snorkel_zombie", "diving", "snorkel_gear"))
        self.assertEqual(classify(ducky, 0.5)["species_id"], "basic_zombie")
        self.assertEqual(classify(blank, 1.0, hat_hypothesis="cone")["species_id"], "conehead_zombie")
        self.assertEqual(classify(blank, 1.5)["source"], "snorkel_track")
        self.assertEqual(classify(blank, 1.81)["species_id"], "basic_zombie")
        for reset in ({"run": "run-b"}, {"epoch": 1}, {}):
            with self.subTest(track_reset=reset):
                classify(rgb, 3.0, epoch=0)
                self.assertEqual(classify(blank, 3.1 if reset else 2.9, **reset)["species_id"], "basic_zombie")
        for pose in ("water_entry", "water_jump", "water_dive", "water_submerged", "water_eating"):
            with self.subTest(water_pose=pose):
                stance = {"water_submerged": "submerged", "water_eating": "eating"}.get(pose, "diving")
                self.assertEqual(classify(None, 4.0, water_pose=pose, species_hint="basic_zombie")["armor_stage"],
                                 stages[stance])
        self.assertEqual(classify(None, 4.0, pose="submerged")["stance"], "submerged")

        def plan_for(pose, plants, sun, *, count=1, cells=None, x=550):
            state = {"zombies": {"candidates": [
                {"row": 4, "head_xy": [x + idx * 50, 405], "water_pose": pose} for idx in range(count)
            ]}}
            slots = [
                {"slot": idx, "status": "observed", "plant_id": plant, "printed_cost": PLANT_CATALOG[plant]["cost"]}
                for idx, (plant, _) in enumerate(plants, start=1)
            ]
            visuals = [{"slot": idx, "visual_state": visual} for idx, (_, visual) in enumerate(plants, start=1)]
            return recommend_counter_plan(classify_state_metadata(state), slots, sun, raw_cards=visuals, cells=cells)

        kelp = ("tangle_kelp", "bright")
        pad = ("lily_pad", "bright")
        nut = ("wallnut", "bright")
        cherry = ("cherry_bomb", "bright")
        left_pad = [{"row": 4, "column": 3, "occupancy": "base_only"}]
        water_empty = [{"row": 4, "column": 4, "occupancy": "empty"}]
        land_empty = [{"row": 5, "column": 7, "occupancy": "empty"}]
        plans = [
            (plan_for("water_submerged", [kelp, pad], 100, cells=left_pad + water_empty), "海草(25)秒杀", True),
            (plan_for("water_submerged", [("tangle_kelp", "dim"), pad, nut], 100), "先垫睡莲再放坚果(50)", False),
            (plan_for("water_submerged", [pad], 25), "睡莲(25)垫潜逼迫露头", False),
            (plan_for("water_eating", [("repeater", "bright"), kelp], 200, cells=left_pad + water_empty),
             "在空睡莲上补双发射手(200)趁啃咬集火", True),
            (plan_for("water_eating", [("peashooter", "bright"), kelp], 100, cells=left_pad + water_empty),
             "豌豆(100)趁啃咬集火", True),
            (plan_for("water_eating", [("squash", "bright"), pad], 75), "先垫睡莲再放窝瓜(50)压杀", False),
            (plan_for("water_submerged", [kelp, cherry], 150, count=2, cells=water_empty + land_empty),
             "5路陆地放樱桃炸弹(150)溅射4路", False),
            (plan_for("water_submerged", [pad, nut], 75, x=280), "先垫睡莲再放坚果(50)", False),
            (plan_for("water_submerged", [kelp, pad], 49, count=2), "海草(25)秒杀", False),
            (plan_for("water_submerged", [cherry], 100, cells=left_pad), "攒阳到150再在空睡莲上放樱桃炸弹", True),
            (plan_for("water_submerged", [nut], 50, cells=[{"row": 4, "column": 8, "occupancy": "base_and_upper"}]),
             "啃咬露头时集中后排火力", False),
            (plan_for("water_submerged", [kelp, pad], 50, count=2), "海草(25)先抓最前，睡莲(25)垫潜", False),
            (plan_for("water_submerged", [("lily_pad", "dim")], 25), "等睡莲冷却后垫前逼露头", False),
            (plan_for("water_eating", [("peashooter", "bright"), kelp], 100, count=2), "海草(25)秒杀", False),
        ]
        for plan, expected, has_plants in plans:
            with self.subTest(counter=expected):
                self.assertIn(expected, plan["action_text"])
                self.assertLessEqual(len(plan["action_text"]), 46)
                self.assertEqual(plan["lane_has_plants"], has_plants)
                self.assertIn(len(plan["alternatives"]), (1, 2, 3))
                self.assertIn("备选对策", plan["llm_counter_line"])
        self.assertTrue(any("海草" in option for option in plans[3][0]["alternatives"]))
        self.assertTrue(any("海草" in option for option in plans[4][0]["alternatives"]))
        self.assertEqual(plans[6][0]["snorkel_count"], 2)

        # A splash-obscured sprite can provide defense evidence with no frozen head candidate.
        splash = (perception.zombie_classifier._ref_rgb.copy()
                  if perception.zombie_classifier._ref_rgb is not None else np.zeros_like(rgb))
        cv2.polylines(splash, [np.array([(387, 402), (390, 396), (395, 391), (402, 390)])], False, (78, 26, 37), 2)
        splash[399:402, 393:417] = (192, 172, 37)
        splash[402:407, 395:415] = (80, 80, 80)
        splash[407:419, 382:422] = (30, 185, 205)
        fake_base = _FakePerception([{"heads": 0, "sun": None}])
        with (
            Image.fromarray(splash) as image,
            mock.patch.object(perception.base, "observe", side_effect=fake_base.observe),
            mock.patch.object(perception, "_observe_cards_in_memory", side_effect=lambda _, state: state["loadout"]),
            mock.patch.object(Image.Image, "save", side_effect=AssertionError("frame write")),
            mock.patch.object(cv2, "imwrite", side_effect=AssertionError("frame write")),
        ):
            state = perception.observe(
                image, run_id="splash", state_id="splash", capture_time="splash",
                frame_time_seconds=0.0, tracking_epoch=0, evidence={},
            )
        self.assertEqual(state["zombies"]["candidates"], [])
        self.assertEqual(state["fields"]["sun"], {"value": 0, "raw_value": None})
        channels = evaluate_three_channels(REPO_ROOT, state)
        self.assertEqual((channels["priority_row"], channels["zombie_heads"]), (None, 0))
        self.assertEqual(len(channels["per_lane"][4]), 1)
        strategy = evaluate_local_strategy(state)["strategy"]
        self.assertEqual(channels["lane_defense"]["by_row"][4]["plants_compact_zh"], "海草x1")
        self.assertEqual((strategy["goal"], strategy["focus_row"]), ("hold_formation", None))

        coach = AsyncLlmCoach()
        self.addCleanup(coach.close)
        prompt = coach._build_prompt(state, {}, {"counter_plan": plans[4][0]})
        for expected in ("露头啃咬", "潜水×1", "4路(水) 无空睡莲", "备选对策", "豌豆"):
            self.assertIn(expected, prompt)
        hypothesis = {"goal": "defend_lane", "focus_row": 4}
        self.assertEqual(coach._validate_hypothesis(state, state, hypothesis), (False, "lane_already_sufficient"))
        unassessed = {**state, "lane_defense": None}
        self.assertEqual(coach._validate_hypothesis(unassessed, unassessed, hypothesis), (True, "ok"))
        self.assertEqual(coach._validate_hypothesis(unassessed, {**unassessed, "zombie_intel": {}}, hypothesis),
                         (False, "focus_lane_unconfirmed"))


    def test_18_plant_board_classifier(self):
        from pvz_agent.readonly_advisor_v1.fast_perception import REFERENCE_PNG
        from pvz_agent.readonly_advisor_v1.plant_intel import PlantBoardClassifier

        # These cells were reviewed visually. The frozen occupancy reader
        # abstains on several special plants in this capture.
        labels = {
            (1, 8): "squash", (2, 1): "peashooter", (2, 5): "wallnut", (3, 2): "lily_pad",
            (4, 4): "tangle_kelp", (5, 1): "sunflower", (5, 3): "repeater", (5, 5): "spikeweed",
            (5, 7): "potato_mine", (6, 2): "wallnut",
        }
        reviewed_fields = {"cells": [
            {"row": row, "column": col, "occupancy": "base_only" if kind == "lily_pad" else "ground_occupied"}
            for (row, col), kind in labels.items()
        ]}

        def indexed(board):
            return {(p["row"], p["column"]): p for row in board["cells_by_row"].values() for p in row}

        perception = None
        for number in (0, 340, 680, 1360, 1530, 1871):
            with self.subTest(frame=number):
                path = REFERENCE_PNG.with_name(f"frame-{number:04d}.png")
                if not REFERENCE_PNG.is_file() or not path.is_file():
                    self.skipTest("Canonical 3-7 evidence frames are absent")
                if perception is None:
                    perception = FastSinglePassPerception(REPO_ROOT)
                blob = path.read_bytes()
                frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(frame.shape, (600, 800, 3))
                with Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) as rgb:
                    state = perception.observe(
                        rgb, run_id="plant-board-replay", state_id=str(number), capture_time=str(number),
                        frame_time_seconds=number / 30, tracking_epoch=0,
                        evidence={"image": str(path), "image_sha256": hashlib.sha256(blob).hexdigest()},
                    )
                board = PlantBoardClassifier().classify_board(frame, state["fields"], state["zombie_intel"])
                actual = indexed(board)
                self.assertEqual(actual[2, 1]["plant_id"], "peashooter")
                self.assertEqual(actual[3, 2]["plant_id"], "lily_pad")
                self.assertEqual(actual[5, 3]["plant_id"], "repeater")
                if number == 680:
                    self.assertEqual(actual[2, 5]["plant_id"], "unknown_plant")
                self.assertLess(board["elapsed_ms"], 15)

                board = PlantBoardClassifier().classify_board(frame, reviewed_fields)
                actual = indexed(board)
                self.assertLess(board["elapsed_ms"], 15)
                self.assertEqual(len(actual), len(labels))
                for coord, kind in labels.items():
                    expected = kind
                    if (number, coord) in ((1871, (5, 1)), (0, (5, 5))):
                        expected = "unknown_plant"
                    self.assertEqual(actual[coord]["plant_id"], expected, (number, coord))
                self.assertEqual(actual[5, 1]["dps"], 0)
                self.assertEqual(actual[5, 7]["armed"], number == 340)
                self.assertEqual(actual[2, 5]["blocker_hp"], 3600 if number in (0, 1530, 1871) else 2400)
                self.assertEqual(board["row_summary"][5]["total_dps"], 27.0)

        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        classifier = PlantBoardClassifier()
        cells = {"cells": [
            {"row": 3, "column": 2, "occupancy": "base_only"},
            {"row": 4, "column": 2, "occupancy": "base_and_upper"},
            {"row": 5, "column": 2, "occupancy": "unknown", "occupied": True},
        ]}
        board = classifier.classify_board(frame, cells, timestamp_s=0)
        actual = indexed(board)
        self.assertEqual((actual[3, 2]["plant_id"], actual[3, 2]["blocker_hp"]), ("lily_pad", 300))
        self.assertEqual((actual[4, 2]["plant_id"], actual[4, 2]["carrier"]), ("unknown_plant", "lily_pad"))
        self.assertEqual(board["row_summary"][5]["unknown_occupied"], 1)
        self.assertEqual(board["row_summary"][4]["total_dps"], 0)

        # Draw a head, muzzle, eye and separate leaf cluster in memory to
        # test history and occlusion without any external evidence files.
        green = (25, 205, 125)
        roi = frame[163:243, 41:119]
        roi[:] = (35, 175, 32)
        cv2.ellipse(roi, (41, 26), (25, 15), 0, 0, 360, green, -1)
        cv2.ellipse(roi, (60, 26), (3, 8), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(roi, (41, 23), (2, 4), 0, 0, 360, (0, 0, 0), -1)
        cv2.ellipse(roi, (39, 67), (21, 10), 0, 0, 360, green, -1)
        occupied = {"cells": [{"row": 2, "column": 1, "occupancy": "ground_occupied"}]}
        empty = {"cells": [{"row": 2, "column": 1, "occupancy": "empty"}]}
        zombies = {"candidates": [{"row": 2, "head_xy": [85, 171]}]}

        def read(timestamp, threats=None):
            return indexed(classifier.classify_board(frame, occupied, threats, timestamp))[2, 1]

        self.assertEqual(read(0.0, zombies)["plant_id"], "unknown_plant")
        self.assertEqual(read(0.1)["plant_id"], "peashooter")
        self.assertEqual(read(0.2)["plant_id"], "peashooter")
        remembered = read(0.3, zombies)
        self.assertEqual(remembered["plant_id"], "peashooter")
        self.assertLess(remembered["confidence"], read(0.4)["confidence"])
        self.assertEqual(indexed(classifier.classify_board(frame, empty, timestamp_s=0.5)), {})
        self.assertEqual(read(0.6, zombies)["plant_id"], "unknown_plant")
        read(0.7)
        read(0.8)
        classifier.reset()
        self.assertEqual(read(0.9, zombies)["plant_id"], "unknown_plant")
        read(1.0)
        read(1.1)
        self.assertEqual(read(2.2, zombies)["plant_id"], "unknown_plant")
        frame[340:420, 121:199] = roi
        water_upper = indexed(classifier.classify_board(frame, cells, timestamp_s=2.3))[4, 2]
        self.assertEqual((water_upper["plant_id"], water_upper["carrier"]), ("peashooter", "lily_pad"))

    def test_19_lane_defense_evaluator(self):
        from pvz_agent.readonly_advisor_v1.lane_defense import evaluate_all_lanes, evaluate_lane_defense

        def plant(kind, column, **extra):
            return {"plant_id": kind, "column": column, **extra}

        def threat(kind="basic_zombie", x=690, **extra):
            return {"species_id": kind, "x": x, **extra}

        pea = plant("peashooter", 2)
        repeater = plant("repeater", 2)
        nut = plant("wallnut", 4, blocker_hp=3600)
        mine = plant("potato_mine", 4, armed=True)
        spike = plant("spikeweed", 4)
        kelp = plant("tangle_kelp", 4)
        squash = plant("squash", 4)
        pad = plant("lily_pad", 4)
        bucket = threat("buckethead_zombie")
        ice = threat("zomboni")
        dive = threat("snorkel_zombie", stance="diving")
        cases = (
            ("ordinary", 2, [pea], [threat()], "sufficient"),
            ("bucket", 2, [pea], [bucket], "deficit"),
            ("repeater and nut", 2, [repeater, nut], [bucket], "sufficient"),
            ("one tire", 2, [spike], [ice], "sufficient"),
            ("two tires", 2, [spike], [ice, threat("zomboni", 740)], "deficit"),
            ("walking over spikes", 2, [spike], [threat()], "deficit"),
            ("armed mine", 2, [mine], [bucket], "sufficient"),
            ("unarmed mine", 2, [plant("potato_mine", 4, armed=False)], [bucket], "deficit"),
            ("crushed mine", 2, [mine], [ice], "deficit"),
            ("mine splash", 2, [mine], [bucket, threat("buckethead_zombie", 725)], "sufficient"),
            ("mine crush before walker", 2, [mine], [ice, threat("basic_zombie", 740)], "deficit"),
            ("one kelp", 4, [kelp], [dive], "sufficient"),
            ("one kelp against two", 4, [kelp], [dive, threat("snorkel_zombie", 740, stance="submerged")], "deficit"),
            ("kelp against bucket", 4, [kelp], [bucket], "sufficient"),
            ("kelp on land", 2, [kelp], [threat()], "deficit"),
            ("squash against ice", 2, [squash], [ice], "sufficient"),
            ("squash against surfaced", 4, [squash], [threat("snorkel_zombie", stance="eating")], "sufficient"),
            ("squash against submerged", 4, [squash], [dive], "deficit"),
            ("no exposure", 4, [repeater, plant("repeater", 1)], [dive], "deficit"),
            ("short exposure", 4, [pea, pad], [dive], "deficit"),
            ("long exposure", 4, [pea, {**nut, "carrier": "lily_pad"}], [dive], "sufficient"),
            ("walker past the plants", 2, [repeater, nut], [threat(x=110)], "deficit"),
            ("wall cannot stop ice", 2, [repeater, nut], [ice], "deficit"),
            ("ice dies to fire", 2, [plant("repeater", c) for c in range(1, 6)], [ice], "sufficient"),
            ("no firing budget for both", 2, [pea], [threat(), threat(x=710)], "deficit"),
            ("marginal ordinary", 2, [pea], [threat(x=590)], "tight"),
            ("unknown firepower", 2, [pea, plant("unknown_plant", 3)], [bucket], "unknown"),
            ("known lower bound", 2, [pea, plant("unknown_plant", 3)], [threat()], "sufficient"),
        )
        for case, row, plants, threats, expected in cases:
            with self.subTest(case=case):
                lane = evaluate_lane_defense(row, plants, threats)
                self.assertEqual(lane["verdict"], expected, lane)
                self.assertEqual(lane["needs_reinforcement"], expected in ("tight", "deficit"))
        self.assertEqual(evaluate_lane_defense(4, [pea, pad], [dive])["stall_seconds"], 3.0)
        self.assertEqual(evaluate_lane_defense(2, [nut], [ice])["stall_seconds"], 0.0)
        self.assertEqual(evaluate_lane_defense(2, [spike], [threat()])["estimated_damage"], 85.0)
        self.assertEqual(evaluate_lane_defense(2, [mine], [ice, threat(x=740)])["estimated_damage"], 0.0)
        self.assertEqual(evaluate_lane_defense(2, [nut], [threat(), threat()])["stall_seconds"], 18.0)
        self.assertEqual(evaluate_lane_defense(4, [repeater], [dive])["effective_dps"], 0.0)

        board = {"cells_by_row": {1: [pea], 3: [kelp], 4: [pea, pad], 5: [pea]}}
        intel = {"by_lane": {1: {"entities": [threat()]}, 3: {"entities": [dive]}}}
        result = evaluate_all_lanes(board, intel)
        self.assertTrue(result["all_threatened_sufficient"])
        self.assertEqual(result["needs_defense_rows"], [])
        self.assertEqual(result["sufficient_threat_rows"], [1, 3])
        self.assertEqual(result["by_row"][6]["verdict"], "clear")
        self.assertFalse(evaluate_all_lanes(board, {})["all_threatened_sufficient"])
        intel["by_lane"].update({
            2: {"entities": [threat()], "lane_urgency": 2},
            4: {"entities": [dive], "lane_urgency": 3},
            5: {"entities": [threat(x=590)], "lane_urgency": 9},
            6: {"entities": [bucket], "lane_urgency": 8},
        })
        fields = {"cells": [{"row": 6, "column": 4, "occupancy": "ground_occupied"}]}
        result = evaluate_all_lanes(board, intel, fields)
        self.assertEqual(result["needs_defense_rows"], [4, 2, 5])
        self.assertEqual(result["unknown_threat_rows"], [6])
        self.assertFalse(result["all_threatened_sufficient"])
        self.assertIn("3路 海草x1，稳守", result["summary_zh"])


    def test_20_plant_low_profile_trap_rescue(self):
        from pvz_agent.readonly_advisor_v1.fast_perception import REFERENCE_PNG
        from pvz_agent.readonly_advisor_v1.plant_intel import PlantBoardClassifier

        classifier = PlantBoardClassifier()
        no_reference = PlantBoardClassifier(REFERENCE_PNG.parent)
        fields = {"cells": [
            {"row": row, "column": col, "occupancy": "empty"}
            for row in range(1, 7) for col in range(1, 10)
        ]}
        for number in (0, 170, 340, 510, 680, 850, 1020, 1190, 1360, 1530, 1700, 1871):
            with self.subTest(frame=number):
                path = REFERENCE_PNG.with_name(f"frame-{number:04d}.png")
                if not REFERENCE_PNG.is_file() or not path.is_file():
                    self.skipTest("Canonical 3-7 replay frame is absent")
                frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                board = classifier.classify_board(frame, fields, timestamp_s=number / 30)
                expected = {(4, 4): "tangle_kelp"}
                if number not in (0, 170, 510):
                    expected[5, 5] = "spikeweed"
                if number == 340:
                    expected[5, 7] = "potato_mine"
                detected = {
                    (p["row"], p["column"]): p["plant_id"]
                    for plants in board["cells_by_row"].values() for p in plants
                    if p["source"] == "low_profile_scan"
                }
                self.assertEqual(detected, expected)
                self.assertLess(board["elapsed_ms"], 15)
                if number == 340:
                    self.assertTrue(next(p for p in board["cells_by_row"][5] if p["column"] == 7)["armed"])
                    self.assertFalse(any(no_reference.classify_board(frame, fields)["cells_by_row"].values()))
                    blocked = {"candidates": [
                        {"row": row, "head_xy": [80 * col, 85 * row]} for row, col in expected
                    ]}
                    occluded = classifier.classify_board(frame, fields, blocked)
                    self.assertFalse(any(occluded["cells_by_row"].values()))
                    perception = FastSinglePassPerception(REPO_ROOT)
                    with Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) as rgb:
                        state = perception.observe(
                            rgb, run_id="trap-replay", state_id=str(number), capture_time=str(number),
                            frame_time_seconds=number / 30, tracking_epoch=1,
                            evidence={"image": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                        )
                    actual = {(p["row"], p["column"]): p["plant_id"]
                              for plants in state["plant_intel"]["cells_by_row"].values() for p in plants
                              if p.get("source") == "low_profile_scan"}
                    self.assertEqual(actual, expected)


    def test_21_lane_sufficiency_runtime_suppression(self):
        from unittest import mock
        from pvz_agent.resource_sidebar import Advice

        clock = _FakeClock()
        perception = FastSinglePassPerception(REPO_ROOT)
        engine = ContinuousAdviceEngine(clock=clock, perception=perception, threaded_expiry=False)
        controller = AdvisorController(clock=clock, engine=engine, min_dwell_s=3.5)
        self.addCleanup(controller.close)
        original_perception = _FakePerception([{"row": 3, "head_x": 690, "occ": 0, "sun": 0}])

        def observe_frozen(image, **kwargs):
            state = original_perception.observe(image, **kwargs)
            state["zombies"]["candidates"][0].update(species_hint="snorkel_zombie", water_pose="water_submerged")
            if original_perception.calls >= 3:
                state["zombies"]["candidates"].append({"row": 2, "head_xy": [690.0, 180.0]})
            return state

        def legacy_backend(state):
            return {"strategy": {
                "source_state_id": state["state_id"], "goal": "defend_lane", "focus_row": 3,
                "reason": "第3路应补火力。", "uncertainties": [],
            }}

        rgb = np.full((600, 800, 3), (32, 175, 35), dtype=np.uint8)
        rgb[256:438] = (55, 155, 220)
        deployed = rgb.copy()
        cv2.ellipse(deployed, (320, 312), (30, 18), 0, 0, 360, (25, 57, 10), -1)
        cv2.ellipse(deployed, (314, 324), (9, 5), 0, 0, 360, (214, 49, 34), -1)

        def ingest(frame):
            published, channels, advice = engine.ingest_canonical_rgb(frame.tobytes(), (800, 600))
            controller._update_display_from_frame(
                ch_info=channels, advice=advice, reason_text=engine.sidebar.current_reason_text,
                now=clock(), rel_time_text="00:00",
            )
            return published, channels, advice

        with (
            mock.patch.object(perception.base, "observe", side_effect=observe_frozen),
            mock.patch.object(perception, "_observe_cards_in_memory", side_effect=lambda _, state: state["loadout"]),
            mock.patch.object(engine.supervisor.reviewer, "backend", side_effect=legacy_backend),
            mock.patch.object(engine.sidebar, "_present", wraps=engine.sidebar._present) as frozen_present,
            mock.patch.object(engine.sidebar, "present_direct_locked", wraps=engine.sidebar.present_direct_locked) as direct,
        ):
            _, _, first = ingest(rgb)
            self.assertIsInstance(first, Advice)
            self.assertEqual((first.goal, first.focus_row), ("defend_lane", 3))
            self.assertEqual(controller.ui_snapshot()["current_advice_title"], "关注第 3 路防守")

            frozen_present.reset_mock()
            clock.now += 0.25
            stable, channels, current = ingest(deployed)
            self.assertEqual(frozen_present.call_args.args[1].advice.focus_row, 3)
            self.assertTrue(direct.called)
            self.assertTrue(stable["lane_defense"]["all_threatened_sufficient"])
            self.assertEqual(channels["counter_plan"]["status"], "all_sufficient")
            self.assertIsNone(channels["counter_plan"]["priority_row"])
            self.assertEqual(channels["counter_plan"]["recommended_plants"], [])
            self.assertIn("3路[海草x1]已稳守", channels["counter_plan"]["action_hint"])
            self.assertEqual(evaluate_local_strategy(stable)["strategy"]["goal"], "hold_formation")
            funded = {**stable, "fields": {**stable["fields"], "sun": {"value": 50}}}
            self.assertEqual(evaluate_local_strategy(funded)["strategy"]["goal"], "develop_economy")
            with mock.patch("pvz_agent.readonly_advisor_v1.fast_perception.sun_cue_ids", return_value=["sun-1"]):
                self.assertEqual(evaluate_local_strategy(funded)["strategy"]["goal"], "collect_resources")
            self.assertEqual((current.goal, current.focus_row), ("hold_formation", None))
            self.assertEqual(controller.ui_snapshot()["current_advice_title"], "维持当前阵型")
            self.assertEqual(engine.sidebar.read().current, current)

            clock.now += 0.25
            mixed, channels, current = ingest(deployed)
            self.assertEqual(mixed["zombie_intel"]["priority_row"], 3)
            self.assertEqual(channels["lane_defense"]["needs_defense_rows"], [2])
            self.assertEqual(channels["counter_plan"]["priority_row"], 2)
            self.assertIn("3路[海草x1]已稳守，无需重复增防", channels["counter_plan"]["llm_counter_line"])
            self.assertEqual((current.goal, current.focus_row), ("defend_lane", 2))
            self.assertEqual(controller.ui_snapshot()["current_advice_title"], "关注第 2 路防守")
            clock.now += 0.25
            _, _, current = ingest(deployed)
            self.assertEqual([advice.focus_row for advice in engine.sidebar.read().history], [3, None, 2])
            self.assertAlmostEqual(current.expires_monotonic, clock() + 2.0)
            clock.now = current.expires_monotonic
            self.assertIsNone(engine.sidebar.read().current)
            self.assertFalse(controller.ui_snapshot()["has_active_advice"])

    def test_22_llm_lane_sufficiency_prompt_and_recheck(self):
        from unittest import mock
        from pvz_agent.readonly_advisor_v1.lane_defense import evaluate_all_lanes

        clock = _FakeClock()
        engine = ContinuousAdviceEngine(
            clock=clock, perception=_FakePerception([{"row": 3, "head_x": 680, "occ": 0}]),
            threaded_expiry=False,
        )
        self.addCleanup(engine.close)
        provider = mock.Mock()
        coach = AsyncLlmCoach(clock=clock, provider_factory=lambda _: provider, enabled=True)
        self.addCleanup(coach.close)
        source, _, _ = engine.ingest_canonical_rgb(bytes(RGB_BYTES), (800, 600))
        plants = {"cells_by_row": {3: [
            {"column": 4, "plant_id": "tangle_kelp"},
            {"column": 2, "plant_id": "repeater", "carrier": "lily_pad"},
        ]}, "row_summary": {3: {"total_dps": 27.0}}}

        def frame(base, board=None, *, new_deficit=False):
            updated = dict(base)
            if new_deficit:
                candidates = base["zombies"]["candidates"] + [{"row": 2, "head_xy": [670.0, 160.0]}]
                updated["zombies"] = {**base["zombies"], "candidates": candidates}
            intel = classify_state_metadata(updated)
            updated["zombie_intel"] = intel
            updated["plant_intel"] = board or {}
            updated["lane_defense"] = evaluate_all_lanes(board or {}, intel)
            return updated

        def hypothesis(source, goal, row=None):
            strategy = {"source_state_id": source["state_id"], "goal": goal, "focus_row": row,
                        "reason": "保留现有部署并按局势调整。", "uncertainties": []}
            provider.complete.return_value = {"text": json.dumps(strategy)}
            coach._worker(source, b"", {}, evaluate_three_channels(REPO_ROOT, source), coach._epoch, clock())
            return strategy

        source = frame(source)
        steady = frame(source, plants)
        mixed = frame(source, plants, new_deficit=True)
        obscured = frame(source, {"cells_by_row": {3: [{"column": 4, "plant_id": "unknown_plant"}]}})
        obscured_channels = evaluate_three_channels(REPO_ROOT, obscured)
        self.assertEqual(obscured_channels["counter_plan"]["status"], "observe_unknown")
        self.assertIsNone(obscured_channels["counter_plan"]["priority_row"])
        self.assertIn("先观察", evaluate_local_strategy(obscured)["strategy"]["reason"])
        prompt = coach._build_prompt(mixed, {}, evaluate_three_channels(REPO_ROOT, mixed))
        for line in ("每路现有部署与充分性评估", "3路(水) 无空睡莲 双发x1+海草x1，27 DPS/阻挡0 HP",
                     "｜稳守", "2路(陆) 无已确认部署", "｜缺口", "稳守路严禁追加射手、坚果或一次性消耗植物"):
            self.assertIn(line, prompt)
        defense = hypothesis(source, "defend_lane", 3)
        self.assertFalse(coach.recheck_against_current(source).possibly_stale)
        now_stale = coach.recheck_against_current(steady)
        self.assertTrue(now_stale.possibly_stale)
        self.assertEqual(now_stale.stale_reason, "lane_already_sufficient")
        self.assertEqual(coach._validate_hypothesis(source, mixed, defense), (False, "higher_deficit_lane_present"))
        self.assertEqual(coach.recheck_against_current(mixed).stale_reason, "higher_deficit_lane_present")
        hypothesis(steady, "develop_economy")
        self.assertFalse(coach.recheck_against_current(steady).possibly_stale)
        now_stale = coach.recheck_against_current(mixed)
        self.assertTrue(now_stale.possibly_stale)
        self.assertEqual(now_stale.stale_reason, "lane_defense_deficit_appeared")

    def test_23_reasoning_none_minimal_and_graduated_400_fallback(self):
        for protocol in llm.SUPPORTED_PROTOCOLS:
            responses = protocol == "responses"
            field = "reasoning" if responses else "reasoning_effort"
            budget = "max_output_tokens" if responses else "max_tokens"
            config = {"base_url": "https://gateway.example/v1", "model": "test-model", "protocol": protocol}
            reply = _completion_payload(protocol, "OK")

            for effort, tokens in zip(llm.SUPPORTED_REASONING_EFFORTS, (140, 140, 240, 320, 160)):
                with self.subTest(protocol=protocol, effort=effort):
                    transport = mock.Mock(return_value=reply)
                    provider = llm.UserConfigurableProvider(
                        {**config, "reasoning_effort": effort}, key_getter=lambda: "key", transport=transport,
                    )
                    provider.complete("ping")
                    body = transport.call_args.args[1]
                    expected = None if effort == "auto" else ({"effort": effort} if responses else effort)
                    self.assertEqual(body.get(field), expected)
                    self.assertEqual(body[budget], tokens)
                    provider.complete("ping", max_tokens=77)
                    self.assertEqual(transport.call_args.args[1][budget], 77)

            for initial, chain, tokens in (
                ("none", ("none", "minimal", "low", None), (140, 140, 240, 160)),
                ("minimal", ("minimal", "low", None), (140, 240, 160)),
                ("low", ("low", None), (240, 160)),
                ("medium", ("medium", "low", None), (320, 240, 160)),
            ):
                with self.subTest(protocol=protocol, fallback=initial):
                    def transport_call(_url, body, _key, _timeout):
                        if field in body:
                            raise ProviderError("provider_http_error", 400)
                        return reply

                    transport = mock.Mock(side_effect=transport_call)
                    provider = llm.UserConfigurableProvider(
                        {**config, "reasoning_effort": initial}, key_getter=lambda: "key", transport=transport,
                    )
                    result = provider.complete("ping")
                    provider.complete("again")
                    sent = [call.args[1] for call in transport.call_args_list]
                    expected = [({"effort": e} if responses else e) if e else None for e in (*chain, None)]
                    self.assertEqual([body.get(field) for body in sent], expected)
                    self.assertEqual([body[budget] for body in sent], [*tokens, 160])
                    self.assertIsNone(result["effective_reasoning_effort"])

    def test_24_prompt_cache_split_store_false_and_schema_fallback(self):
        coach = AsyncLlmCoach()
        self.addCleanup(coach.close)
        first, second = {"state_id": "frame-100"}, {"state_id": "frame-101"}
        instructions, user_prompt = coach._build_prompt_parts(first, {"sun_estimate": 50})
        next_instructions, next_user_prompt = coach._build_prompt_parts(second, {"sun_estimate": 125})
        self.assertEqual(instructions, next_instructions)
        self.assertIn("稳守路严禁追加", instructions)
        self.assertIn("reason≤48字", instructions)
        self.assertIn("阳光=50", user_prompt)
        self.assertIn("阳光=125", next_user_prompt)
        self.assertNotIn("frame-100", instructions + user_prompt)
        schema = llm.LLM_COMPACT_STRATEGY_SCHEMA
        self.assertEqual(set(schema["required"]), {"goal", "focus_row", "reason"})
        self.assertEqual(set(schema["properties"]), set(schema["required"]))

        pool = llm._KeepAlivePool()
        self.addCleanup(pool.close_all)
        conn = mock.Mock(sock=None)
        success = _completion_payload("responses", "OK")
        conn.getresponse.side_effect = [
            _http_response({"error": {"message": "store and strict are unsupported"}}, status=400),
            _http_response({"error": {"param": "text.format", "message": "json_schema unsupported"}}, status=400),
            _http_response({"error": {"message": "json_object unsupported"}}, status=400),
            _http_response(success),
            _http_response(success),
        ]
        config = {"base_url": "https://gateway.example/v1", "model": "test-model", "reasoning_effort": "none"}
        with mock.patch.object(pool, "_open_conn", return_value=conn):
            provider = llm.UserConfigurableProvider(config, key_getter=lambda: "key", transport=pool.post_json)
            provider.complete(user_prompt, instructions=instructions, json_schema=schema)
            provider.complete(next_user_prompt, instructions=instructions, json_schema=schema)
        sent = [json.loads(call.kwargs["body"]) for call in conn.request.call_args_list]
        self.assertEqual(len(sent), 5)
        self.assertIs(sent[0]["store"], False)
        self.assertTrue(all("store" not in body for body in sent[1:]))
        self.assertTrue(all(body["instructions"] == instructions for body in sent))
        self.assertEqual(sent[0]["input"], [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}])
        self.assertEqual(sent[-1]["input"][0]["content"][0]["text"], next_user_prompt)
        self.assertTrue(sent[0]["text"]["format"]["strict"])
        self.assertEqual(sent[1]["text"]["format"]["type"], "json_schema")
        self.assertEqual(sent[2]["text"]["format"], {"type": "json_object"})
        self.assertTrue(all("text" not in body for body in sent[3:]))
        self.assertTrue(all(body["reasoning"] == {"effort": "none"} for body in sent))
        self.assertTrue(provider._store_unsupported)
        self.assertEqual(provider._schema_fallback_mode, "plain")

        provider = llm.UserConfigurableProvider(
            {**config, "protocol": "chat_completions", "schema_mode": "json_object"},
            key_getter=lambda: "key", transport=mock.Mock(return_value=_completion_payload("chat_completions", "OK")),
        )
        provider.complete(user_prompt, instructions=instructions, json_schema=schema)
        body = provider._transport.call_args.args[1]
        self.assertEqual(body["messages"], [
            {"role": "system", "content": instructions},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("store", body)

    def test_25_compact_output_source_state_id_backfill_and_sse_stream(self):
        compact = {"goal": "hold", "focus_row": None, "reason": "保留阳光并继续观察。"}
        clock = _FakeClock()
        state = {"state_id": "local-frame-id"}
        transport = mock.Mock()
        coach = AsyncLlmCoach(clock=clock, enabled=True, transport=transport)
        self.addCleanup(coach.close)
        coach.configure_connection(base_url="https://gateway.example/v1", api_key="key", protocol="responses")
        for extra_fields in ({}, {"source_state_id": "  ", "uncertainties": None}):
            transport.return_value = _completion_payload("responses", json.dumps({**compact, **extra_fields}))
            coach._worker(state, b"", {}, {}, coach._epoch, clock())
            result = coach._pending_hypothesis["strategy"]
            self.assertEqual(result, {**compact, "source_state_id": state["state_id"], "uncertainties": []})
            self.assertEqual(result, validate_strategy(result, state["state_id"]))
        coach._pending_hypothesis = None
        transport.return_value = _completion_payload("responses", json.dumps({**compact, "source_state_id": "wrong"}))
        coach._worker(state, b"", {}, {}, coach._epoch, clock())
        self.assertIsNone(coach._pending_hypothesis)
        self.assertEqual(coach.status_snapshot()["status"], "error")

        def event(value):
            return f"data: {json.dumps(value, ensure_ascii=False)}\r\n\r\n"

        text = json.dumps(compact, ensure_ascii=False)
        streams = {
            "responses": (
                event({"type": "response.created", "response": {"model": "stream-model"}})
                + event({"type": "response.output_text.delta", "delta": text[:20]})
                + event({"type": "response.output_text.delta", "delta": text[20:]})
                + event({"type": "response.completed", "response": {"status": "completed"}})
            ),
            "chat_completions": (
                event({"model": "stream-model", "choices": [{"delta": {"reasoning_content": "internal"}}]})
                + event({"choices": [{"delta": {"content": text[:20]}}]})
                + event({"choices": [{"delta": {"content": text[20:]}, "finish_reason": "stop"}]})
                + "data: [DONE]\r\n\r\n"
            ),
        }
        for protocol, sse in streams.items():
            with self.subTest(protocol=protocol):
                pool = llm._KeepAlivePool()
                self.addCleanup(pool.close_all)
                conn = mock.Mock(sock=None)
                plain = _completion_payload(protocol, "JSON fallback")
                conn.getresponse.side_effect = [
                    _http_response(sse.encode(), content_type="text/event-stream; charset=utf-8"),
                    _http_response(plain),
                    _http_response(plain),
                ]
                config = {"base_url": "https://gateway.example/v1", "model": "test-model", "protocol": protocol,
                          "stream_enabled": True}
                provider = llm.UserConfigurableProvider(config, key_getter=lambda: "key", transport=pool.post_json)
                with mock.patch.object(pool, "_open_conn", return_value=conn), mock.patch.object(
                    llm.time, "perf_counter", side_effect=[1.0, 1.125, 2.0, 3.0],
                ):
                    result = provider.complete("ping")
                    self.assertEqual(result["text"], text)
                    self.assertEqual(result["model_returned"], "stream-model")
                    self.assertAlmostEqual(result["ttft_ms"], 125.0)
                    fallback = provider.complete("ping")
                    self.assertEqual(fallback["text"], "JSON fallback")
                    self.assertIsNone(fallback["ttft_ms"])
                    self.assertTrue(provider._stream_unsupported)
                    provider.complete("ping")
                sent = [json.loads(call.kwargs["body"]) for call in conn.request.call_args_list]
                self.assertEqual([body.get("stream") for body in sent], [True, True, None])

                conn.reset_mock()
                conn.getresponse.side_effect = [
                    _http_response({"error": {"param": "stream", "message": "not supported"}}, status=400),
                    _http_response(plain),
                ]
                provider = llm.UserConfigurableProvider(config, key_getter=lambda: "key", transport=pool.post_json)
                with mock.patch.object(pool, "_open_conn", return_value=conn):
                    self.assertEqual(provider.complete("ping")["text"], "JSON fallback")
                sent = [json.loads(call.kwargs["body"]) for call in conn.request.call_args_list]
                self.assertEqual([body.get("stream") for body in sent], [True, None])
                self.assertEqual(sent[0].get("reasoning", sent[0].get("reasoning_effort")),
                                 sent[1].get("reasoning", sent[1].get("reasoning_effort")))

    def test_26_event_driven_early_dispatch_cooldown_and_warmup(self):
        clock = _FakeClock()
        coach = AsyncLlmCoach(clock=clock, provider_factory=lambda _: mock.Mock(), enabled=True, min_interval_s=0.5)
        self.addCleanup(coach.close)
        self.assertEqual(coach.status_snapshot()["interval_s"], 3)
        coach.set_interval(100)
        self.assertEqual(coach.status_snapshot()["interval_s"], 60)
        coach.set_interval(18)
        state = {}
        channels = {"per_lane": {3: [400]}, "lane_defense": {"needs_defense_rows": [3]}}

        def dispatch(now):
            clock.now = now
            sent = coach.maybe_dispatch(state=state, rgb_bytes=b"", phase_info={}, channel_info=channels)
            if sent:
                coach._inflight = False
            return sent

        with mock.patch.object(coach._executor, "submit") as submit:
            self.assertTrue(dispatch(100))
            self.assertFalse(dispatch(106))
            state["vehicles"] = {"candidates": [{"row": 5, "front_xy": [600, 440]}]}
            self.assertTrue(dispatch(106))
            channels["per_lane"][3] = [199]
            self.assertFalse(dispatch(109.49))
            self.assertTrue(dispatch(109.5))
            channels["lane_defense"]["needs_defense_rows"] = [3, 5]
            self.assertFalse(dispatch(112))
            self.assertTrue(dispatch(113))
            channels["lane_defense"]["needs_defense_rows"] = [5, 3]
            self.assertFalse(dispatch(130.99))
            self.assertTrue(dispatch(131))
            self.assertEqual(submit.call_count, 5)

        pool = llm._KeepAlivePool()
        self.addCleanup(pool.close_all)
        conn = mock.Mock(sock=None)
        conn.getresponse.return_value = _http_response(_completion_payload("responses", "OK"))
        entered, release, loop_started = threading.Event(), threading.Event(), threading.Event()

        def connect():
            entered.set()
            release.wait(2.0)

        conn.connect.side_effect = connect
        warm_coach = AsyncLlmCoach(enabled=True)
        self.assertEqual(warm_coach.status_snapshot()["interval_s"], 18)
        warm_coach.configure_connection(base_url="https://localhost/v1", api_key="key", protocol="responses")
        controller = AdvisorController(root=REPO_ROOT, llm_coach=warm_coach, selector=mock.Mock())
        self.addCleanup(controller.close)
        controller._lease = mock.Mock(closed=False)
        with mock.patch.object(llm, "_SHARED_KEEPALIVE_POOL", pool), mock.patch.object(
            pool, "_open_conn", return_value=conn,
        ) as open_conn, mock.patch.object(controller, "_loop", side_effect=loop_started.set):
            try:
                self.assertTrue(controller.start_monitoring()[0])
                self.assertTrue(entered.wait(1.0))
                self.assertTrue(loop_started.wait(1.0))
                conn.request.assert_not_called()
            finally:
                release.set()
            deadline = time.monotonic() + 2.0
            route = pool._route("https://localhost/v1")
            while route not in pool._conns and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIs(pool._conns.get(route), conn)
            self.assertEqual(warm_coach._get_provider().complete("ping")["text"], "OK")
            open_conn.assert_called_once_with("https", "localhost", 443, "", 5.0)
            conn.connect.assert_called_once()

        import tkinter as tk

        def ping(_url, body, _key, _timeout):
            clock.now += 0.21
            if body.get("reasoning_effort") == "none":
                raise ProviderError("provider_http_error", 400)
            return _completion_payload("chat_completions", "OK")

        ping_coach = AsyncLlmCoach(clock=clock, transport=ping)
        ctrl = AdvisorController(root=REPO_ROOT, clock=clock, llm_coach=ping_coach)
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalSettingsStore(path=Path(tmp) / "settings.json")
            root = tk.Tk()
            app = ReadonlyAdvisorSidebarApp(root, ctrl, settings_store=store)
            try:
                app._base_url_var.set("https://gateway.example/v1")
                app._api_key_var.set("test-key")
                app._protocol_var.set("chat_completions")
                app._reasoning_effort_var.set("none(关闭思考)")
                app._llm_interval_var.set("3")
                app._stream_var.set(True)
                app._test_connection()
                deadline = time.monotonic() + 2.0
                while app._ui_queue.empty() and time.monotonic() < deadline:
                    time.sleep(0.005)
                app._drain_ui_queue()
                self.assertIn("420 ms (0.42s) · chat_completions · 生效思考档 minimal",
                              app._conn_feedback_lbl.cget("text"))
                saved = store.load()
                self.assertIs(saved["stream_enabled"], True)
                self.assertEqual(saved["min_interval_s"], 3.0)
                self.assertEqual(saved["protocol"], "chat_completions")
                self.assertEqual(saved["reasoning_effort"], "none")
            finally:
                app.close()


    def test_27_terrain_placement_legality_guardrails(self):
        from pvz_agent.readonly_advisor_v1.lane_defense import evaluate_all_lanes
        from pvz_agent.readonly_advisor_v1.plant_intel import PlantBoardClassifier

        def cell(row, occupancy, column=7):
            return {"row": row, "column": column, "occupancy": occupancy}

        def plan(species, row, cards=(), sun=250, cells=None):
            candidate = {"row": row, "head_xy": [550, 85 * row], "species_hint": species,
                         "helmet_support": {"area": 600}}
            state = {"zombies": {"candidates": [candidate]}}
            if species == "snorkel_zombie":
                candidate["water_pose"] = "water_eating"
            if species == "zomboni":
                state = {"vehicles": {"candidates": [{"row": row, "front_xy": [550, 85 * row]}]}}
            intel = classify_state_metadata(state)
            self.assertEqual(intel["priority_row"], row)
            self.assertEqual(intel["by_lane"][row]["primary_entity"]["species_id"], species)
            slots = [
                {"slot": i, "status": "observed", "plant_id": plant, "printed_cost": PLANT_CATALOG[plant]["cost"]}
                for i, (plant, _) in enumerate(cards, 1)
            ]
            visuals = [{"slot": i, "visual_state": visual} for i, (_, visual) in enumerate(cards, 1)]
            return recommend_counter_plan(intel, slots, sun, raw_cards=visuals, cells=cells)

        common = ("basic_zombie", "conehead_zombie", "buckethead_zombie")
        for row in (3, 4):
            for species in (*common, "snorkel_zombie"):
                for sun, cards in (
                    (250, ()), (250, (("tangle_kelp", "dim"), ("potato_mine", "bright"))),
                    (0, (("tangle_kelp", "bright"), ("spikeweed", "dim"))),
                    (250, (("potato_mine", "bright"), ("spikeweed", "bright"))),
                ):
                    with self.subTest(row=row, species=species, sun=sun, cards=cards):
                        result = str(plan(species, row, cards, sun))
                        for illegal in ("potato_mine", "土豆雷", "spikeweed", "地刺"):
                            self.assertNotIn(illegal, result)
        for row in (1, 2, 5, 6):
            for species in (*common, "zomboni"):
                for cards in ((), (("tangle_kelp", "bright"),), (("lily_pad", "bright"),)):
                    with self.subTest(row=row, species=species, cards=cards):
                        result = str(plan(species, row, cards))
                        for illegal in ("tangle_kelp", "海草", "lily_pad", "睡莲"):
                            self.assertNotIn(illegal, result)

        for species in ("buckethead_zombie", "snorkel_zombie"):
            for plant in ("squash", "repeater"):
                with self.subTest(species=species, plant=plant):
                    cards = [(plant, "bright")]
                    empty = [cell(3, "empty")]
                    occupied_pad = [cell(3, "base_and_upper")]
                    for cells in (None, empty, occupied_pad):
                        result = plan(species, 3, cards, cells=cells)
                        self.assertNotIn(plant, result["recommended_plants"])
                        self.assertNotIn(PLANT_CATALOG[plant]["name_cn"], str(result))
                    result = plan(species, 3, cards, cells=[cell(3, "base_only")])
                    self.assertEqual(result["recommended_plants"], [plant])
                    self.assertIn("在空睡莲上", result["action_text"])
                    cards.append(("lily_pad", "bright"))
                    total = PLANT_CATALOG[plant]["cost"] + 25
                    result = plan(species, 3, cards, sun=total, cells=empty)
                    self.assertEqual(result["recommended_plants"], ["lily_pad", plant])
                    self.assertIn("先垫睡莲再", result["action_text"])
                    self.assertNotIn(plant, plan(species, 3, cards, sun=total - 1, cells=empty)["ready_counters"])
                    self.assertNotIn(plant, plan(species, 3, cards, cells=occupied_pad)["ready_counters"])
        for plant in ("lily_pad", "tangle_kelp"):
            for visual in ("bright", "dim"):
                result = plan("snorkel_zombie", 3, [(plant, visual)], cells=[cell(3, "base_only")])
                self.assertNotIn(plant, result["recommended_plants"])
                self.assertNotIn(PLANT_CATALOG[plant]["name_cn"] + "(25)", str(result))

        for water, land in ((3, 2), (4, 5)):
            cards = [("cherry_bomb", "bright")]
            action = plan("buckethead_zombie", water, cards, cells=[cell(land, "empty")])["action_text"]
            self.assertIn(f"{land}路陆地放樱桃炸弹(150)溅射{water}路", action)
            for cells in ([cell(land, "unknown")], [cell(land, "empty", 1)], [cell(land, "ground_occupied")]):
                self.assertNotIn("cherry_bomb", plan("buckethead_zombie", water, cards, cells=cells)["recommended_plants"])

        board_classifier = PlantBoardClassifier()
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        with mock.patch("pvz_agent.readonly_advisor_v1.plant_intel._identify", side_effect=[
            ("tangle_kelp", 1.0, 0, False), ("potato_mine", 1.0, 0, True),
        ]), mock.patch("pvz_agent.readonly_advisor_v1.plant_intel._rescue_trap",
                       return_value=("spikeweed", 1.0, 0, False)):
            board = board_classifier.classify_board(frame, {
                "cells": [cell(1, "base_only"), cell(2, "base_and_upper"), cell(3, "ground_occupied"),
                          cell(4, "empty"), cell(4, "base_only", 6)],
            })
        self.assertEqual(board["cells_by_row"][1], [])
        self.assertEqual(board["cells_by_row"][2][0]["plant_id"], "unknown_plant")
        self.assertEqual(board["cells_by_row"][2][0]["occupancy"], "ground_occupied")
        self.assertIsNone(board["cells_by_row"][2][0]["carrier"])
        self.assertEqual(board["cells_by_row"][3][0]["plant_id"], "unknown_plant")
        self.assertEqual([p["plant_id"] for p in board["cells_by_row"][4]], ["lily_pad"])

        corrupted = {"cells_by_row": {
            2: [{"plant_id": "repeater", "column": 4, "carrier": "lily_pad"},
                {"plant_id": "lily_pad", "column": 5, "blocker_hp": 300}],
            3: [{"plant_id": "potato_mine", "armed": True, "column": 4},
                {"plant_id": "repeater", "column": 5, "occupancy": "ground_occupied"}],
        }, "row_summary": {2: {"compact_zh": "双发x1+睡莲x1"}, 3: {"compact_zh": "土豆雷x1+双发x1"}}}
        intel = classify_state_metadata({"zombies": {"candidates": [
            {"row": row, "head_xy": [690, 85 * row]} for row in (2, 3)
        ]}})
        defense = evaluate_all_lanes(corrupted, intel)
        for row in (2, 3):
            self.assertEqual(defense["by_row"][row]["verdict"], "deficit")
            self.assertEqual(defense["by_row"][row]["plants_compact_zh"], "")

        clock = _FakeClock()
        provider = mock.Mock()
        coach = AsyncLlmCoach(clock=clock, enabled=True, provider_factory=lambda _: provider)
        self.addCleanup(coach.close)
        engine = ContinuousAdviceEngine(
            clock=clock, perception=_FakePerception([{"row": 2, "head_x": 500}]), threaded_expiry=False,
        )
        self.addCleanup(engine.close)
        source, _, _ = engine.ingest_canonical_rgb(bytes(RGB_BYTES), (800, 600))
        channels = {"zombie_intel": intel, "lane_defense": defense, "plant_intel": board}
        instructions, prompt = coach._build_prompt_parts(source, channels)
        self.assertIn("地形铁律：1/2/5/6路为陆地", instructions)
        self.assertIn("3/4路为水路，严禁推荐土豆雷或地刺", instructions)
        self.assertIn("2路(陆):", prompt)
        self.assertIn("3路(水) 无空睡莲:", prompt)
        self.assertIn("3路(水) 无空睡莲 无已确认部署", prompt)
        self.assertIn("4路(水) 空睡莲x1", coach._build_dynamic_user_prompt(
            source, {"plant_intel": board, "per_lane": {4: [550]}}
        ))
        self.assertEqual(set(llm.LLM_COMPACT_STRATEGY_SCHEMA["properties"]), {"goal", "focus_row", "reason"})

        for count, (row, reason, error) in enumerate((
            (3, "3路放土豆雷解铁桶", "illegal_terrain_water_land_only"),
            (2, "2路放缠绕海草秒杀路障", "illegal_terrain_land_water_only"),
            (4, "4路放地刺", "illegal_terrain_water_land_only"),
            (4, "优先埋土豆雷", "illegal_terrain_water_land_only"),
            (2, "用睡莲拖住路障", "illegal_terrain_land_water_only"),
            (2, "2路挡冰车，地刺放在4路", "illegal_terrain_water_land_only"),
            (None, "1/2/5/6路补睡莲", "illegal_terrain_land_water_only"),
        ), 1):
            with self.subTest(reason=reason):
                strategy = {"goal": "defend_lane" if row else "develop_economy", "focus_row": row, "reason": reason}
                self.assertEqual(coach._validate_hypothesis(source, source, strategy), (False, error))
                provider.complete.return_value = {"text": json.dumps(strategy)}
                coach._worker(source, b"", {}, {}, coach._epoch, clock())
                self.assertEqual(coach._rejected_count, count)
                self.assertEqual(coach.status_snapshot()["status"], "rejected_illegal_terrain")
                self.assertIsNone(coach._pending_hypothesis)
                self.assertIsNone(coach.recheck_against_current(source))
        with mock.patch.object(llm, "validate_terrain_advice", return_value=(True, "ok")):
            provider.complete.return_value = {"text": json.dumps({
                "goal": "defend_lane", "focus_row": 4, "reason": "4路补地刺",
            })}
            coach._worker(source, b"", {}, {}, coach._epoch, clock())
        self.assertIsNotNone(coach._pending_hypothesis)
        self.assertIsNone(coach.recheck_against_current(source))
        self.assertEqual(coach.status_snapshot()["status"], "rejected_illegal_terrain")
        for reason in (
            "2路放土豆雷解铁桶，3路已有海草稳守",
            "2路补地刺防冰车，3路已有海草无需增防",
            "2路无海草，4路勿放土豆雷，先留阳",
            "别在4路放土豆雷，2路用单发守线",
            "2路陆地放樱桃炸弹溅射3路",
        ):
            with self.subTest(reason=reason):
                strategy = {"goal": "defend_lane", "focus_row": 2, "reason": reason}
                provider.complete.return_value = {"text": json.dumps(strategy)}
                coach._worker(source, b"", {}, {}, coach._epoch, clock())
                advice = coach.recheck_against_current(source)
                self.assertEqual((advice.reason, advice.possibly_stale), (reason, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
