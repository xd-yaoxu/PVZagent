"""Persistent whole-match read-only advisor runtime composing fast perception, 2.0s watchdog, timeline, and opt-in LLM coach."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
import threading
import time
from typing import Any, Callable
from weakref import WeakValueDictionary, ref

from PIL import Image

from ..resource_runtime import ResourceStrategySupervisor
from ..resource_sidebar import Advice, ReadonlyResourceSidebar, _Presentation, _SidebarFrames
from ..resource_sidebar_tk import _advice_text
from ..snapshot_frames import SnapshotFrames
from ..strategy_review import fresh
from .async_llm_coach import AsyncLlmCoach, VerifiedLlmAdvice
from .fast_perception import (
    CANONICAL_SIZE,
    NO_EVIDENCE_TEXT,
    REPO_ROOT,
    RGB_BYTES,
    FastGeometryNormalizer,
    FastSinglePassPerception,
    PhaseTracker,
    evaluate_local_strategy,
    evaluate_three_channels,
    format_mmss,
)
from .wgc_capture import (
    SUPPORTED_CLIENT_SIZES,
    ContinuousWgcSource,
    OriginalPvzWindowSelector,
    WindowLease,
)
from .zombie_intel import supplementary_snorkels

INPUT_AGE = 1.5
RESULT_AGE = 2.0
POLICY = {"left_warning_x": 200, "other_lane_margin_px": 80}


def _goal_title(goal: str, focus_row: int | None) -> str:
    if goal == "defend_lane" and focus_row is not None:
        return f"关注第 {focus_row} 路防守"
    if goal == "collect_resources":
        return "优先收取地面阳光"
    if goal == "develop_economy":
        return "继续发展经济"
    return "稳固当前阵型"


def _conflicts_with_defense(goal, focus_row, defense):
    if not defense:
        return False
    if goal == "defend_lane" and defense.get("by_row", {}).get(focus_row, {}).get("verdict") == "sufficient":
        return True
    rows = defense.get("needs_defense_rows", ())
    if rows:
        return (goal, focus_row) != ("defend_lane", rows[0])
    return goal == "defend_lane"


def _local_advice_text(advice):
    return "维持当前阵型" if advice.goal == "hold_formation" else _advice_text(advice)


@dataclass(frozen=True)
class _Arm:
    selection: object
    epoch: int
    generation: int
    deadline: float


class _ExpiryWatchdog:
    """Independent 2.0s OS timer thread that withdraws expired advice even if capture/perception stalls."""

    def __init__(self, registry, retire: Callable[[object], bool], *, threaded: bool = True):
        self.registry = registry
        self.condition = registry._condition
        self.retire = retire
        self.armed: _Arm | None = None
        self.closed = False
        self.retire_count = 0
        self.thread = None
        if threaded:
            self.thread = threading.Thread(target=self._watch, name="pvz-advisor-expiry-watchdog", daemon=True)
            self.thread.start()

    def arm_locked(self, selection: object, presentation: object, deadline: float) -> None:
        token = presentation.token
        prior = self.armed
        if (
            prior is not None
            and prior.selection is selection
            and prior.epoch == token[1]
            and prior.generation == token[0]
            and prior.deadline == deadline
        ):
            return
        self.armed = _Arm(selection, token[1], token[0], deadline)
        self.condition.notify_all()

    def disarm_locked(self) -> None:
        self.armed = None
        self.condition.notify_all()

    def due_locked(self, expected: _Arm | None, now: float) -> bool:
        if (
            self.closed
            or expected is None
            or expected is not self.armed
            or expected.epoch != self.registry._epoch
        ):
            return False
        if type(now) in (int, float) and isfinite(now) and now < expected.deadline:
            return False
        self.disarm_locked()
        if not self.retire(expected.selection):
            return False
        self.retire_count += 1
        return True

    def enforce_locked(self) -> bool:
        return self.due_locked(self.armed, self.registry.clock())

    def _watch(self) -> None:
        with self.condition:
            while not self.closed:
                arm = self.armed
                if arm is None:
                    self.condition.wait()
                    continue
                now = self.registry.clock()
                if not self.due_locked(arm, now):
                    if arm is not self.armed:
                        continue
                    self.condition.wait(max(0.0, arm.deadline - now))

    def stop_locked(self) -> None:
        self.closed = True
        self.disarm_locked()

    def join(self) -> bool:
        if self.thread is None:
            return True
        self.thread.join(timeout=0.5)
        return not self.thread.is_alive()


class _AdvisorFrames(_SidebarFrames):
    def admit_frame(self, state_id: str, monotonic_ns: int):
        stamp = monotonic_ns / 1e9
        SnapshotFrames.put(
            self,
            {
                "_source_binding": (state_id, monotonic_ns),
                "_continuity_epoch": self._epoch,
                "_continuity_capture_valid": True,
            },
        )
        self._last_capture = stamp
        self._gap_reported = False
        return self.take()

    def _break_locked(self, reason, state_id=None):
        had_current = self._sidebar._current is not None
        super()._break_locked(reason, state_id)
        self._sidebar.current_evidence_text = NO_EVIDENCE_TEXT
        self._sidebar.current_reason_text = ""
        if had_current:
            self._sidebar.withdrawn_count += 1
        if self._sidebar.expiry is not None:
            self._sidebar.expiry.disarm_locked()
        if self._listener is not None:
            self._listener(reason)


class _AdvisorSidebar(ReadonlyResourceSidebar):
    def __init__(self, clock: Callable[[], float]):
        super().__init__()
        self._clock = clock
        self._history_epoch = None
        self._direct_tag = object()
        self.expiry: _ExpiryWatchdog | None = None
        self._active_selection = None
        self.presentations_accepted = 0
        self.duplicate_coalesced_count = 0
        self.withdrawn_count = 0
        self.current_evidence_text = NO_EVIDENCE_TEXT
        self.current_reason_text = ""
        self.rich_history: deque[str] = deque(maxlen=40)
        self._last_rich_key: tuple[str, int | None] | None = None

    def _bind(self):
        with self._bind_lock:
            if self._registry is not None:
                raise RuntimeError("already bound")
            self._registry = _AdvisorFrames(self)
            self._registry.clock = self._clock
            return self._registry

    @staticmethod
    def _deadline(result) -> float:
        captured, published_expiry = result.token[4:6]
        return min(captured + RESULT_AGE, published_expiry, result.source_time + result.review_window)

    def _is_current_locked(self, result, now: float) -> bool:
        if result is not None and result.reviewer is self._direct_tag:
            return self._is_direct_current_locked(result, now)
        return (
            result is not None
            and type(now) in (int, float)
            and isfinite(now)
            and now < self._deadline(result)
            and super()._is_current_locked(result, now)
        )

    def _is_direct_current_locked(self, result: _Presentation, now: float) -> bool:
        registry = self._registry
        if self.expiry.closed or registry._source_ended or registry._error is not None or registry._published is None:
            return False
        generation, state, captured, expires = registry._published
        return (
            generation == registry._generation
            and fresh(state, now)
            and captured <= now < expires
            and now < self._deadline(result)
            and (generation, registry._epoch, state["run_id"], state["state_id"],
                 captured, expires, registry._strategy_context_version) == result.token
        )

    def present_direct_locked(self, strategy: dict[str, Any], state: dict[str, Any]) -> bool:
        """Present a local strategy from the current registry publication."""
        if self.expiry.enforce_locked() or self.expiry.closed:
            return False
        selected = self._active_selection()
        registry = self._registry
        if selected is None or registry._published is None:
            return False
        generation, current, captured, expires = registry._published
        if current["state_id"] != state["state_id"] or current["continuity"]["epoch"] != registry._epoch:
            return False
        token = (generation, registry._epoch, state["run_id"], state["state_id"],
                 captured, expires, registry._strategy_context_version)
        deadline = min(captured + RESULT_AGE, expires)
        advice = Advice(strategy["goal"], strategy.get("focus_row"), strategy["source_state_id"],
                        state["state_id"], deadline)
        result = _Presentation(self._binding, advice, token, 0, self._direct_tag, captured, 30.0)
        if not self._is_current_locked(result, self._clock()):
            return False
        previous = tuple(self._history)
        self._current = result
        if not self._history or self._history[-1] != advice:
            self._history.append(advice)
        self._record_presentation_locked(selected, result, deadline, previous)
        return True

    def _record_presentation_locked(self, selected, result, deadline, previous) -> None:
        self.presentations_accepted += 1
        self.expiry.arm_locked(selected, result, deadline)
        epoch = result.token[1]
        old = previous[-1] if previous else None
        if (
            old is not None
            and self._history_epoch == epoch
            and (old.goal, old.focus_row) == (result.advice.goal, result.advice.focus_row)
        ):
            self._history.clear()
            self._history.extend(previous)
            self.duplicate_coalesced_count += 1
        else:
            self._history_epoch = epoch

    def _present(self, reviewer, result):
        with self._registry._condition:
            if self.expiry.enforce_locked() or self.expiry.closed:
                return
            selected = self._active_selection()
            if selected is None:
                return
            if self._registry._published is not None:
                state = self._registry._published[1]
                defense = state.get("lane_defense") or {}
                local_hold = result.advice.goal == "hold" and (
                    defense.get("all_threatened_sufficient") or defense.get("unknown_threat_rows")
                )
                # Publish the assessed formation hold via the direct path so
                # it keeps its specific reason and a single history entry.
                if local_hold or _conflicts_with_defense(result.advice.goal, result.advice.focus_row, defense):
                    return
            deadline = self._deadline(result)
            if deadline < result.advice.expires_monotonic:
                result = replace(result, advice=replace(result.advice, expires_monotonic=deadline))
            previous = tuple(self._history)
            super()._present(reviewer, result)
            if self._current is not result:
                return
            self._record_presentation_locked(selected, result, deadline, previous)


@dataclass
class _SessionSelection:
    version: int
    client_size: tuple[int, int]
    last_serial: int = 0
    last_ns: int = 0


class ContinuousAdviceEngine:
    """In-memory continuous sidebar runtime with 2.0s watchdog, duplicate coalescing, and fail-closed expiry."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        perception: Any = None,
        root: Path = REPO_ROOT,
        threaded_expiry: bool = True,
        policy: dict[str, int] | None = None,
    ):
        self.root = Path(root).resolve()
        self.clock = clock
        self.policy = dict(policy or POLICY)
        self.perception = perception
        self.sidebar = _AdvisorSidebar(clock)
        self.registry = self.sidebar._bind()
        self.journal: deque[dict[str, Any]] = deque(maxlen=100)
        self.supervisor = ResourceStrategySupervisor(
            self.registry,
            self._supervisor_strategy,
            policy=self.policy,
            record_sink=self.journal.append,
            clock=clock,
            max_requests=3,
            min_request_interval=5.0,
            sidebar=self.sidebar,
        )
        # Allow unlimited continuous local evaluations across a 5-20 minute match
        self.supervisor.max_requests = 10_000_000
        self.supervisor.min_request_interval = 0.0
        self.supervisor.reviewer.max_requests = 10_000_000
        self.supervisor.reviewer.min_request_interval = 0.0
        self.registry.bind_invalidator(self.supervisor.reviewer.invalidate)
        self._selection: _SessionSelection | None = None
        self._version = 0
        self._closed = False
        self.expiry = _ExpiryWatchdog(self.registry, self._expire_locked, threaded=threaded_expiry)
        self.sidebar.expiry = self.expiry
        self.sidebar._active_selection = lambda: self._selection
        self.published_frames = 0
        self.stale_rejected = 0
        self._engine_uid = f"{id(self) & 0xFFFF:04x}"

    def _supervisor_strategy(self, state):
        proposal = evaluate_local_strategy(state, self.policy)
        if proposal["strategy"]["goal"] == "hold_formation":
            proposal["strategy"]["goal"] = "hold"
        return proposal

    def _expire_locked(self, expected) -> bool:
        if self._selection is not expected:
            return False
        # Withdraw current expired advice and advance continuity epoch without discarding
        # a brand-new in-flight observation that was captured after the expired frame.
        self.registry._break_locked("deadline_expired")
        return True

    def _retire_locked(self, reason: str) -> None:
        if self._selection is not None:
            self._selection = None
            self.registry._break_locked(reason)

    def open_selection(self, client_size: tuple[int, int]) -> _SessionSelection:
        if client_size not in SUPPORTED_CLIENT_SIZES:
            raise ValueError(f"unsupported_client_size:{client_size}")
        with self.registry._condition:
            if self._closed:
                raise RuntimeError("engine_closed")
            self._retire_locked("selection_replaced")
            self.registry._source_run = None
            self.registry._last_capture = None
            self.registry._gap_reported = False
            self._version += 1
            self._selection = _SessionSelection(version=self._version, client_size=client_size)
            return self._selection

    def source_lost(self, reason: str = "source_lost") -> None:
        with self.registry._condition:
            self._retire_locked(reason)

    def ingest_canonical_rgb(
        self,
        rgb_bytes: bytes,
        client_size: tuple[int, int],
        *,
        rel_time_text: str = "00:00",
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Advice | None]:
        """Run single-pass perception on canonical 800x600 RGB, publish if fresh, and settle advice."""
        if len(rgb_bytes) != RGB_BYTES:
            raise ValueError("canonical_800x600_rgb_required")
        if self.perception is None:
            self.perception = FastSinglePassPerception(self.root)

        with self.registry._condition:
            self.expiry.enforce_locked()
            now = self.clock()
            stamp_ns = round(now * 1e9)
            if (
                self._selection is None
                or self._selection.client_size != client_size
                or (self._selection.last_ns and not 0 < (stamp_ns - self._selection.last_ns) / 1e9 <= 2.0)
            ):
                self._retire_locked("epoch_rollover")
                self.registry._source_run = None
                self.registry._last_capture = None
                self.registry._gap_reported = False
                self._version += 1
                self._selection = _SessionSelection(version=self._version, client_size=client_size)
            sel = self._selection
            sel.last_serial += 1
            sel.last_ns = stamp_ns
            serial = sel.last_serial
            state_id = f"live/{self._engine_uid}/{sel.version}/{serial}"
            ticket = self.registry.admit_frame(state_id, stamp_ns)
            capture_epoch = ticket.item["_continuity_epoch"]

        stamp = stamp_ns / 1e9
        run_id = f"live-source/{self._engine_uid}/{sel.version}"
        digest = sha256(rgb_bytes).hexdigest()
        evidence = {"image": f"memory:{state_id}", "image_sha256": digest, "source_kind": "wgc_client_memory"}
        capture_time = f"monotonic-ns:{stamp_ns}"

        with Image.frombytes("RGB", CANONICAL_SIZE, rgb_bytes) as img:
            state = self.perception.observe(
                img,
                run_id=run_id,
                state_id=state_id,
                capture_time=capture_time,
                frame_time_seconds=stamp,
                evidence=evidence,
                tracking_epoch=capture_epoch,
            )

        with self.registry._condition:
            self.expiry.enforce_locked()
            if self._selection is not sel or ticket.generation != self.registry._generation:
                self.stale_rejected += 1
                return None, None, None
            now_after = self.clock()
            if not 0 <= now_after - stamp < RESULT_AGE:
                self._retire_locked("observation_expired")
                self.stale_rejected += 1
                return None, None, None
            if capture_epoch != self.registry._epoch:
                # Only a prior presentation expiring may carry an in-flight frame across the break.
                if (
                    self.registry._epoch != capture_epoch + 1
                    or self.registry._continuity_events[-1]["reason"] != "deadline_expired"
                ):
                    self.stale_rejected += 1
                    return None, None, None
                ticket = self.registry.admit_frame(state_id, stamp_ns)
            published = self.registry.publish_observation(
                ticket,
                state,
                captured_at=stamp,
                now=now_after,
                max_age=RESULT_AGE,
                input_age_ms=0.0,
            )
            if (
                not isinstance(published, dict)
                or published.get("readonly_runtime", {}).get("publication_at_completion") != "published"
            ):
                self._retire_locked("nonplaying_or_rejected")
                return published if isinstance(published, dict) else None, None, None
            self.published_frames += 1

        ch_info = evaluate_three_channels(self.root, published, frame_index=self.published_frames)
        local_strategy: dict[str, Any] | None
        try:
            local_strategy = evaluate_local_strategy(published, self.policy)["strategy"]
            local_reason = local_strategy["reason"]
        except Exception:
            local_strategy = None
            local_reason = ""

        req_before = self.supervisor.requests
        hist_before = len(self.sidebar._history)
        self.supervisor.poll()
        future = self.supervisor.reviewer._future
        if (self.supervisor.requests > req_before or self.supervisor.reviewer._active) and future is not None:
            try:
                future.result(timeout=1.0)
            except Exception:
                pass
            self.supervisor.poll()

        with self.registry._condition:
            self.expiry.enforce_locked()
            if ticket.generation != self.registry._generation or published["continuity"]["epoch"] != self.registry._epoch:
                return published, ch_info, None
            current = self.sidebar._current
            is_current = self.sidebar._is_current_locked(current, self.clock())
            defense = ch_info.get("lane_defense") or {}
            defense_reviewed = (defense.get("all_threatened_sufficient") or defense.get("needs_defense_rows")
                                or defense.get("unknown_threat_rows"))
            supplemental = ch_info.get("confirmed_vehicles", 0) > 0 or bool(
                supplementary_snorkels(ch_info.get("zombie_intel") or {})
            )
            if local_strategy is not None and ch_info.get("uncertain_threats", 0) == 0:
                current_key = (current.advice.goal, current.advice.focus_row) if is_current else None
                local_key = local_strategy["goal"], local_strategy["focus_row"]
                if (defense_reviewed and current_key != local_key) or (supplemental and not is_current):
                    self.sidebar.present_direct_locked(local_strategy, published)
            if self.sidebar._is_current_locked(self.sidebar._current, self.clock()):
                self.sidebar.current_evidence_text = ch_info["summary_text"]
                self.sidebar.current_reason_text = local_reason
                advice = self.sidebar._current.advice
                key = (advice.goal, advice.focus_row)
                if len(self.sidebar._history) != hist_before and key != self.sidebar._last_rich_key:
                    self.sidebar._last_rich_key = key
                    self.sidebar.rich_history.append(
                        f"[{rel_time_text}] [本地即时] {_local_advice_text(advice)} — {local_reason}"
                    )
                return published, ch_info, advice
            if self.sidebar._current is not None:
                self.sidebar._current = None
                self.sidebar.withdrawn_count += 1
            self.sidebar.current_evidence_text = NO_EVIDENCE_TEXT
            self.sidebar.current_reason_text = ""
            self.expiry.disarm_locked()
            return published, ch_info, None

    def current_published_state(self) -> dict[str, Any] | None:
        return self.registry.current(self.clock())

    def close(self) -> None:
        with self.registry._condition:
            if self._closed:
                return
            self._retire_locked("engine_closed")
            self._closed = True
            self.expiry.stop_locked()
            self.registry.close()
        self.supervisor.stop()
        self.expiry.join()


class AdvisorController:
    """High-level controller driving persistent whole-match monitoring for the single right-hand sidebar."""

    def __init__(
        self,
        *,
        root: Path = REPO_ROOT,
        selector: OriginalPvzWindowSelector | None = None,
        wgc_source: ContinuousWgcSource | None = None,
        normalizer: FastGeometryNormalizer | None = None,
        engine: ContinuousAdviceEngine | None = None,
        phase_tracker: PhaseTracker | None = None,
        llm_coach: AsyncLlmCoach | None = None,
        clock: Callable[[], float] = time.monotonic,
        min_dwell_s: float = 3.5,
        display_stale_s: float = 4.5,
    ):
        self.root = Path(root).resolve()
        self.clock = clock
        self.selector = selector or OriginalPvzWindowSelector()
        self.wgc = wgc_source or ContinuousWgcSource()
        self.normalizer = normalizer
        self.engine = engine or ContinuousAdviceEngine(clock=clock, root=self.root, threaded_expiry=True)
        self.phase_tracker = phase_tracker or PhaseTracker()
        self.llm_coach = llm_coach or AsyncLlmCoach(clock=clock, enabled=False)
        self.min_dwell_s = max(0.0, float(min_dwell_s))
        self.display_stale_s = max(2.0, float(display_stale_s))

        self._lock = threading.RLock()
        self._lease: WindowLease | None = None
        self._monitoring = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._status_text = "未选窗 · 点击「选窗(5秒)」后切到原版 PVZ 窗口"
        self._mode_state = "unselected"  # unselected | ready | monitoring | waiting_game | paused | error
        self._last_frame_ms = 0.0
        self._last_norm_mode = "-"
        self._frames_processed = 0
        self._last_wgc_error = ""
        self._last_llm_history_key: str | None = None

        # Visual stabilization state (prevents 1-frame flicker & rapid ping-pong while remaining fail-closed)
        self._display_key: tuple[str, int | None] | None = None
        self._display_title: str = "暂无当前建议"
        self._display_reason: str = ""
        self._display_evidence: str = NO_EVIDENCE_TEXT
        self._display_since: float = 0.0
        self._display_last_fresh_at: float = 0.0
        self._display_result_deadline: float | None = None
        self._last_obs_at: float = 0.0
        self._consecutive_uncertain: int = 0
        self._stable_history: deque[str] = deque(maxlen=40)
        self._llm_history: deque[str] = deque(maxlen=40)
        self._last_stable_hist_key: tuple[str, int | None] | None = None

    def _clear_display_locked(self) -> None:
        self._display_key = None
        self._display_title = "暂无当前建议"
        self._display_reason = ""
        self._display_evidence = NO_EVIDENCE_TEXT
        self._display_since = 0.0
        self._display_last_fresh_at = 0.0
        self._display_result_deadline = None
        self._last_obs_at = 0.0
        self._consecutive_uncertain = 0

    def _update_display_from_frame(
        self,
        *,
        ch_info: dict[str, Any] | None,
        advice: Advice | None,
        reason_text: str,
        rel_time_text: str,
        now: float,
    ) -> None:
        llm_enabled = bool(self.llm_coach.status_snapshot().get("enabled"))
        with self._lock:
            if ch_info is None:
                self._clear_display_locked()
                return
            self._last_obs_at = now
            lane_defense = ch_info.get("lane_defense") or {}
            display_conflict = (
                self._display_key is not None and _conflicts_with_defense(*self._display_key, lane_defense)
            )
            if advice is not None and _conflicts_with_defense(advice.goal, advice.focus_row, lane_defense):
                advice = None
            if advice is None:
                self._consecutive_uncertain += 1
                self._display_evidence = ch_info["summary_text"]
                # Tolerate at most 1 transient uncertain frame during continuous active play;
                # withdraw immediately on 2 consecutive uncertain frames or when stale budget is reached.
                if (
                    self._display_key is not None
                    and (
                        display_conflict
                        or self._display_result_deadline is not None
                        or self._consecutive_uncertain >= 2
                        or (now - self._display_last_fresh_at) >= self.display_stale_s
                    )
                ):
                    self._clear_display_locked()
                return

            self._consecutive_uncertain = 0
            self._display_last_fresh_at = now
            self._display_evidence = ch_info["summary_text"]
            new_key = (advice.goal, advice.focus_row)
            by_lane = ch_info.get("zombie_intel", {}).get("by_lane", {})
            primary = by_lane.get(ch_info.get("priority_row"), {}).get("primary_entity", {})
            leftmost_threat = ch_info.get("leftmost_threat_x")
            has_vehicles = ch_info.get("confirmed_vehicles", 0) > 0
            is_high_threat = (
                has_vehicles
                or (leftmost_threat is not None and leftmost_threat <= 200)
                or (
                    primary.get("species_id") in ("zomboni", "buckethead_zombie", "gargantuar", "football_zombie")
                    and primary["x"] <= 560
                )
            )
            leftmost_head = ch_info.get("leftmost_head_x")
            is_urgent = (leftmost_head is not None and leftmost_head <= 200) or is_high_threat
            focus_lane_gone = (
                self._display_key is not None
                and self._display_key[0] == "defend_lane"
                and self._display_key[1] is not None
                and not ch_info.get("per_lane", {}).get(self._display_key[1])
                and not by_lane.get(self._display_key[1])
            )
            can_switch = (
                self._display_key is None
                or new_key == self._display_key
                or is_urgent
                or focus_lane_gone
                or display_conflict
                or (now - self._display_since) >= self.min_dwell_s
            )
            if can_switch:
                if new_key != self._display_key:
                    self._display_since = now
                self._display_key = new_key
                self._display_result_deadline = advice.expires_monotonic if has_vehicles or lane_defense else None
                self._display_title = _local_advice_text(advice)
                self._display_reason = reason_text
                if not llm_enabled and new_key != self._last_stable_hist_key:
                    self._last_stable_hist_key = new_key
                    self._stable_history.append(
                        f"[{rel_time_text}] [本地即时] {self._display_title} — {reason_text}"
                    )

    def bind_verified_lease(self, lease: WindowLease) -> None:
        with self._lock:
            if self._lease is not None and self._lease is not lease:
                self.selector.release(self._lease)
            self._lease = lease
            self._mode_state = "ready"
            self._status_text = f"已绑定原版 PVZ 窗口 (PID={lease.pid}) · 点击「开始监测」后切回游戏"

    def try_select_foreground_now(self) -> tuple[bool, str]:
        status, lease = self.selector.inspect_foreground()
        if status == "ok" and lease is not None:
            self.bind_verified_lease(lease)
            return True, f"已核验原版 PVZ-Windowed.exe (PID={lease.pid})，可点击「开始监测」"
        messages = {
            "no_foreground": "未检测到前台窗口，请重试并切回原版游戏",
            "outside_scope": "当前前台不是原版 PVZ-Windowed.exe（请勿选择 Magpie 放大窗）",
            "minimized_or_hidden": "目标窗口处于最小化或不可见状态",
            "window_closed": "窗口已关闭",
        }
        return False, messages.get(status, f"无法核实窗口身份({status})，请重试")

    def start_monitoring(self) -> tuple[bool, str]:
        with self._lock:
            if self._lease is None or self._lease.closed:
                return False, "请先点击「选窗(5秒)」绑定原版游戏窗口"
            if self._monitoring:
                return True, "监测已在运行中"
            self._monitoring = True
            self._stop_event.clear()
            self._mode_state = "waiting_game"
            self._status_text = "正在加载感知模型并等待切回原版游戏..."
            self.llm_coach.warmup_connection_async()
            self._worker = threading.Thread(target=self._loop, name="pvz-readonly-advisor-loop", daemon=True)
            self._worker.start()
            return True, "已启动整局只读监测，请切回原版游戏窗口"

    def pause_monitoring(self) -> None:
        with self._lock:
            self._monitoring = False
            self._stop_event.set()
            worker = self._worker
            self._worker = None
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        self.wgc.close()
        self.engine.source_lost("operator_paused")
        self.llm_coach.invalidate("operator_paused")
        self.phase_tracker.record_gap_event(self.clock(), "人工暂停监测")
        with self._lock:
            self._clear_display_locked()
            if self._lease is not None and not self._lease.closed:
                self._mode_state = "paused"
                self._status_text = "已暂停监测 · 当前建议已撤下，历史保留"
            else:
                self._mode_state = "unselected"

    def _ensure_models_loaded(self) -> None:
        if self.normalizer is None:
            self.normalizer = FastGeometryNormalizer(self.root)
        if self.engine.perception is None:
            self.engine.perception = FastSinglePassPerception(self.root)

    def _loop(self) -> None:
        try:
            self._ensure_models_loaded()
        except Exception as exc:
            with self._lock:
                self._mode_state = "error"
                self._status_text = f"感知基线加载失败：{type(exc).__name__}"
                self._monitoring = False
            return

        was_away = False
        while not self._stop_event.is_set():
            with self._lock:
                lease = self._lease
            if lease is None or lease.closed:
                break

            alive, is_fg, st = self.selector.is_lease_alive(lease)
            if not alive:
                self.wgc.close()
                self.engine.source_lost(f"window_lost:{st}")
                self.llm_coach.invalidate("window_lost")
                self.phase_tracker.record_gap_event(self.clock(), "游戏窗口已关闭")
                with self._lock:
                    self._clear_display_locked()
                    self._mode_state = "unselected"
                    self._status_text = "所选原版 PVZ 窗口已关闭或进程已退出，请重新选窗"
                    self._monitoring = False
                return

            if not is_fg:
                if self.wgc.is_open:
                    self.wgc.close()
                if not was_away:
                    self.engine.source_lost("game_not_foreground")
                    self.llm_coach.invalidate("game_not_foreground")
                    self.phase_tracker.record_gap_event(self.clock(), "切出游戏/暂停")
                    was_away = True
                with self._lock:
                    self._clear_display_locked()
                    self._mode_state = "waiting_game"
                    diag_tail = f"（上次状态：{self._last_wgc_error}）" if (self._frames_processed == 0 and self._last_wgc_error) else ""
                    self._status_text = f"游戏已切出或暂停 · 切回游戏自动恢复{diag_tail}"
                self._stop_event.wait(0.15)
                continue

            # Game is foreground & alive: ensure WGC session is open
            if not self.wgc.is_open:
                try:
                    self.wgc.open(lease)
                    if self.normalizer is not None:
                        self.normalizer.invalidate_cache()
                    if was_away:
                        self.phase_tracker.record_gap_event(self.clock(), "切回游戏恢复监测")
                        was_away = False
                except Exception as exc:
                    self.wgc.close()
                    self.engine.source_lost("wgc_open_retry")
                    err_tag = (str(exc).strip() or type(exc).__name__)[:60]
                    with self._lock:
                        self._last_wgc_error = f"open:{err_tag}"
                        self._mode_state = "waiting_game"
                        self._status_text = f"等待原版游戏画面就绪... ({self._last_wgc_error})"
                    self._stop_event.wait(0.25)
                    continue

            t_frame = time.perf_counter()
            try:
                cap = self.wgc.read_frame(timeout=3.0)
            except Exception as exc:
                self.wgc.close()
                self.engine.source_lost("wgc_read_reset")
                self.llm_coach.invalidate("wgc_read_reset")
                err_tag = (str(exc).strip() or type(exc).__name__)[:60]
                with self._lock:
                    self._last_wgc_error = f"read:{err_tag}"
                    self._mode_state = "waiting_game"
                    self._status_text = f"等待原版游戏帧... ({self._last_wgc_error})"
                self._stop_event.wait(0.15)
                continue

            if self._stop_event.is_set():
                break

            rgb, reason, norm_diag = self.normalizer.normalize(cap.bgra, cap.client_size, cap.rect_signature)
            cap.bgra.fill(0)
            if rgb is None or reason is not None:
                self.engine.source_lost(f"scene_or_geom:{reason}")
                self.llm_coach.invalidate("scene_or_geom")
                with self._lock:
                    self._clear_display_locked()
                    self._mode_state = "waiting_game"
                    self._status_text = "当前非对局画面（暂停/选卡/过场）· 已撤下建议，继续对局自动恢复"
                self._stop_event.wait(0.20)
                continue

            rgb_bytes = rgb.tobytes()
            rgb.fill(0)
            now = self.clock()
            rel_txt = self.phase_tracker.elapsed_text(now)
            published, ch_info, advice = self.engine.ingest_canonical_rgb(
                rgb_bytes, cap.client_size, rel_time_text=rel_txt
            )
            elapsed_ms = (time.perf_counter() - t_frame) * 1000.0

            if ch_info is not None and published is not None:
                with self.engine.registry._condition:
                    cur_reason = self.engine.sidebar.current_reason_text
                self._update_display_from_frame(
                    ch_info=ch_info,
                    advice=advice,
                    reason_text=cur_reason,
                    rel_time_text=rel_txt,
                    now=self.clock(),
                )
                self.phase_tracker.update(ch_info, self.clock(), advice_added=advice is not None)
                phase_snap = self.phase_tracker.snapshot(self.clock())
                # Opt-in async LLM dispatch (0 network calls when disabled)
                self.llm_coach.maybe_dispatch(
                    state=published,
                    rgb_bytes=rgb_bytes,
                    phase_info=phase_snap,
                    channel_info=ch_info,
                )
                verified_llm = self.llm_coach.recheck_against_current(self.engine.current_published_state())
                if verified_llm is not None:
                    key = f"{verified_llm.source_state_id}:{verified_llm.reviewed_state_id}"
                    if key != self._last_llm_history_key:
                        self._last_llm_history_key = key
                        stale_tag = "（可能过时）" if verified_llm.possibly_stale else ""
                        with self._lock:
                            self._llm_history.append(
                                f"[{rel_txt}] [{verified_llm.model} · {verified_llm.latency_s:.1f}s]{stale_tag} "
                                f"{_goal_title(verified_llm.goal, verified_llm.focus_row)} — {verified_llm.reason}"
                            )
                with self._lock:
                    self._frames_processed += 1
                    self._last_wgc_error = ""
                    self._last_frame_ms = elapsed_ms
                    self._last_norm_mode = str(norm_diag.get("mode", "-"))
                    self._mode_state = "monitoring"
                    self._status_text = (
                        f"监测中 · {elapsed_ms / 1000.0:.2f}s/帧 ({self._last_norm_mode}) · "
                        f"已观测 {self._frames_processed} 帧"
                    )
            else:
                self.llm_coach.recheck_against_current(None)
                with self._lock:
                    self._clear_display_locked()
                    self._mode_state = "waiting_game"
                    self._status_text = "当前非对局或帧不确定 · 已撤下当前建议"

            # Yield briefly so Tk & watchdog stay responsive without adding needless lag
            self._stop_event.wait(0.05)

        self.wgc.close()

    def ui_snapshot(self) -> dict[str, Any]:
        """Thread-safe read-only projection for the Tk main thread (120ms poll)."""
        now = self.clock()
        self.engine.sidebar.read()  # Keep watchdog & registry expiry enforced
        with self.engine.registry._condition:
            wd_count = self.engine.expiry.retire_count
            withdrawn_count = self.engine.sidebar.withdrawn_count

        llm = self.llm_coach.status_snapshot()
        phase = self.phase_tracker.snapshot(now)

        with self._lock:
            # Fail-closed stale guard: if no fresh valid observation/advice arrived within display_stale_s, withdraw!
            if self._display_key is not None and (
                (now - self._display_last_fresh_at) >= self.display_stale_s
                or (self._display_result_deadline is not None and now >= self._display_result_deadline)
            ):
                self._clear_display_locked()

            if llm["enabled"]:
                # When LLM tactical advice is enabled, hide local-instant advice & history cleanly.
                live_obs_ok = (
                    self._display_evidence != NO_EVIDENCE_TEXT
                    and self._last_obs_at > 0.0
                    and (now - self._last_obs_at) < self.display_stale_s
                )
                verified_llm: VerifiedLlmAdvice | None = llm["advice"] if live_obs_ok else None
                if verified_llm is not None:
                    has_active = True
                    title_out = _goal_title(verified_llm.goal, verified_llm.focus_row)
                    age_s = max(0.0, now - verified_llm.verified_at)
                    stale_tag = "（可能过时）" if verified_llm.possibly_stale else ""
                    display_meta = (
                        f"[{verified_llm.model} · {verified_llm.latency_s:.1f}s]{stale_tag} · 已展示 {age_s:.1f}s"
                    )
                    unc = "；".join(verified_llm.uncertainties)
                    reason_out = verified_llm.reason + (f"\n不确定项：{unc}" if unc else "")
                else:
                    has_active = False
                    if live_obs_ok and llm["inflight"]:
                        title_out = "等待 LLM 建议..."
                        display_meta = f"正在请求 {llm['model']}（已用 {llm['waiting_s']:.1f}s）"
                    else:
                        title_out = "暂无 LLM 建议"
                        display_meta = "当前无有效 LLM 建议（暂停、切窗或等待下一轮返回）"
                    reason_out = ""
                history_out = list(self._llm_history)
                llm_out = {**llm, "advice": None}
            else:
                has_active = self._display_key is not None
                dwell_s = max(0.0, now - self._display_since) if has_active else 0.0
                fresh_age_s = max(0.0, now - self._display_last_fresh_at) if has_active else 0.0
                title_out = self._display_title
                display_meta = (
                    f"持续有效 · 已展示 {dwell_s:.1f}s（最近确认 {fresh_age_s:.1f}s 前）"
                    if has_active else "当前无有效建议（暂停、切窗或等待新帧）"
                )
                reason_out = self._display_reason
                history_out = list(self._stable_history)
                llm_out = {**llm, "advice": None}

            return {
                "mode_state": self._mode_state,
                "status_text": self._status_text,
                "has_lease": self._lease is not None and not self._lease.closed,
                "monitoring": self._monitoring,
                "frames_processed": self._frames_processed,
                "last_frame_ms": self._last_frame_ms,
                "last_norm_mode": self._last_norm_mode,
                "has_active_advice": has_active,
                "current_advice_title": title_out,
                "current_advice_meta": display_meta,
                "current_reason_text": reason_out,
                "evidence_text": self._display_evidence,
                "rich_history": history_out,
                "watchdog_retire_count": wd_count,
                "withdrawn_count": withdrawn_count,
                "phase": phase,
                "llm": llm_out,
            }

    def close(self) -> None:
        self.pause_monitoring()
        self.llm_coach.close()
        self.engine.close()
        with self._lock:
            if self._lease is not None:
                self.selector.release(self._lease)
                self._lease = None
