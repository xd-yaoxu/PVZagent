"""Optional in-memory sidebar for committed, non-actionable resource advice.

Each instance is single-use and bound to one resource registry. A display
adapter must pull on its own UI dispatch, including when no frames arrive,
and drop its projection on detach. It must not queue a snapshot as a write.
"""
from collections import deque
from dataclasses import dataclass
import threading

from .resource_threat_v2 import ThreatCheckedContinuityFrames
from .strategy_review import fresh


@dataclass(frozen=True)
class Advice:
    goal: str
    focus_row: int | None
    source_state_id: str
    reviewed_state_id: str
    expires_monotonic: float


@dataclass(frozen=True)
class SidebarSnapshot:
    current: Advice | None
    history: tuple[Advice, ...]


@dataclass(frozen=True)
class _Presentation:
    binding: object
    advice: Advice
    token: tuple
    version: int
    reviewer: object
    source_time: float
    review_window: float


class _SidebarFrames(ThreatCheckedContinuityFrames):
    def __init__(self, sidebar):
        super().__init__()
        self._sidebar = sidebar

    def _break_locked(self, reason, state_id=None):
        listener = super()._break_locked(reason, state_id)
        self._sidebar._current = None
        return listener


class ReadonlyResourceSidebar:
    """Only pass via run_resource_readonly(sidebar=...). No UI or worker is started."""

    def __init__(self):
        self._bind_lock = threading.Lock()
        self._registry = None
        self._binding = object()
        self._current = None
        self._history = deque(maxlen=20)

    def _bind(self):
        with self._bind_lock:
            if self._registry is not None:
                raise RuntimeError('Sidebar instances belong to one run')
            self._registry = _SidebarFrames(self)
            return self._registry

    def _prepare(self, reviewer, review, ticket):
        # Keep the exact token that review() committed, before the historical
        # sink runs. Never generate a new token for a delayed presentation.
        with self._registry._condition:
            commit = reviewer._commit
            if commit is None or commit[0] is not ticket:
                return None
            _, token, version, source_time = commit
            strategy = review['strategy']
            advice = Advice(strategy['goal'], strategy.get('focus_row'),
                            review['source_state_id'], token[3], token[5])
            return _Presentation(self._binding, advice, token, version, reviewer, source_time, reviewer.review_window)

    def _present(self, reviewer, result):
        with self._registry._condition:
            now = reviewer.clock()
            candidate = reviewer._candidate
            if (result.binding is not self._binding or reviewer.registry is not self._registry
                    or reviewer._closed or candidate is None or candidate['version'] != result.version
                    or not self._is_current_locked(result, now)):
                return
            self._current = result
            if not self._history or self._history[-1] != result.advice:
                self._history.append(result.advice)

    def _is_current_locked(self, result, now):
        registry = self._registry
        reviewer = result.reviewer
        candidate = reviewer._candidate
        if (reviewer._closed or candidate is None or candidate['version'] != result.version
                or registry._source_ended or registry._error is not None or registry._published is None):
            return False
        generation, state, captured, expires = registry._published
        return (
            generation == registry._generation
            and captured <= now < expires
            and fresh(state, now)
            and 0 <= now - result.source_time <= result.review_window
            and (generation, registry._epoch, state['run_id'], state['state_id'], captured,
                 expires, registry._strategy_context_version) == result.token
        )

    def read(self):
        """Read at UI dispatch time, including expiry without a new frame."""
        registry = self._registry
        if registry is None:
            return SidebarSnapshot(None, ())
        with registry._condition:
            result = self._current
            if result is not None and not self._is_current_locked(result, registry.clock()):
                self._current = result = None
            return SidebarSnapshot(result.advice if result else None, tuple(self._history))
