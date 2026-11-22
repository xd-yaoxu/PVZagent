"""Coordinate-free model strategies reviewed against fresh observed context.

The 30s review window is a budget for historical strategic hypotheses, never an
action TTL. Reviews never enter AsyncAdviser or an input executor.
"""
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time

from .llm_advice import strict_json, screenshot_request
from .llm_provider import CompatibleProvider, ProviderError, http_transport, load_local_api_key


STRATEGY_SCHEMA = dict(type='object', additionalProperties=False,
    required=['source_state_id', 'goal', 'focus_row', 'reason', 'uncertainties'],
    properties=dict(source_state_id=dict(type='string'),
        goal=dict(type='string', enum=['collect_resources', 'defend_lane', 'develop_economy', 'hold']),
        focus_row=dict(type=['integer', 'null']), reason=dict(type='string'),
        uncertainties=dict(type='array', items=dict(type='string'))))


class StrategyProvider(CompatibleProvider):
    """Opt-in Responses reasoning parameter; preserve the frozen base client."""
    def __init__(self, config, *, transport=http_transport, key_reader=load_local_api_key):
        effort = config.get('reasoning_effort')
        if effort not in (None, 'low', 'medium', 'high'):
            raise ValueError('Supported explicit reasoning effort required')
        if config['protocol'] != 'responses':
            raise ValueError('Strategic experiments use explicit Responses protocol')
        def configured_transport(url, body, key, timeout):
            body = copy.deepcopy(body)
            if effort is not None:
                body['reasoning'] = dict(effort=effort)
            return transport(url, body, key, timeout)
        super().__init__(config, transport=configured_transport, key_reader=key_reader)


def validate_strategy(value, source_id):
    if type(value) is not dict or set(value) != set(STRATEGY_SCHEMA['required']):
        raise ValueError('Strategy fields mismatch')
    if type(source_id) is not str or not source_id or value['source_state_id'] != source_id:
        raise ValueError('Strategy source mismatch')
    if value['goal'] not in ['collect_resources', 'defend_lane', 'develop_economy', 'hold']:
        raise ValueError('Unknown strategic goal')
    row = value['focus_row']
    if row is not None and (type(row) is not int or not 1 <= row <= 6):
        raise ValueError('Strategic row range')
    if (value['goal'] == 'defend_lane') != (row is not None):
        raise ValueError('Only lane defense requires a focus row')
    if type(value['reason']) is not str or not 1 <= len(value['reason']) <= 96:
        raise ValueError('Brief strategic reason required')
    uncertainty = value['uncertainties']
    if type(uncertainty) is not list or len(uncertainty) > 3 or any(type(s) is not str or not 1 <= len(s) <= 96 for s in uncertainty):
        raise ValueError('Bounded strategic uncertainty required')
    return copy.deepcopy(value)


def recommend_strategy(provider, root, state):
    # Reuse the exact existing path, PNG size and hash checks without shrinking images.
    _, image = screenshot_request(root, state)
    prompt = ('PVZ3-7历史截图，只给一条中文战略，非即时动作，不给像素坐标/种植位置。'
              'goal:collect_resources/defend_lane/develop_economy/hold；仅defend_lane填focus_row1..6，其余null。'
              'reason≤48字，uncertainties≤2条，每条≤48字。未知不猜，无检测不代表安全。'
              '建议须新图复核，未操作。原样回填source_state_id=' + state['state_id'])
    result = provider.complete(prompt, image_data_url=image, json_schema=STRATEGY_SCHEMA)
    try:
        strategy = validate_strategy(strict_json(result['text']), state['state_id'])
    except (ValueError, KeyError, TypeError):
        raise ProviderError('invalid_structured_strategy') from None
    return dict(strategy=strategy, protocol=result['protocol'], response_id=result['response_id'],
                model_returned=result['model_returned'], usage=result['usage'],
                input_sent=False, action_eligible=False, mode='historical_strategy_hypothesis')


def observed_basis(state):
    """Selected visual estimates only; equality never proves world-state equality."""
    if state['scene']['value'] != 'playing':
        raise ValueError('Playing scene required')
    sun = state['fields']['sun']['value']
    cards = state['fields']['cards']
    if type(sun) is not int or sun < 0 or len(cards) != 10:
        raise ValueError('Known resources and fixed ten-slot layout required')
    if {c['slot'] for c in cards if type(c['slot']) is int} != set(range(1, 11)):
        raise ValueError('Unique ten-slot layout required')
    card_values = tuple(sorted((c['slot'], c['visual_state']) for c in cards))
    if any(value not in ['bright', 'dim'] for _, value in card_values):
        raise ValueError('Occluded or unknown cards require reobservation')
    heads = Counter()
    for candidate in state['zombies']['candidates']:
        row = candidate['row']; x = candidate['head_xy'][0]
        if type(row) is not int or not 1 <= row <= 6 or type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 800:
            raise ValueError('Uncertain candidate position')
        heads[(row, int(x // 200))] += 1
    # Any vehicle or unlocalized edge/water context is outside the first review contract.
    if any(state.get('vehicles', {}).get(key) for key in ['candidates', 'tracks', 'uncertain_candidates']):
        raise ValueError('Vehicle context requires separate strategy review')
    zombies = state['zombies']
    if any(zombies.get(k) for k in ['uncertain_edge_presence', 'uncertain_water_candidates']):
        raise ValueError('Uncertain edge or water evidence')
    return dict(sun=sun, cards=card_values, headed_zones=tuple(sorted(heads.items())))


def fresh(state, now):
    try:
        captured = state['frame_time_seconds']; runtime = state['readonly_runtime']
        expires = runtime['expires_monotonic']
        return (type(now) in (int, float) and math.isfinite(now)
                and all(type(v) in (int, float) and math.isfinite(v) for v in [captured, expires])
                and 0 <= now - captured <= 2 and now < expires <= captured + 2
                and runtime['publication_at_completion'] == 'published'
                and type(state['run_id']) is str and bool(state['run_id'])
                and type(state['state_id']) is str and bool(state['state_id']))
    except (KeyError, TypeError):
        return False


class StrategicReviewer:
    """Single request, no pending queue; new live snapshots recheck past strategy.

    current_state must read the in-process fresh snapshot registry, never a JSON
    archive. Runtime must change epoch on restart, pause, focus/capture failure.
    """
    def __init__(self, backend, *, current_state, clock=time.monotonic, review_window=30):
        if type(review_window) not in (int, float) or not 0 < review_window <= 30:
            raise ValueError('Historical strategy window must be at most30 seconds')
        self.backend = backend; self.current_state = current_state; self.clock = clock
        self.review_window = review_window; self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pvz-strategy-review')
        self._future = None; self._candidate = None; self._closed = False; self._epoch = 0
        self._events = []

    def _snapshot(self):
        try:
            return copy.deepcopy(self.current_state())
        except Exception:
            return None

    def submit(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Reviewer closed')
            if self._future is not None and not self._future.done():
                return False
            state = self._snapshot(); now = self.clock()
            self._candidate = None
            if state is None or not fresh(state, now):
                self.invalidate('source_unavailable'); return False
            # Do not require complete context to ask the model, but it cannot pass
            # review until both source and current selected observations are known.
            epoch = self._epoch
            self._future = self._executor.submit(self._request, state, epoch)
            return True

    def _request(self, source, epoch):
        with self._lock:
            if self._closed or epoch != self._epoch or not fresh(source, self.clock()):
                self._events.append(dict(kind='stale_before_request', source_state_id=source['state_id']))
                return
        try:
            result = self.backend(copy.deepcopy(source))
            strategy = validate_strategy(result['strategy'], source['state_id'])
            error = None
        except Exception as exc:
            error = getattr(exc, 'code', type(exc).__name__)
        with self._lock:
            if self._closed or epoch != self._epoch:
                self._events.append(dict(kind='invalidated_response', source_state_id=source['state_id']))
            elif error is not None:
                self._candidate = None
                self._events.append(dict(kind='request_failed', error=error))
            elif not 0 <= self.clock() - source['frame_time_seconds'] <= self.review_window:
                self._events.append(dict(kind='historical_window_expired', source_state_id=source['state_id']))
            else:
                self._candidate = dict(source=source, strategy=strategy, epoch=epoch)
                self._events.append(dict(kind='hypothesis_retained', source_state_id=source['state_id']))

    def invalidate(self, reason='continuity_lost'):
        with self._lock:
            self._epoch += 1; self._candidate = None
            self._events.append(dict(kind='invalidated', reason=reason))

    def review(self):
        with self._lock:
            if self._closed or self._candidate is None:
                return None
            candidate = self._candidate; source = candidate['source']; now = self.clock()
            state = self._snapshot()
            def reject(reason):
                self._candidate = None
                return dict(disposition=reason, source_state_id=source['state_id'], input_sent=False, action_eligible=False)
            if not 0 <= now - source['frame_time_seconds'] <= self.review_window:
                return reject('historical_window_expired')
            if state is None or not fresh(state, now) or state['run_id'] != source['run_id']:
                self.invalidate('fresh_continuous_state_unavailable')
                return reject('fresh_continuous_state_unavailable')
            if state['frame_time_seconds'] < source['frame_time_seconds']:
                return reject('nonmonotonic_observation')
            try:
                basis = observed_basis(source)
                if basis != observed_basis(state):
                    return reject('observed_context_changed')
            except (KeyError, ValueError, TypeError, IndexError):
                return reject('observed_context_unknown')
            goal = candidate['strategy']
            if goal['goal'] == 'defend_lane' and not any(row == goal['focus_row'] for (row, _), count in basis['headed_zones'] if count):
                return reject('focus_lane_unconfirmed')
            return dict(disposition='selected_observations_match', source_state_id=source['state_id'],
                reviewed_state_id=state['state_id'], expires_monotonic=state['readonly_runtime']['expires_monotonic'],
                source_age_seconds=now-source['frame_time_seconds'], strategy=copy.deepcopy(goal),
                review_scope='Scene, known sun, ten card appearances and headed row/x zones only; not full board, plant legality, health or target verification.',
                requires_new_action_proposal=True, input_sent=False, action_eligible=False)

    def events(self):
        with self._lock:
            return copy.deepcopy(self._events)

    def close(self):
        with self._lock:
            self._closed = True; self._candidate = None; self._epoch += 1
        self._executor.shutdown(wait=False, cancel_futures=True)
