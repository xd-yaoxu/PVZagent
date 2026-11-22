"""Goal-specific, attention-only review. This module never authorizes actions."""
import copy
import math

from .strategy_review import StrategicReviewer,fresh


def validate_policy(policy):
    if type(policy) is not dict or set(policy)!= {'left_warning_x','other_lane_margin_px'}:
        raise ValueError('Explicit review policy fields required')
    if any(type(value) is not int for value in policy.values()):
        raise ValueError('Integer review geometry required')
    if not 40<=policy['left_warning_x']<=240 or not 40<=policy['other_lane_margin_px']<=160:
        raise ValueError('Bounded conservative review geometry required')
    return dict(policy)


class ReviewBlocked(ValueError):
    def __init__(self,code):self.code=code;super().__init__(code)


def lane_observations(state):
    if state.get('scene',{}).get('value')!='playing':raise ReviewBlocked('nonplaying_scene')
    # An unsupported channel can alter priorities even if its precise row is unknown.
    vehicle=state.get('vehicles')
    if type(vehicle) is not dict or any(key not in vehicle for key in ['candidates','tracks','uncertain_candidates']):
        raise ReviewBlocked('vehicle_context_unknown')
    if any(vehicle[key] for key in ['candidates','tracks','uncertain_candidates']):
        raise ReviewBlocked('vehicle_context_unsupported')
    zombies=state.get('zombies')
    if type(zombies) is not dict or any(key not in zombies for key in ['candidates','uncertain_edge_presence','uncertain_water_candidates']):
        raise ReviewBlocked('head_context_unknown')
    if zombies['uncertain_edge_presence'] or zombies['uncertain_water_candidates']:
        raise ReviewBlocked('unlocalized_threat_context')
    if type(zombies['candidates']) is not list:raise ReviewBlocked('head_context_unknown')
    rows={row:[] for row in range(1,7)}
    for item in zombies['candidates']:
        try:row=item['row'];x=item['head_xy'][0]
        except (KeyError,TypeError,IndexError):raise ReviewBlocked('head_position_unknown') from None
        if type(row) is not int or not 1<=row<=6 or type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<800:
            raise ReviewBlocked('head_position_unknown')
        rows[row].append(float(x))
    return {row:sorted(xs) for row,xs in rows.items()}


def defense_evidence(source,current,focus_row,policy):
    """Validate relevance of attention to a lane, not a particular zombie/action."""
    policy=validate_policy(policy)
    if type(focus_row) is not int or not 1<=focus_row<=6:raise ReviewBlocked('focus_row_unknown')
    before=lane_observations(source);after=lane_observations(current)
    initial=before[focus_row];latest=after[focus_row]
    if not initial or not latest:raise ReviewBlocked('focus_lane_unconfirmed')
    if len(initial)!=len(latest):raise ReviewBlocked('focus_count_changed')
    threshold=policy['left_warning_x'];margin=policy['other_lane_margin_px']
    if (min(initial)<=threshold)!=(min(latest)<=threshold):
        raise ReviewBlocked('focus_warning_boundary_crossed')
    # A new, nearer-to-left observation elsewhere invalidates a single-lane priority.
    # Pixel position is a warning heuristic, not a calibrated danger/health score.
    def warnings(rows):
        focus=min(rows[focus_row])
        return {row for row,xs in rows.items() if row!=focus_row and xs
                and (min(xs)<=threshold or min(xs)+margin<focus)}
    new_warnings=warnings(after)-warnings(before)
    if new_warnings:raise ReviewBlocked('new_other_lane_warning')
    return dict(focus_row=focus_row,source_visible_count=len(initial),current_visible_count=len(latest),
        source_leftmost_reference_x=min(initial),current_leftmost_reference_x=min(latest),
        current_left_warning=min(latest)<=threshold,new_other_lane_warnings=sorted(new_warnings),
        identity_verified=False,death_verified=False,full_board_verified=False,
        irrelevant_to_attention_only=['sun','card_appearance','ordinary_head_motion_within_warning_boundary'])


class GoalStrategicReviewer(StrategicReviewer):
    def __init__(self,*args,policy,**kwargs):
        self.policy=validate_policy(policy);super().__init__(*args,**kwargs)

    def review(self):
        with self._lock:
            if self._closed or self._candidate is None:return None
            if self._candidate['strategy']['goal']!='defend_lane':
                return super().review()
            candidate=self._candidate;source=candidate['source'];now=self.clock();current=self._snapshot()
            def reject(reason):
                self._candidate=None
                return dict(disposition=reason,source_state_id=source['state_id'],
                    reviewed_state_id=current.get('state_id') if isinstance(current,dict) else None,
                    review_policy='defense_attention/v1',input_sent=False,action_eligible=False)
            if candidate['epoch']!=self._epoch or self._candidate is not candidate:
                return reject('continuity_epoch_changed')
            if not 0<=now-source['frame_time_seconds']<=self.review_window:
                return reject('historical_window_expired')
            if current is None or not fresh(current,now) or current['run_id']!=source['run_id']:
                self.invalidate('fresh_continuous_state_unavailable')
                return reject('fresh_continuous_state_unavailable')
            if current['frame_time_seconds']<source['frame_time_seconds']:return reject('nonmonotonic_observation')
            try:evidence=defense_evidence(source,current,candidate['strategy']['focus_row'],self.policy)
            except ReviewBlocked as exc:return reject(exc.code)
            return dict(disposition='selected_observations_match',review_policy='defense_attention/v1',
                source_state_id=source['state_id'],reviewed_state_id=current['state_id'],
                expires_monotonic=current['readonly_runtime']['expires_monotonic'],
                source_age_seconds=now-source['frame_time_seconds'],strategy=copy.deepcopy(candidate['strategy']),
                evidence=evidence,review_scope='Current visible-headed lane attention only. Not an exclusive optimal priority, planting/shovel proposal, exact target, resources, cooldown, health or safety claim.',
                requires_new_action_proposal=True,requires_current_target_and_legality_check=True,
                input_sent=False,action_eligible=False)
