"""Nonexclusive resource attention needs a current sun cue, never just a counter."""
import copy
import math
from .goal_review import GoalStrategicReviewer,ReviewBlocked,lane_observations,validate_policy
from .strategy_review import fresh


def sun_cue_ids(state):
    try:
        cue=state['resources']['sun_appearance'];sid=state['state_id'];capture=state['capture_time'];image=state['evidence']
        if cue['schema']!='pvz-sun-appearance/1' or cue['status']!='observed' or cue['state_id']!=sid or cue['capture_time']!=capture:
            raise ValueError('binding')
        if any(cue['evidence'][k]!=image[k] for k in ['image','image_sha256']):raise ValueError('evidence')
        candidates=cue['candidates']
        if type(candidates) is not list:raise ValueError('candidates')
        ids=[]
        for item in candidates:
            if item['kind']!='visible_sun_appearance_candidate' or item['state_id']!=sid or item['capture_time']!=capture or item['action_eligible'] is not False:
                raise ValueError('candidate_binding')
            if any(item['evidence'][k]!=image[k] for k in ['image','image_sha256']):raise ValueError('candidate_evidence')
            point=item['visible_reference_xy'];box=item['box_xyxy']
            if len(point)!=2 or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in point+box):raise ValueError('geometry')
            if not (0<=box[0]<=point[0]<box[2]<=800 and 0<=box[1]<=point[1]<box[3]<=600):raise ValueError('bounds')
            cid=item['cue_id']
            if type(cid) is not str or not cid or cid in ids:raise ValueError('cue_id')
            ids.append(cid)
        return ids
    except (KeyError,ValueError,TypeError,IndexError):raise ReviewBlocked('sun_cue_context_unknown') from None


def resource_evidence(source,current,policy):
    policy=validate_policy(policy);before=lane_observations(source);after=lane_observations(current)
    source_ids=sun_cue_ids(source);current_ids=sun_cue_ids(current)
    if not source_ids:raise ReviewBlocked('source_sun_cue_unconfirmed')
    if not current_ids:raise ReviewBlocked('current_sun_cue_unconfirmed')
    warning=policy['left_warning_x'];margin=policy['other_lane_margin_px']
    if any(xs and min(xs)<=warning for rows in [before,after] for xs in rows.values()):
        raise ReviewBlocked('resource_left_warning')
    source_left=min((min(xs) for xs in before.values() if xs),default=800.)
    for row,xs in after.items():
        if not xs:continue
        reference=min(before[row]) if before[row] else source_left
        if min(xs)+margin<reference:raise ReviewBlocked('resource_new_nearer_warning')
    return dict(source_cue_ids=source_ids,current_cue_ids=current_ids,current_cue_count=len(current_ids),
        original_sun_identity_verified=False,interactable_verified=False,full_board_verified=False,
        irrelevant_to_attention_only=['stored_sun_counter','card_appearance','ordinary_head_motion_without_new_warning'])


class ResourceStrategicReviewer(GoalStrategicReviewer):
    def review(self):
        with self._lock:
            if self._closed or self._candidate is None:return None
            if self._candidate['strategy']['goal']!='collect_resources':return super().review()
            candidate=self._candidate;source=candidate['source'];now=self.clock();current=self._snapshot()
            def reject(reason):
                self._candidate=None
                return dict(disposition=reason,source_state_id=source['state_id'],
                    reviewed_state_id=current.get('state_id') if type(current) is dict else None,
                    review_policy='resource_attention/v1',input_sent=False,action_eligible=False)
            if candidate['epoch']!=self._epoch:return reject('continuity_epoch_changed')
            if not 0<=now-source['frame_time_seconds']<=self.review_window:return reject('historical_window_expired')
            if current is None or not fresh(current,now) or current['run_id']!=source['run_id']:
                self.invalidate('fresh_continuous_state_unavailable');return reject('fresh_continuous_state_unavailable')
            if current['frame_time_seconds']<source['frame_time_seconds']:return reject('nonmonotonic_observation')
            try:evidence=resource_evidence(source,current,self.policy)
            except ReviewBlocked as exc:return reject(exc.code)
            return dict(disposition='selected_observations_match',review_policy='resource_attention/v1',
                source_state_id=source['state_id'],reviewed_state_id=current['state_id'],
                expires_monotonic=current['readonly_runtime']['expires_monotonic'],source_age_seconds=now-source['frame_time_seconds'],
                strategy=copy.deepcopy(candidate['strategy']),evidence=evidence,
                review_scope='Nonexclusive resource observation intent, supported by current sun appearance cues. Not original-sun identity, clickability, value, optimal priority or legal action.',
                requires_new_action_proposal=True,requires_current_target_and_legality_check=True,
                requires_fresh_pickup_proposal=True,input_sent=False,action_eligible=False)
