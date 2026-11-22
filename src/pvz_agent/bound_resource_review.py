"""Product-only, registry-atomic strategic commitments. No input authority.

The registry condition owns all reviewer state. Evidence, backend and sinks run
outside it. Tickets certify an instant, not future validity or remote cancellation.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import time

from .resource_threat_v2 import ThreatCheckedContinuityFrames, validate_threat_context
from .strategy_review import fresh, observed_basis, validate_strategy
from .goal_review import ReviewBlocked, defense_evidence, validate_policy
from . import resource_review


class BoundResourceReviewer:
    def __init__(self, registry, backend, *, policy, clock=time.monotonic,max_requests=2,min_request_interval=15.):
        if not isinstance(registry, ThreatCheckedContinuityFrames):
            raise TypeError('Product review requires threat-checked continuity registry')
        self.registry=registry; self.backend=backend; self.clock=clock; self.policy=validate_policy(policy)
        self.review_window=30.; self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pvz-bound-strategy')
        self._future=None; self._active=False; self._launch_pending=False; self._closed=False; self._candidate=None; self._version=0; self._events=[];self._commit=None
        self.requests=0;self.max_requests=max_requests;self.min_request_interval=min_request_interval;self.next_request=0

    def _snapshot_locked(self, now):
        r=self.registry
        if r._source_ended or r._error is not None or r._published is None:return None,None
        generation,state,captured,expires=r._published
        if generation!=r._generation or not captured<=now<expires or not fresh(state,now):return None,None
        token=(generation,r._epoch,state['run_id'],state['state_id'],captured,expires,r._strategy_context_version)
        return copy.deepcopy(state),token

    def snapshot(self):
        now=self.clock()
        # Detect capture stalls using the existing observable-continuity mechanism.
        self.registry.current(now)
        with self.registry._condition:return self._snapshot_locked(self.clock())

    def _valid_locked(self, token, now):
        _,actual=self._snapshot_locked(now)
        return token is not None and actual==token

    def submit(self, state=None, token=None):
        if state is None:state,token=self.snapshot()
        try:
            validate_threat_context(state); observed_basis(state)
        except (ValueError,KeyError,TypeError,IndexError):return False
        with self.registry._condition:
            now=self.clock()
            actual,actual_token=self._snapshot_locked(now)
            if (self._closed or self._active or self._launch_pending or self.requests>=self.max_requests or now<self.next_request
                    or token!=actual_token or state!=actual):return False
            self._active=True;self._launch_pending=True; self._version+=1; version=self._version; self._candidate=None;self._commit=None
            self.requests+=1;self.next_request=now+self.min_request_interval
        try:future=self._executor.submit(self._request,actual,token,version)
        except Exception as exc:
            future=None
            with self.registry._condition:self._events.append(dict(kind='launch_failed',error=type(exc).__name__))
        with self.registry._condition:
            self._launch_pending=False
            if future is None:self._active=False
            else:self._future=future
            closed=self._closed
        if closed:self._executor.shutdown(wait=False,cancel_futures=True)
        if future is None:return False
        return True

    def _request(self, source, token, version):
        try:
            with self.registry._condition:
                if self._closed or version!=self._version or not self._valid_locked(token,self.clock()):
                    self._events.append(dict(kind='stale_before_request',source_state_id=source['state_id']));return
                self._events.append(dict(kind='local_dispatch_committed',source_state_id=source['state_id']))
            try:
                result=self.backend(copy.deepcopy(source));strategy=validate_strategy(result['strategy'],source['state_id']);error=None
            except Exception as exc:error=getattr(exc,'code',type(exc).__name__)
            with self.registry._condition:
                if (self._closed or version!=self._version or token[1]!=self.registry._epoch
                        or token[6]!=self.registry._strategy_context_version or self.registry._source_ended):
                    self._events.append(dict(kind='invalidated_response',source_state_id=source['state_id']))
                elif error is not None:self._events.append(dict(kind='request_failed',error=error))
                elif not 0<=self.clock()-source['frame_time_seconds']<=self.review_window:
                    self._events.append(dict(kind='historical_window_expired',source_state_id=source['state_id']))
                else:
                    self._candidate=dict(source=source,strategy=strategy,version=version,epoch=token[1],strategy_version=token[6])
                    self._events.append(dict(kind='hypothesis_retained',source_state_id=source['state_id']))
        finally:
            with self.registry._condition:self._active=False

    def invalidate(self,reason='continuity_lost'):
        with self.registry._condition:
            self._version+=1;self._candidate=None;self._commit=None;self._events.append(dict(kind='invalidated',reason=reason))

    def review(self):
        state,token=self.snapshot()
        with self.registry._condition:
            if self._closed or self._candidate is None:return None
            candidate=copy.deepcopy(self._candidate);version=candidate['version']
        source=candidate['source'];now=self.clock();reason=None;evidence=None;goal=candidate['strategy']
        if token is None:reason='fresh_continuous_state_unavailable'
        elif (candidate['epoch']!=token[1] or candidate['strategy_version']!=token[6]
                or state['run_id']!=source['run_id']):reason='continuity_epoch_changed'
        elif not 0<=now-source['frame_time_seconds']<=self.review_window:reason='historical_window_expired'
        elif state['frame_time_seconds']<source['frame_time_seconds']:reason='nonmonotonic_observation'
        else:
            try:
                validate_threat_context(source);validate_threat_context(state)
                if goal['goal']=='collect_resources':evidence=resource_review.resource_evidence(source,state,self.policy)
                elif goal['goal']=='defend_lane':evidence=defense_evidence(source,state,goal['focus_row'],self.policy)
                elif observed_basis(source)!=observed_basis(state):reason='observed_context_changed'
            except ReviewBlocked as exc:reason=exc.code
            except (ValueError,KeyError,TypeError,IndexError):reason='observed_context_unknown'
        with self.registry._condition:
            now=self.clock()
            if (self._closed or self._candidate is None or self._candidate['version']!=version
                or candidate['epoch']!=self.registry._epoch or candidate['strategy_version']!=self.registry._strategy_context_version
                or not self._valid_locked(token,now)
                or not 0<=now-source['frame_time_seconds']<=self.review_window):return None
            if reason:
                self._candidate=None
                return dict(disposition=reason,source_state_id=source['state_id'],reviewed_state_id=state['state_id'],input_sent=False,action_eligible=False)
            policy={'collect_resources':'resource_attention/v1','defend_lane':'defense_attention/v1'}.get(goal['goal'],'selected_observations/v1')
            scopes={'collect_resources':'Nonexclusive resource observation intent, supported by current sun appearance cues. Not original-sun identity, clickability, value, optimal priority or legal action.',
                'defend_lane':'Current visible-headed lane attention only. Not an exclusive optimal priority, planting/shovel proposal, exact target, resources, cooldown, health or safety claim.',
                'hold':'Scene, known sun, ten card appearances and headed row/x zones only; not full board, plant legality, health or target verification.',
                'develop_economy':'Scene, known sun, ten card appearances and headed row/x zones only; not full board, plant legality, health or target verification.'}
            review=dict(disposition='selected_observations_match',review_policy=policy,source_state_id=source['state_id'],
                reviewed_state_id=state['state_id'],expires_monotonic=token[5],source_age_seconds=now-source['frame_time_seconds'],
                strategy=copy.deepcopy(goal),evidence=evidence,review_scope=scopes[goal['goal']],
                requires_new_action_proposal=True,requires_current_target_and_legality_check=True,input_sent=False,action_eligible=False,
                validity='historical_at_commit')
            if goal['goal']=='collect_resources':review['requires_fresh_pickup_proposal']=True
            opaque=object();self._commit=(opaque,token,version,source['frame_time_seconds'])
            return review,opaque

    def consume(self, ticket):
        with self.registry._condition:
            commit=self._commit
            if commit is None or ticket is not commit[0]:return False
            self._commit=None
            _,token,version,source_time=commit
            now=self.clock()
            return (not self._closed and self._candidate is not None and self._candidate['version']==version
                and self._candidate['epoch']==self.registry._epoch
                and self._candidate['strategy_version']==self.registry._strategy_context_version and self._valid_locked(token,now)
                and 0<=now-source_time<=self.review_window)

    def events(self):
        with self.registry._condition:return copy.deepcopy(self._events)

    def launch_pending(self):
        with self.registry._condition:return self._launch_pending

    def close(self):
        with self.registry._condition:
            self._closed=True;self._candidate=None;self._commit=None;self._version+=1
            pending=self._launch_pending
        if not pending:self._executor.shutdown(wait=False,cancel_futures=True)
