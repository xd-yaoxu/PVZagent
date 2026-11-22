"""Explicit opt-in supervisor for scoped attention review; base runtime retained."""
import threading
import time

from .continuity_frames import ContinuityFrames
from .goal_review import GoalStrategicReviewer
from .readonly_longrun import run_long_readonly
from .strategy_runtime import StrategySupervisor


class GoalStrategySupervisor(StrategySupervisor):
    def __init__(self,registry,backend,*,policy,record_sink,clock=time.monotonic,max_requests=2,min_request_interval=15.):
        if type(max_requests) is not int or not 0<=max_requests<=3:raise ValueError('Bounded0..3 requests required')
        if type(min_request_interval) not in (int,float) or not 5<=min_request_interval<=60:
            raise ValueError('Request spacing5..60s required')
        self.registry=registry;self.clock=clock;self.sink=record_sink
        self.reviewer=GoalStrategicReviewer(backend,current_state=lambda:registry.current(clock()),clock=clock,policy=policy)
        registry.bind_invalidator(self.reviewer.invalidate)
        self.max_requests=max_requests;self.min_request_interval=min_request_interval
        self.requests=0;self.next_request=0;self.records=[];self.errors=[];self._last_key=None;self._unknown_sid=None
        self._stop=threading.Event();self._thread=None
        # Inherited dispatch still requires the old full known observed_basis.
        # Only returned defend_lane review is scoped; other goals keep old rules.


def run_goal_readonly(*,backend,policy,strategy_record_sink,max_requests=2,min_request_interval=15.,**runtime_args):
    registry=ContinuityFrames()
    supervisor=GoalStrategySupervisor(registry,backend,policy=policy,record_sink=strategy_record_sink,
        max_requests=max_requests,min_request_interval=min_request_interval)
    supervisor.start()
    try:result=run_long_readonly(frame_registry=registry,**runtime_args)
    finally:
        registry.close();strategy=supervisor.stop()
    result.update(strategy=strategy,continuity_events=registry.continuity_events(),review_policy='defense_attention/v1',
        limits=result['limits']+['Base capture/perception has no LLM; wrapper supports a strategic background model.',
            'Defense matches support lane attention only and never authorize an action.',
            'Current snapshots remain2s; historical strategy window remains30s; dispatch gate unchanged.'])
    return result
