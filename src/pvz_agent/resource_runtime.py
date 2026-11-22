"""Explicit resource-attention supervisor, preserving old source dispatch gates."""
import threading
import time
from .goal_runtime import GoalStrategySupervisor
from .resource_review import ResourceStrategicReviewer
from .continuity_frames import ContinuityFrames
from .readonly_longrun import run_long_readonly
from .resource_threat_v2 import ThreatCheckedContinuityFrames, validate_threat_context
from .bound_resource_review import BoundResourceReviewer
from .strategy_review import observed_basis


class ResourceStrategySupervisor(GoalStrategySupervisor):
    def __init__(self,registry,backend,*,policy,record_sink,clock=time.monotonic,max_requests=2,min_request_interval=15.,sidebar=None):
        if type(max_requests) is not int or not 0<=max_requests<=3:raise ValueError('Bounded0..3 requests required')
        if type(min_request_interval) not in (int,float) or not 5<=min_request_interval<=60:raise ValueError('Request spacing5..60s required')
        self.registry=registry;self.clock=clock;self.sink=record_sink
        self.sidebar=sidebar
        self.reviewer=BoundResourceReviewer(registry,backend,clock=clock,policy=policy,
            max_requests=max_requests,min_request_interval=min_request_interval)
        self.max_requests=max_requests;self.min_request_interval=min_request_interval;self.requests=0;self.next_request=0
        self.records=[];self.errors=[];self._last_key=None;self._unknown_sid=None;self._stop=threading.Event();self._thread=None;self._poll_busy=False

    def poll(self):
        # Ownership is a short registry transaction; the body and all callbacks
        # run after releasing the registry lock, including reentrant sinks.
        with self.registry._condition:
            if self._poll_busy or self._stop.is_set():return
            self._poll_busy=True
        try:self._poll_once()
        finally:
            with self.registry._condition:self._poll_busy=False

    def _poll_once(self):
        if self.registry.source_ended:return
        result=self.reviewer.review()
        ticket=None;review=None
        if isinstance(result,tuple):review,ticket=result
        elif result is not None:review=result
        presentation = None
        if self.sidebar is not None and ticket is not None:
            presentation = self.sidebar._prepare(self.reviewer, review, ticket)
        if review is not None:
            key=(review.get('reviewed_state_id'),review['source_state_id'],review['disposition'])
            if key!=self._last_key:
                self._emit(kind='strategy_review',reviewed_state_id=review.get('reviewed_state_id'),review=review)
                self._last_key=key
        # The journal is historical. Only a fresh consumption can suppress a request.
        if ticket is not None and self.reviewer.consume(ticket):
            if presentation is not None:
                self.sidebar._present(self.reviewer, presentation)
            return
        if self.requests>=self.max_requests or self.clock()<self.next_request:return
        current,source_ticket=self.reviewer.snapshot()
        if current is None:return
        try:
            validate_threat_context(current);observed_basis(current)
        except (KeyError,ValueError,TypeError,IndexError):
            if current['state_id']!=self._unknown_sid:
                self._emit(kind='strategy_source_unknown',state_id=current['state_id'])
                self._unknown_sid=current['state_id']
            return
        if self.reviewer.submit(current,source_ticket):
            with self.registry._condition:
                self.requests=self.reviewer.requests;self.next_request=self.reviewer.next_request
                number=self.requests
            self._emit(kind='strategy_requested',source_state_id=current['state_id'],request_number=number)

    def stop(self):
        self._stop.set()
        if self._thread is not None:self._thread.join(timeout=2)
        self.reviewer.close()
        with self.registry._condition:
            pending=self.reviewer._launch_pending;future=self.reviewer._future;requests=self.reviewer.requests;busy=self._poll_busy
        joined=not pending and not busy
        if future is not None:
            try:future.result(timeout=46)
            except TimeoutError:joined=False
            except BaseException as exc:self.errors.append(dict(stage='strategy_worker_shutdown',error=type(exc).__name__))
        if self._thread is not None and self._thread.is_alive():joined=False
        return dict(requests=requests,records=self.records,errors=self.errors,
            adviser_events=self.reviewer.events(),workers_finished=joined,
            current_review=None,input_sent=False,action_eligible=False)


def run_resource_readonly(*,backend,policy,strategy_record_sink,max_requests=2,min_request_interval=15.,sidebar=None,**runtime_args):
    if sidebar is None:
        registry=ThreatCheckedContinuityFrames()
    else:
        from .resource_sidebar import ReadonlyResourceSidebar
        if type(sidebar) is not ReadonlyResourceSidebar:
            raise TypeError('Read-only resource sidebar required')
        registry=sidebar._bind()
    supervisor=ResourceStrategySupervisor(registry,backend,policy=policy,
        record_sink=strategy_record_sink,max_requests=max_requests,min_request_interval=min_request_interval,sidebar=sidebar)
    supervisor.start()
    try:result=run_long_readonly(frame_registry=registry,**runtime_args)
    finally:registry.close();strategy=supervisor.stop()
    result.update(strategy=strategy,continuity_events=registry.continuity_events(),review_policy='goal_attention_with_resource/v1',
        limits=result['limits']+['Current resource appearance cues support only nonexclusive attention; no pickup or input is authorized.',
            'Current snapshots remain2s and historical strategies30s; dispatch retains full-known source requirements and adds strict threat channels and version binding.'])
    return result
