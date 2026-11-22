"""Background strategic review attached to the frozen read-only live pipeline."""
import copy
import threading
import time

from .continuity_frames import ContinuityFrames
from .readonly_longrun import run_long_readonly
from .strategy_review import StrategicReviewer, observed_basis


class StrategySupervisor:
    def __init__(self,registry,backend,*,record_sink,clock=time.monotonic,max_requests=2,min_request_interval=15.):
        if type(max_requests) is not int or not 0<=max_requests<=3:
            raise ValueError('Bounded0..3 requests required')
        if type(min_request_interval) not in (int,float) or not 5<=min_request_interval<=60:
            raise ValueError('Request spacing5..60s required')
        self.registry=registry;self.clock=clock;self.sink=record_sink
        self.reviewer=StrategicReviewer(backend,current_state=lambda:registry.current(clock()),clock=clock)
        registry.bind_invalidator(self.reviewer.invalidate)
        self.max_requests=max_requests;self.min_request_interval=min_request_interval
        self.requests=0;self.next_request=0;self.records=[];self.errors=[];self._last_key=None;self._unknown_sid=None
        self._stop=threading.Event();self._thread=None

    def _emit(self,**record):
        record.update(monotonic=self.clock(),input_sent=False,action_eligible=False)
        self.sink(record);self.records.append(copy.deepcopy(record))

    def poll(self):
        if self.registry.source_ended:return
        current=self.registry.current(self.clock())
        # Current is routinely absent between capture and perception. Registry
        # callbacks, not this transient absence, revoke continuity epochs.
        if current is None:return
        review=self.reviewer.review()
        if review is not None:
            key=(current['state_id'],review['source_state_id'],review['disposition'])
            if key!=self._last_key:
                self._emit(kind='strategy_review',reviewed_state_id=current['state_id'],review=review)
                self._last_key=key
        if self.requests>=self.max_requests or self.clock()<self.next_request:return
        if review is not None and review['disposition']=='selected_observations_match':return
        try:observed_basis(current)
        except (KeyError,ValueError,TypeError,IndexError):
            if current['state_id']!=self._unknown_sid:
                self._emit(kind='strategy_source_unknown',state_id=current['state_id'])
                self._unknown_sid=current['state_id']
            return
        if self.reviewer.submit():
            self.requests+=1;self.next_request=self.clock()+self.min_request_interval
            self._emit(kind='strategy_requested',source_state_id=current['state_id'],request_number=self.requests)

    def start(self):
        if self._thread is not None:raise RuntimeError('Supervisor already started')
        def run():
            try:
                while not self._stop.wait(.05):self.poll()
            except BaseException as exc:
                self.errors.append(dict(stage='strategy_supervisor',error=type(exc).__name__))
                self.registry.fail(RuntimeError('strategy_supervisor_failed'))
        self._thread=threading.Thread(target=run,name='pvz-strategy-supervisor',daemon=True);self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:self._thread.join(timeout=2)
        self.reviewer.close()
        # The base reviewer cannot cancel already transmitted HTTP requests.
        future=self.reviewer._future
        joined=True
        if future is not None:
            try:future.result(timeout=46)
            except TimeoutError:joined=False
            except BaseException as exc:self.errors.append(dict(stage='strategy_worker_shutdown',error=type(exc).__name__))
        if self._thread is not None and self._thread.is_alive():joined=False
        return dict(requests=self.requests,records=self.records,errors=self.errors,
            adviser_events=self.reviewer.events(),workers_finished=joined,
            current_review=None,input_sent=False,action_eligible=False)


def run_strategy_readonly(*,backend,strategy_record_sink,max_requests=2,min_request_interval=15.,**runtime_args):
    """Real capture hooks and independent model supervisor; no activation/input."""
    registry=ContinuityFrames()
    supervisor=StrategySupervisor(registry,backend,record_sink=strategy_record_sink,
        max_requests=max_requests,min_request_interval=min_request_interval)
    supervisor.start()
    try:result=run_long_readonly(frame_registry=registry,**runtime_args)
    finally:
        registry.close();strategy=supervisor.stop()
    result.update(strategy=strategy,continuity_events=registry.continuity_events(),
        limits=result['limits']+[
            'The base runner has no LLM; this wrapper adds a background strategic-only model request.',
            'Continuity is screenshot/metadata observable only; unseen short pauses/restarts cannot be inferred.',
            'No strategy review authorizes actions; no sidebar focus behavior or autonomous wins verified.'])
    return result
