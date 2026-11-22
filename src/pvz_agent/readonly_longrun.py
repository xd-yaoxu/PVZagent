"""Finite long read-only runner, bounded snapshots and incremental fault journal."""
from datetime import datetime,timezone
from pathlib import Path
import threading
import time
from PIL import Image
from .latest_frames import SkippingSchedule,finite_nonnegative
from .snapshot_frames import SnapshotFrames
from .readonly_pipeline import capture_valid


def run_long_readonly(*,capture_factory,observe,count,interval,max_input_age,max_result_age,
                      max_duration,evidence_dir,archive_state,record_sink,frame_registry=None,
                      bind_tracking_continuity=False,on_failure_stage=None):
    if type(bind_tracking_continuity) is not bool:raise ValueError('Explicit boolean tracking continuity choice required')
    if on_failure_stage is not None and not callable(on_failure_stage):raise ValueError('Failure stage observer must be callable')
    if type(count) is not int or not 1<=count<=600:raise ValueError('Finite1..600 capture count required')
    if not all(finite_nonnegative(v) and v>0 for v in [interval,max_input_age,max_result_age,max_duration]):
        raise ValueError('Finite positive timing settings required')
    if max_result_age<max_input_age or max_duration>600:raise ValueError('Ordered age budgets and duration<=600s required')
    registry=frame_registry if frame_registry is not None else SnapshotFrames()
    stop=threading.Event();records=[];errors=[];lock=threading.Lock();started=time.monotonic();deadline=started+max_duration
    def emit(**value):
        with lock:
            value.update(sequence=len(records),monotonic_ns=time.monotonic_ns(),recorded_at=datetime.now(timezone.utc).isoformat())
            record_sink(value);records.append(value)
    def fail(stage,exc):
        try:
            detail=repr(exc)
        except BaseException:
            detail='Exception representation unavailable'
        failure=dict(stage=stage,error=detail)
        with lock:
            errors.append(failure)

        secondary_errors=[]
        try:
            registry.fail(exc)
        except BaseException:
            secondary_errors.append({'source':'registry.fail'})
        finally:
            stop.set()

        if on_failure_stage is not None:
            try:
                on_failure_stage(stage)
            except BaseException:
                secondary_errors.append({'source':'on_failure_stage'})

        if secondary_errors:
            with lock:
                failure['secondary_errors']=secondary_errors
    def producer():
        try:
            capture=capture_factory();schedule=SkippingSchedule(time.monotonic(),interval)
            for index in range(count):
                if schedule.next_due>=deadline:
                    emit(kind='duration_limit',next_capture_index=index);break
                while True:
                    if stop.wait(max(0,schedule.next_due-time.monotonic())):break
                    if time.monotonic()>=schedule.next_due:break
                if stop.is_set():break
                if time.monotonic()>=deadline:
                    emit(kind='duration_limit',next_capture_index=index);break
                due=schedule.next_due;tick=time.monotonic()
                image,path,meta=capture.capture(evidence_dir,label='longrun')
                done=time.monotonic();skipped=schedule.completed(done)
                item=dict(index=index,image=Path(image),metadata_file=Path(path),metadata=meta)
                replaced=registry.put(item)
                emit(kind='captured',index=index,state_id=meta['state_id'],valid=capture_valid(meta),
                    capture_and_persist_ms=(done-tick)*1000,schedule_lateness_ms=max(0,tick-due)*1000,
                    skipped_capture_slots=skipped,profile=meta.get('profile',{}))
                if replaced is not None:emit(kind='replaced_pending',state_id=replaced.item['metadata']['state_id'])
            registry.close()
        except BaseException as exc:fail('capture_or_journal',exc)
    worker=threading.Thread(target=producer,name='pvz-long-readonly-capture',daemon=True);worker.start()
    try:
        while True:
            ticket=registry.take()
            if ticket is None:break
            item=ticket.item;meta=item['metadata'];captured_at=meta['monotonic_ns']/1e9;age=time.monotonic()-captured_at
            if not capture_valid(meta):
                emit(kind='rejected_capture',state_id=meta['state_id'],input_age_ms=age*1000);continue
            if not 0<=age<=max_input_age:
                emit(kind='expired_before_inference',state_id=meta['state_id'],input_age_ms=age*1000);continue
            tick=time.monotonic()
            if bind_tracking_continuity:
                epoch=item.get('_continuity_epoch')
                if type(epoch) is not int or epoch<0:raise ValueError('Valid ticket continuity epoch required')
            with Image.open(item['image']) as image:
                if bind_tracking_continuity:
                    state=observe(image,meta,item['metadata_file'],item['image'],tracking_epoch=epoch)
                else:
                    state=observe(image,meta,item['metadata_file'],item['image'])
            if state['input_sent'] is not False or state['action_eligible'] is not False:
                raise ValueError('Read-only observation required')
            now=time.monotonic()
            state=registry.publish_observation(ticket,state,captured_at=captured_at,now=now,max_age=max_result_age,input_age_ms=age*1000)
            archive_state(state)
            emit(kind='inferred',state_id=meta['state_id'],processing_and_load_ms=(now-tick)*1000,
                input_age_ms=age*1000,result_age_ms=(now-captured_at)*1000,
                publication=state['readonly_runtime']['publication_at_completion'],scene=state['scene']['value'])
    except BaseException as exc:fail('inference_archive_or_journal',exc)
    finally:
        stop.set();worker.join(timeout=10)
        if worker.is_alive():fail('capture_shutdown',RuntimeError('Capture worker did not finish within10s'))
    return dict(records=records,errors=errors,duration_seconds=time.monotonic()-started,max_duration_seconds=max_duration,
        pending_capacity=1,capture_threads=1,consumer_threads=1,input_sent=False,action_eligible=False,
        agent_manipulated_focus=False,
        limits=['Read-only runtime, no LLM, sidebar or game inputs.',
            'Capture already in progress and final inference may finish after duration limit; no new slot starts after deadline.',
            'Original evidence and suppressed results retained; dropped frames may change association.',
            'Journal failures stop publication; final summary may be absent if the process is forcibly terminated.'])
