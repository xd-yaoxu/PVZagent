"""Two-thread read-only observation; fresh evidence before single-slot delivery."""
from pathlib import Path
import threading
import time
from PIL import Image
from .latest_frames import LatestFrames,SkippingSchedule,finite_nonnegative


def capture_valid(meta):
    return bool(meta['geometry_stable_during_capture'] and meta['display_mapping']['validated'] and
                meta['foreground_stable'] and meta['foreground_game_or_scaler'])


def run_readonly_pipeline(*,capture_factory,observe,count,interval,max_input_age,max_result_age,
                          evidence_dir,archive_state):
    if type(count) is not int or not 1<=count<=60:raise ValueError('Finite1..60 capture count required')
    if not all(finite_nonnegative(v) and v>0 for v in (interval,max_input_age,max_result_age)) or max_result_age<max_input_age:
        raise ValueError('Finite positive interval and ordered age budgets required')
    inbox=LatestFrames();stop=threading.Event();records=[];lock=threading.Lock();errors=[]
    def emit(**record):
        with lock:records.append(record)
    def producer():
        try:
            capture=capture_factory();schedule=SkippingSchedule(time.monotonic(),interval)
            for index in range(count):
                while True:
                    if stop.wait(max(0,schedule.next_due-time.monotonic())):break
                    if time.monotonic()>=schedule.next_due:break
                if stop.is_set():break
                due=schedule.next_due;tick=time.monotonic()
                image,path,meta=capture.capture(evidence_dir,label='pipeline')
                done=time.monotonic();skipped=schedule.completed(done)
                item=dict(index=index,image=Path(image),metadata_file=Path(path),metadata=meta)
                replaced=inbox.put(item)
                emit(kind='captured',index=index,state_id=meta['state_id'],valid=capture_valid(meta),
                    capture_and_persist_ms=(done-tick)*1000,schedule_lateness_ms=max(0,tick-due)*1000,
                    skipped_capture_slots=skipped,profile=meta.get('profile',{}))
                if replaced is not None:emit(kind='replaced_pending',state_id=replaced.item['metadata']['state_id'])
            inbox.close()
        except BaseException as exc:
            errors.append(repr(exc));inbox.fail(exc);stop.set()
    started=time.monotonic();worker=threading.Thread(target=producer,name='pvz-readonly-capture',daemon=True);worker.start()
    try:
        while True:
            ticket=inbox.take()
            if ticket is None:break
            item=ticket.item;meta=item['metadata'];captured_at=meta['monotonic_ns']/1e9
            age=time.monotonic()-captured_at
            if not capture_valid(meta):
                emit(kind='rejected_capture',state_id=meta['state_id'],input_age_ms=age*1000);continue
            if not 0<=age<=max_input_age:
                emit(kind='expired_before_inference',state_id=meta['state_id'],input_age_ms=age*1000);continue
            tick=time.monotonic()
            with Image.open(item['image']) as image:state=observe(image,meta,item['metadata_file'],item['image'])
            assert state['input_sent'] is False and state['action_eligible'] is False
            now=time.monotonic();result=inbox.publish(ticket,state,captured_at=captured_at,now=now,max_age=max_result_age)
            state['readonly_runtime']=dict(generation=ticket.generation,input_age_ms=age*1000,
                result_age_ms=(now-captured_at)*1000,publication_at_completion=result,
                currentness_limits='Archival snapshot; consumers must query generation/TTL, never execute this stored file.')
            archive_state(state)
            emit(kind='inferred',state_id=meta['state_id'],processing_and_load_ms=(now-tick)*1000,
                input_age_ms=age*1000,result_age_ms=(now-captured_at)*1000,publication=result,
                scene=state['scene']['value'])
    except BaseException as exc:
        errors.append(repr(exc));stop.set();inbox.fail(exc)
    finally:
        stop.set();worker.join(timeout=10)
        if worker.is_alive():errors.append('Capture worker did not finish within10s');inbox.fail(errors[-1])
    return dict(records=records,errors=errors,duration_seconds=time.monotonic()-started,
        pending_capacity=1,consumer_threads=1,capture_threads=1,
        input_sent=False,action_eligible=False,agent_manipulated_focus=False,
        limits=['Read-only experiment, no LLM, sidebar, input or autonomous control.',
            'Dropped frames keep original capture evidence. Superseded/expired states are archival only.',
            'Association consumes valid observed frames even if completed late; no publication authorizes action.'])
