"""Offline deterministic timing and cancellation tests for both capture runners."""
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
from PIL import Image
from pvz_agent import readonly_longrun, readonly_pipeline

class CaptureScheduleWaitTests(unittest.TestCase):
    def exercise(self,module,actions,*,count=2,max_duration=10,capture_delays=()):
        clock=[100.0]; waits=[]; captures=[]; instances=[]; delays=iter(capture_delays)
        class Event:
            def __init__(self): self.real=threading.Event(); instances.append(self)
            def set(self): self.real.set()
            def is_set(self): return self.real.is_set()
            def wait(self,timeout=None):
                waits.append(timeout)
                if self.real.is_set(): return True
                if not timeout: return False
                action=actions.pop(0) if actions else 'due'
                if action=='stop': self.real.set(); return True
                if action=='exception': raise OSError('injected wait failure')
                clock[0]+=timeout if action=='due' else action
                return False
        def capture(*args,**kwargs):
            captures.append(clock[0]);clock[0]+=next(delays,0)
            return 'unused.png','unused.json',dict(state_id=str(len(captures)),monotonic_ns=int(clock[0]*1e9),geometry_stable_during_capture=False)
        fake_threading=types.SimpleNamespace(Event=Event,Lock=threading.Lock,Thread=threading.Thread)
        fake_time=types.SimpleNamespace(monotonic=lambda:clock[0],monotonic_ns=lambda:int(clock[0]*1e9))
        kwargs=dict(capture_factory=lambda:types.SimpleNamespace(capture=capture),observe=lambda *a:None,
            count=count,interval=1,max_input_age=1,max_result_age=2,evidence_dir=Path('.'),archive_state=lambda _:None)
        if module is readonly_longrun:kwargs.update(max_duration=max_duration,record_sink=lambda _:None)
        with patch.object(module,'threading',fake_threading),patch.object(module,'time',fake_time):
            result=(module.run_long_readonly if module is readonly_longrun else module.run_readonly_pipeline)(**kwargs)
        return result,waits,captures

    def test_multiple_early_wakes_retry_remaining_wait_before_capture(self):
        for module in (readonly_longrun,readonly_pipeline):
            with self.subTest(module=module.__name__):
                result,waits,captures=self.exercise(module,[.25,.25,'due'])
                self.assertEqual(result['errors'],[]);self.assertEqual(captures,[100.,101.]);self.assertEqual(waits,[0,1,.75,.5])

    def test_stop_after_early_wake_prevents_capture(self):
        for module in (readonly_longrun,readonly_pipeline):
            with self.subTest(module=module.__name__):
                result,waits,captures=self.exercise(module,[.25,'stop'])
                self.assertEqual(result['errors'],[]);self.assertEqual(captures,[100.]);self.assertEqual(waits,[0,1,.75])

    def test_wait_exception_is_reported(self):
        for module in (readonly_longrun,readonly_pipeline):
            with self.subTest(module=module.__name__):
                result,_,captures=self.exercise(module,['exception'])
                self.assertEqual(captures,[100.]);self.assertIn('injected wait failure',str(result['errors']))

    def test_crossing_deadline_prevents_new_capture(self):
        result,_,captures=self.exercise(readonly_longrun,[10],max_duration=2)
        self.assertEqual(captures,[100.]);self.assertEqual(result['errors'],[])
        self.assertTrue(any(r['kind']=='duration_limit' for r in result['records']))

    def test_late_capture_skips_slots_without_backlog(self):
        for module in (readonly_longrun,readonly_pipeline):
            with self.subTest(module=module.__name__):
                result,waits,captures=self.exercise(module,[3.4,'due'],count=3)
                self.assertEqual(result['errors'],[]);self.assertEqual(captures,[100.,103.4,104.]);self.assertAlmostEqual(waits[-1],.6)
                records=[r for r in result['records'] if r['kind']=='captured']
                self.assertEqual([r['skipped_capture_slots'] for r in records],[0,2,0])

    def test_actual_thread_event_interrupts_pending_wait_on_consumer_error(self):
        for module in (readonly_longrun,readonly_pipeline):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temp:
                image=Path(temp)/'image.png';Image.new('RGB',(1,1)).save(image);captures=[]
                def capture(*a,**k):
                    captures.append(time.monotonic());return image,image,dict(state_id='1',monotonic_ns=time.monotonic_ns(),geometry_stable_during_capture=True,display_mapping=dict(validated=True),foreground_stable=True,foreground_game_or_scaler=True)
                wait_entered=threading.Event()
                class InstrumentedEvent:
                    def __init__(self):self.real=threading.Event()
                    def set(self):self.real.set()
                    def is_set(self):return self.real.is_set()
                    def wait(self,timeout=None):
                        # Signal the positive wait call before delegating to the real event.
                        # The stop may race the underlying wait, so this establishes the
                        # wait/cancellation ordering, not entry into a kernel sleep.
                        if timeout and timeout>0:wait_entered.set()
                        return self.real.wait(timeout)
                def observe(*a):
                    if not wait_entered.wait(2):raise AssertionError('producer never reached the next wait')
                    raise RuntimeError('injected inference failure')
                kwargs=dict(capture_factory=lambda:types.SimpleNamespace(capture=capture),observe=observe,count=2,interval=5,max_input_age=1,max_result_age=2,evidence_dir=Path(temp),archive_state=lambda _:None)
                if module is readonly_longrun:kwargs.update(max_duration=10,record_sink=lambda _:None)
                fake_threading=types.SimpleNamespace(Event=InstrumentedEvent,Lock=threading.Lock,Thread=threading.Thread)
                start=time.monotonic()
                with patch.object(module,'threading',fake_threading):
                    result=(module.run_long_readonly if module is readonly_longrun else module.run_readonly_pipeline)(**kwargs)
                self.assertTrue(wait_entered.is_set());self.assertLess(time.monotonic()-start,3)
                self.assertEqual(len(captures),1);self.assertIn('injected inference failure',str(result['errors']))

if __name__=='__main__': unittest.main()
