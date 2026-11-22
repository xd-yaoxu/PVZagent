"""Deterministic regression for concurrent product supervisor poll ownership."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import threading
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from pvz_agent.resource_threat_v2 import ThreatCheckedContinuityFrames
from pvz_agent.resource_runtime import ResourceStrategySupervisor
from test_continuity_frames import metadata
from test_resource_review import resource_frame
from test_strategy_review import strategy

class PollOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.now=[10.];self.registry=ThreatCheckedContinuityFrames(clock=lambda:self.now[0]);self.calls=[];self.events=[];self.failures=[]
        def backend(s):self.calls.append(s['state_id']);return {'strategy':strategy(s['state_id'],goal='collect_resources',focus_row=None)}
        self.s=ResourceStrategySupervisor(self.registry,backend,policy={'left_warning_x':200,'other_lane_margin_px':80},record_sink=self.events.append,clock=lambda:self.now[0],max_requests=2,min_request_interval=5.)
        self.addCleanup(self.s.stop)
    def pub(self,sid,now):
        self.now[0]=now;self.registry.put({'metadata':metadata(sid,now)});t=self.registry.take()
        self.registry.publish_observation(t,resource_frame(sid,now),captured_at=now,now=now,max_age=2,input_age_ms=0)
    def ready(self):
        self.pub('old',10.);self.s.poll();self.s.reviewer._future.result(timeout=2)
        self.pub('p12',12.);self.pub('p135',13.5);self.pub('p15',15.)
    def run_poll(self,done):
        try:self.s.poll()
        except BaseException as e:self.failures.append(repr(e))
        finally:done.set()
    def test_two_polls_share_one_match_and_do_not_request_again(self):
        self.ready();first=threading.Event();release1=threading.Event();release2=threading.Event();outcome=threading.Event();done1=threading.Event();done2=threading.Event();count=[]
        def sink(e):
            self.events.append(e)
            if e.get('kind')=='strategy_review':
                count.append(e)
                if len(count)==1:first.set();self.assertTrue(release1.wait(3))
                else:outcome.set();self.assertTrue(release2.wait(3))
        self.s.sink=sink
        def second():self.run_poll(done2);outcome.set()
        a=threading.Thread(target=self.run_poll,args=(done1,));b=threading.Thread(target=second)
        try:
            a.start();self.assertTrue(first.wait(2));b.start();self.assertTrue(outcome.wait(2))
            release1.set();self.assertTrue(done1.wait(2));release2.set();self.assertTrue(done2.wait(2))
            a.join(2);b.join(2)
            self.assertEqual(self.failures,[]);self.assertEqual(len(count),1);self.assertEqual(self.s.reviewer.requests,1);self.assertEqual(self.calls,['old'])
        finally:release1.set();release2.set();a.join(3);b.join(3)
    def test_nested_poll_does_not_reenter_and_stop_reports_busy(self):
        self.ready();entered=threading.Event();release=threading.Event();done=threading.Event();events=[];nested=[False]
        def sink(e):
            if e.get('kind')=='strategy_review':
                events.append(e)
                if not nested[0]:
                    nested[0]=True;self.s.poll();entered.set();self.assertTrue(release.wait(3))
        self.s.sink=sink;t=threading.Thread(target=self.run_poll,args=(done,))
        try:
            t.start();self.assertTrue(entered.wait(2));invalidated=threading.Event()
            iv=threading.Thread(target=lambda:(self.registry.session_changed(),invalidated.set()));iv.start()
            self.assertTrue(invalidated.wait(2));self.assertFalse(self.s.stop()['workers_finished'])
            release.set();self.assertTrue(done.wait(2));iv.join(2);t.join(2)
            self.assertEqual(self.failures,[]);self.assertEqual(len(events),1)
        finally:release.set();t.join(3)

    def test_resource_v3_inherits_previous_and_rejects_tampered_history(self):
        from pvz_agent import resource_stream as resource
        current=json.loads((ROOT/resource.PRODUCT_RESOURCE_FREEZE).read_text(encoding='utf-8'))
        previous=json.loads((ROOT/resource.PREVIOUS_PRODUCT_RESOURCE_FREEZE).read_text(encoding='utf-8'))
        self.assertEqual(set(current['artifacts']),set(previous['artifacts']))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for rel in (resource.PRODUCT_RESOURCE_FREEZE,resource.PREVIOUS_PRODUCT_RESOURCE_FREEZE,resource.HISTORICAL_RESOURCE_FREEZE):
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,path)
            previous_path=root/resource.PREVIOUS_PRODUCT_RESOURCE_FREEZE;previous_path.write_bytes(previous_path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Previous resource freeze changed'):resource.create_resource_stream(root)

if __name__=='__main__':unittest.main()
