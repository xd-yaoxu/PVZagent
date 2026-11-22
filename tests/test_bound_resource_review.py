"""Offline event-barrier tests for the actual product strategy wiring."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
sys.path.insert(0,str(ROOT/'scripts'))
from pvz_agent.resource_runtime import ResourceStrategySupervisor,run_resource_readonly
from pvz_agent.resource_threat_v2 import ThreatCheckedContinuityFrames
from pvz_agent.continuity_frames import ContinuityFrames
from test_continuity_frames import metadata
from test_resource_review import resource_frame
from test_strategy_review import strategy
import pvz_agent.resource_review as review_module
POLICY={'left_warning_x':200,'other_lane_margin_px':80}

class BoundProductTests(unittest.TestCase):
    def setUp(self):
        self.now=[10.];self.events=[];self.calls=[];self.registry=ThreatCheckedContinuityFrames(clock=lambda:self.now[0]);self.goal='collect_resources'
        def backend(s):
            self.calls.append(s['state_id']);return {'strategy':strategy(s['state_id'],goal=self.goal,focus_row=1 if self.goal=='defend_lane' else None)}
        self.s=ResourceStrategySupervisor(self.registry,backend,policy=POLICY,record_sink=self.events.append,clock=lambda:self.now[0],max_requests=2,min_request_interval=5.)
        self.addCleanup(self.s.stop)
    def pub(self,sid):
        self.registry.put({'metadata':metadata(sid,self.now[0])});t=self.registry.take()
        return self.registry.publish_observation(t,resource_frame(sid,self.now[0]),captured_at=self.now[0],now=self.now[0],max_age=2,input_age_ms=0)
    def ready(self):
        self.pub('old');self.s.poll();self.s.reviewer._future.result(timeout=2);self.now[0]=10.5;self.pub('good')
    def test_plain_registry_rejected_and_source_snapshot_tampering_fails(self):
        with self.assertRaisesRegex(TypeError,'threat-checked'):ResourceStrategySupervisor(ContinuityFrames(),lambda s:None,policy=POLICY,record_sink=lambda s:None)
        self.pub('old');state,token=self.s.reviewer.snapshot();state['fields']['sun']['value']=999
        self.assertFalse(self.s.reviewer.submit(state,token));self.assertEqual(self.calls,[])
    def test_all_four_goals_normal_and_opaque_single_consume(self):
        for goal in ('collect_resources','defend_lane','hold','develop_economy'):
            with self.subTest(goal=goal):
                now=[10.];r=ThreatCheckedContinuityFrames(clock=lambda:now[0]);events=[]
                def backend(s):return {'strategy':strategy(s['state_id'],goal=goal,focus_row=1 if goal=='defend_lane' else None)}
                supervisor=ResourceStrategySupervisor(r,backend,policy=POLICY,record_sink=events.append,clock=lambda:now[0],max_requests=1)
                try:
                    def publish(sid):
                        r.put({'metadata':metadata(sid,now[0])});t=r.take();r.publish_observation(t,resource_frame(sid,now[0]),captured_at=now[0],now=now[0],max_age=2,input_age_ms=0)
                    publish('old');supervisor.poll();supervisor.reviewer._future.result(timeout=2);now[0]=10.5;publish('new')
                    review,ticket=supervisor.reviewer.review();self.assertEqual(review['disposition'],'selected_observations_match')
                    self.assertFalse(supervisor.reviewer.consume(('forged',)));self.assertTrue(supervisor.reviewer.consume(ticket));self.assertFalse(supervisor.reviewer.consume(ticket))
                    self.assertEqual(review['validity'],'historical_at_commit');self.assertFalse(review['action_eligible'])
                    if goal=='collect_resources':self.assertTrue(review['requires_fresh_pickup_proposal'])
                finally:supervisor.stop()
    def test_invalidate_replacement_and_expiry_during_comparison_do_not_commit(self):
        for mode in ('invalidate','replacement','expiry'):
            with self.subTest(mode=mode):
                now=[10.];r=ThreatCheckedContinuityFrames(clock=lambda:now[0]);events=[]
                s=ResourceStrategySupervisor(r,lambda st:{'strategy':strategy(st['state_id'],goal='collect_resources',focus_row=None)},policy=POLICY,record_sink=events.append,clock=lambda:now[0],max_requests=1)
                def pub(sid):
                    r.put({'metadata':metadata(sid,now[0])});t=r.take();r.publish_observation(t,resource_frame(sid,now[0]),captured_at=now[0],now=now[0],max_age=2,input_age_ms=0)
                entered=threading.Event();release=threading.Event();original=review_module.resource_evidence
                def slow(*a):entered.set();self.assertTrue(release.wait(3));return original(*a)
                try:
                    pub('old');s.poll();s.reviewer._future.result(timeout=2);now[0]=10.5;pub('good')
                    with patch.object(review_module,'resource_evidence',side_effect=slow):
                        thread=threading.Thread(target=s.poll);thread.start();self.assertTrue(entered.wait(2))
                        if mode=='invalidate':r.session_changed()
                        elif mode=='replacement':now[0]=11.;pub('replacement')
                        else:now[0]=12.5
                        release.set();thread.join(3);self.assertFalse(thread.is_alive())
                    self.assertFalse(any(x.get('kind')=='strategy_review' and x['review']['disposition']=='selected_observations_match' for x in events))
                finally:release.set();s.stop()
    def test_unknown_between_polls_cannot_revive(self):
        self.ready();self.now[0]=11.;bad=resource_frame('bad',11.);bad['vehicles']['uncertain_candidates']=None
        self.registry.put({'metadata':metadata('bad',11.)});t=self.registry.take();self.registry.publish_observation(t,bad,captured_at=11.,now=11.,max_age=2,input_age_ms=0)
        self.now[0]=11.5;self.pub('new');self.s.poll()
        self.assertFalse(any(e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match' for e in self.events))
    def test_committed_ticket_rechecks_replacement_and_expiry(self):
        self.ready();review,ticket=self.s.reviewer.review();self.now[0]=11.;self.pub('replacement')
        self.assertFalse(self.s.reviewer.consume(ticket))
        review,ticket=self.s.reviewer.review();self.now[0]=13.
        self.assertFalse(self.s.reviewer.consume(ticket))
    def test_close_revokes_committed_ticket(self):
        self.ready();review,ticket=self.s.reviewer.review();self.s.reviewer.close()
        self.assertFalse(self.s.reviewer.consume(ticket))
    def test_slow_sink_does_not_block_invalidation_and_stale_consumption(self):
        self.ready();entered=threading.Event();release=threading.Event()
        def slow_sink(e):
            self.events.append(e)
            if e.get('kind')=='strategy_review':entered.set();self.assertTrue(release.wait(3))
        self.s.sink=slow_sink;thread=threading.Thread(target=self.s.poll);thread.start();self.assertTrue(entered.wait(2))
        done=threading.Event();invalidator=threading.Thread(target=lambda:(self.registry.session_changed(),done.set()));invalidator.start()
        self.assertTrue(done.wait(2));self.now[0]=16.;self.pub('new');release.set();thread.join(3);invalidator.join(3)
        self.assertFalse(thread.is_alive());self.s.reviewer._future.result(timeout=2);self.assertEqual(self.calls,['old','new'])
        self.assertEqual([e for e in self.events if e.get('kind')=='strategy_review'][0]['review']['validity'],'historical_at_commit')
    def test_slow_backend_invalidation_and_concurrent_request_cap(self):
        entered=threading.Event();release=threading.Event()
        def backend(s):self.calls.append(s['state_id']);entered.set();self.assertTrue(release.wait(3));return {'strategy':strategy(s['state_id'],goal='collect_resources',focus_row=None)}
        self.s.reviewer.backend=backend;self.pub('old');threads=[threading.Thread(target=self.s.poll) for _ in range(8)]
        for t in threads:t.start()
        self.assertTrue(entered.wait(2))
        for t in threads:t.join(2)
        self.assertEqual(len(self.calls),1);self.registry.session_changed();release.set();self.s.reviewer._future.result(timeout=2)
        self.assertIn('invalidated_response',[e['kind'] for e in self.s.reviewer.events()]);self.assertIsNone(self.s.reviewer.review())
    def test_slow_executor_launch_does_not_block_invalidation_or_close(self):
        self.pub('old');entered=threading.Event();release=threading.Event();original=self.s.reviewer._executor.submit
        def paused(*a,**k):entered.set();self.assertTrue(release.wait(3));return original(*a,**k)
        with patch.object(self.s.reviewer._executor,'submit',side_effect=paused):
            worker=threading.Thread(target=self.s.poll);worker.start();self.assertTrue(entered.wait(2))
            done=threading.Event();invalidator=threading.Thread(target=lambda:(self.registry.session_changed(),done.set()));invalidator.start()
            self.assertTrue(done.wait(2));self.assertFalse(self.s.stop()['workers_finished'])
            release.set();worker.join(3);invalidator.join(3);self.assertFalse(worker.is_alive())
        self.assertEqual(self.calls,[])

class ActualFactoryTests(unittest.TestCase):
    def test_single_and_continuous_entries_use_real_factory_with_fake_capture(self):
        import product_single_readonly as single
        import product_continuous_readonly as continuous
        import pvz_agent.terra_single_smoke as capture_module
        import pvz_agent.continuous_local_readonly as continuous_module
        from test_terra_single_smoke import FakeCapture
        captured=[]
        def fake_capture(_):
            value=FakeCapture();captured.append(value);return value
        output=io.StringIO()
        with patch.object(capture_module,'WindowsClientCapture',side_effect=fake_capture), \
             patch.object(sys,'argv',['single']),contextlib.redirect_stdout(output):
            single.main()
        result=json.loads(output.getvalue());self.assertEqual(captured[0].grabs,1)
        self.assertFalse(result['input_sent']);self.assertFalse(result['action_eligible'])
        observed=[]
        def fake_run(**kw):
            observed.append((type(kw['stream']).__name__,kw['capture']))
            return 'duration_complete'
        output=io.StringIO()
        with patch.object(capture_module,'WindowsClientCapture',side_effect=fake_capture), \
             patch.object(continuous_module,'run_observations',side_effect=fake_run), \
             patch.object(sys,'argv',['continuous','--seconds','5']),contextlib.redirect_stdout(output):
            self.assertEqual(continuous.main(),0)
        self.assertEqual(observed[0][0],'ResourceCueStream');self.assertIs(observed[0][1],captured[1])
        self.assertFalse(json.loads(output.getvalue())['action_eligible'])

    def test_default_factory_uses_checked_registry_and_barrier_reproduction(self):
        now=[10.];holder={};events=[];entered=threading.Event();release=threading.Event();orig=review_module.resource_evidence
        def start(s):holder['s']=s;s.clock=lambda:now[0];s.reviewer.clock=lambda:now[0];s.registry.clock=lambda:now[0]
        def run_long_readonly(*,frame_registry,**kw):
            self.assertIsInstance(frame_registry,ThreatCheckedContinuityFrames);s=holder['s']
            def pub(sid):
                frame_registry.put({'metadata':metadata(sid,now[0])});t=frame_registry.take();frame_registry.publish_observation(t,resource_frame(sid,now[0]),captured_at=now[0],now=now[0],max_age=2,input_age_ms=0)
            pub('old');s.poll();s.reviewer._future.result(timeout=2);now[0]=10.5;pub('good')
            s.poll();self.assertTrue(any(e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match' for e in events));events.clear();s._last_key=None
            def slow(*a):entered.set();self.assertTrue(release.wait(3));return orig(*a)
            with patch.object(review_module,'resource_evidence',side_effect=slow):
                th=threading.Thread(target=s.poll);th.start();self.assertTrue(entered.wait(2));frame_registry.session_changed();release.set();th.join(3);self.assertFalse(th.is_alive())
            self.assertFalse(any(e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match' for e in events))
            now[0]=11.;pub('new');s.poll();return {'limits':[]}
        backend=lambda s:{'strategy':strategy(s['state_id'],goal='collect_resources',focus_row=None)}
        with patch.object(ResourceStrategySupervisor,'start',start),patch('pvz_agent.resource_runtime.run_long_readonly',side_effect=run_long_readonly):
            result=run_resource_readonly(backend=backend,policy=POLICY,strategy_record_sink=events.append,max_requests=1)
        self.assertFalse(result['strategy']['input_sent'])

if __name__=='__main__':unittest.main()
