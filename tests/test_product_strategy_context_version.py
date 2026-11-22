"""Offline resource strategy context version and capture continuity regression."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from pvz_agent.resource_threat_v2 import ThreatCheckedContinuityFrames
from pvz_agent.resource_runtime import ResourceStrategySupervisor
from pvz_agent.readonly_longrun import run_long_readonly
from test_continuity_frames import metadata
from test_resource_review import resource_frame
from test_strategy_review import strategy
FIXTURES=json.loads((ROOT/'evidence/product-review-race-v1/revision-v3/unsupported-fixtures.json').read_text(encoding='utf-8'))
POLICY={'left_warning_x':200,'other_lane_margin_px':80}

class ImmediateSchedule:
    def __init__(self,start,interval):self.next_due=0
    def completed(self,now):return 0

class ContextVersionTests(unittest.TestCase):
    def make(self):
        now=[10.];registry=ThreatCheckedContinuityFrames(clock=lambda:now[0]);calls=[];events=[]
        def backend(s):calls.append(s['state_id']);return {'strategy':strategy(s['state_id'],goal='collect_resources',focus_row=None)}
        supervisor=ResourceStrategySupervisor(registry,backend,policy=POLICY,record_sink=events.append,clock=lambda:now[0],max_requests=2,min_request_interval=5.)
        self.addCleanup(supervisor.stop)
        def publish(sid,t,mutate=lambda s:None,*,pub_now=None):
            now[0]=t;registry.put({'metadata':metadata(sid,t)});ticket=registry.take();state=resource_frame(sid,t);mutate(state)
            value=registry.publish_observation(ticket,state,captured_at=t,now=t if pub_now is None else pub_now,max_age=2,input_age_ms=0)
            return value
        return now,registry,supervisor,calls,events,publish
    def test_each_unsupported_frame_is_visible_but_old_candidate_never_revives(self):
        for channel,fixture in FIXTURES.items():
            group,key=channel.split('.')
            for item in ({},fixture['item']):
                with self.subTest(channel=channel,item='empty' if not item else 'emitted'):
                    now,r,s,calls,events,pub=self.make();pub('old',10.);s.poll();s.reviewer._future.result(timeout=2)
                    v=pub('unsupported',10.5,lambda state:state[group].__setitem__(key,[copy.deepcopy(item)]))
                    observed=r.current(10.5);self.assertEqual(v['readonly_runtime']['publication_at_completion'],'published')
                    self.assertEqual(observed[group][key],[item]);self.assertEqual(observed['continuity']['epoch'],0)
                    version=getattr(r,'_strategy_context_version',0)
                    pub('new',11.);s.poll()
                    matched=[e for e in events if e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match']
                    self.assertEqual((version,len(matched)),(1,0))
    def test_new_clear_frame_can_request_and_ordinary_head_remains_normal(self):
        now,r,s,calls,events,pub=self.make();pub('old',10.);s.poll();s.reviewer._future.result(timeout=2)
        pub('unsupported',10.5,lambda st:st['vehicles'].__setitem__('candidates',[copy.deepcopy(FIXTURES['vehicles.candidates']['item'])]))
        pub('clear',11.);s.poll();pub('p125',12.5);pub('p14',14.);pub('recovered',15.);s.poll()
        s.reviewer._future.result(timeout=2);pub('new',15.5);s.poll()
        matches=[e['review']['source_state_id'] for e in events if e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match']
        self.assertEqual(calls,['old','recovered']);self.assertEqual(matches,['recovered']);self.assertEqual(r.current(15.5)['continuity']['epoch'],0)
    def test_inflight_response_across_unsupported_never_retained(self):
        now,r,s,calls,events,pub=self.make();entered=threading.Event();release=threading.Event()
        def backend(st):entered.set();self.assertTrue(release.wait(3));return {'strategy':strategy(st['state_id'],goal='collect_resources',focus_row=None)}
        s.reviewer.backend=backend
        try:
            pub('old',10.);s.poll();self.assertTrue(entered.wait(2));pub('bad',10.5,lambda st:st['zombies'].__setitem__('uncertain_water_candidates',[{}]));pub('new',11.)
            release.set();s.reviewer._future.result(timeout=2);s.poll()
            self.assertIn('invalidated_response',[e['kind'] for e in s.reviewer.events()]);self.assertIsNone(s.reviewer.review())
        finally:release.set()
    def test_malformed_channels_still_break_epoch_and_preserve_diagnostic_scene(self):
        for group,key in (tuple(c.split('.')) for c in FIXTURES):
            for form in ('missing','none'):
                with self.subTest(channel=group+'.'+key,form=form):
                    now,r,s,calls,events,pub=self.make();pub('old',10.);s.poll();s.reviewer._future.result(timeout=2)
                    def mutate(st):
                        if form=='missing':del st[group][key]
                        else:st[group][key]=None
                    v=pub('bad',10.5,mutate)
                    self.assertEqual(v['readonly_runtime']['publication_at_completion'],'continuity_rejected');self.assertEqual(v['scene']['value'],'playing')
                    self.assertIsNone(r.current(10.5));self.assertEqual(r.continuity_events()[-1]['epoch'],1)
                    pub('new',11.);s.poll()
                    self.assertFalse(any(e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match' for e in events))
    def test_unsupported_source_is_never_dispatched(self):
        for channel,fixture in FIXTURES.items():
            with self.subTest(channel=channel):
                now,r,s,calls,events,pub=self.make();group,key=channel.split('.')
                pub('unsupported',10.,lambda st:st[group].__setitem__(key,[copy.deepcopy(fixture['item'])]));s.poll()
                self.assertEqual(calls,[]);self.assertEqual(s.reviewer.requests,0)
    def test_committed_match_cannot_suppress_new_request_after_unsupported(self):
        now,r,s,calls,events,pub=self.make();pub('old',10.);s.poll();s.reviewer._future.result(timeout=2);pub('good',10.5)
        entered=threading.Event();release=threading.Event();errors=[]
        def sink(value):
            events.append(value)
            if value.get('kind')=='strategy_review':entered.set();self.assertTrue(release.wait(3))
        def poll():
            try:s.poll()
            except BaseException as exc:errors.append(repr(exc))
        s.sink=sink;thread=threading.Thread(target=poll)
        try:
            thread.start();self.assertTrue(entered.wait(2))
            pub('unsupported',11.,lambda st:st['zombies'].__setitem__('uncertain_edge_presence',[{}]))
            pub('p125',12.5);pub('p14',14.);pub('recovered',15.);release.set();thread.join(3)
            self.assertFalse(thread.is_alive());self.assertEqual(errors,[]);s.reviewer._future.result(timeout=2)
            self.assertEqual(calls,['old','recovered']);self.assertEqual(r.current(15.)['continuity']['epoch'],0)
        finally:release.set();thread.join(3)
    def test_superseded_or_expired_unsupported_frames_do_not_increment(self):
        now,r,s,calls,events,pub=self.make();pub('old',10.);now[0]=10.5;r.put({'metadata':metadata('bad',10.5)});bad=r.take()
        now[0]=11.;r.put({'metadata':metadata('new',11.)});state=resource_frame('bad',10.5);state['vehicles']['tracks']=[{}]
        v=r.publish_observation(bad,state,captured_at=10.5,now=11.,max_age=2,input_age_ms=0)
        self.assertEqual(v['readonly_runtime']['publication_at_completion'],'superseded');self.assertEqual(getattr(r,'_strategy_context_version',0),0)
        latest=r.take();state=resource_frame('new',11.);state['vehicles']['tracks']=[{}]
        v=r.publish_observation(latest,state,captured_at=11.,now=13.1,max_age=2,input_age_ms=0)
        self.assertEqual(v['readonly_runtime']['publication_at_completion'],'expired');self.assertEqual(getattr(r,'_strategy_context_version',0),0)
    def test_continuity_listener_callback_does_not_hold_registry_lock(self):
        now,r,s,calls,events,pub=self.make();entered=threading.Event();release=threading.Event();acquired=threading.Event()
        def listener(reason):entered.set();self.assertTrue(release.wait(3))
        r.bind_invalidator(listener);now[0]=10.;r.put({'metadata':metadata('old',10.)});ticket=r.take();wrong=resource_frame('wrong',10.)
        worker=threading.Thread(target=lambda:r.publish_observation(ticket,wrong,captured_at=10.,now=10.,max_age=2,input_age_ms=0))
        def acquire():
            with r._condition:acquired.set()
        try:
            worker.start();self.assertTrue(entered.wait(2));probe=threading.Thread(target=acquire);probe.start()
            self.assertTrue(acquired.wait(2));release.set();worker.join(2);probe.join(2);self.assertFalse(worker.is_alive())
        finally:release.set();worker.join(3)

class RunnerRegressionTests(unittest.TestCase):
    def test_unsupported_frame_archived_without_tracking_epoch_change(self):
        archive=[];records=[];gates=[threading.Event(),threading.Event()];index=[0];tracking=[]
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'synthetic.png';Image.new('RGB',(10,10),'green').save(image)
            def capture(*args,**kwargs):
                i=index[0];index[0]+=1
                if i:self.assertTrue(gates[i-1].wait(2))
                return image,Path(temp)/f'{i}.json',metadata(str(i),time.monotonic())
            def observe(image,meta,*args,tracking_epoch):
                tracking.append(tracking_epoch);state=resource_frame(meta['state_id'],meta['monotonic_ns']/1e9)
                state.update(input_sent=False,action_eligible=False)
                if meta['state_id']=='1':state['vehicles']['candidates']=[copy.deepcopy(FIXTURES['vehicles.candidates']['item'])]
                return state
            def save(state):
                archive.append(state)
                if len(archive)<=2:gates[len(archive)-1].set()
            registry=ThreatCheckedContinuityFrames()
            with patch('pvz_agent.readonly_longrun.SkippingSchedule',ImmediateSchedule):
                result=run_long_readonly(capture_factory=lambda:SimpleNamespace(capture=capture),observe=observe,count=3,interval=.01,
                    max_input_age=1,max_result_age=2,max_duration=3,evidence_dir=Path(temp),archive_state=save,record_sink=records.append,
                    frame_registry=registry,bind_tracking_continuity=True)
        self.assertEqual(result['errors'],[]);self.assertEqual(len(archive),3);self.assertEqual(tracking,[0,0,0])
        self.assertEqual(archive[1]['readonly_runtime']['publication_at_completion'],'published')
        self.assertEqual(archive[1]['continuity']['epoch'],0);self.assertEqual(registry._strategy_context_version,1)
        self.assertEqual(archive[1]['vehicles']['candidates'][0],FIXTURES['vehicles.candidates']['item'])
        self.assertTrue(all(state['input_sent'] is False and state['action_eligible'] is False for state in archive))

    def test_bad_channel_archives_and_runner_continues_without_scene_crash(self):
        archive=[];records=[];first=threading.Event();index=[0]
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'synthetic.png';Image.new('RGB',(10,10),'green').save(image)
            def capture(*args,**kwargs):
                i=index[0];index[0]+=1
                if i==1:self.assertTrue(first.wait(2))
                return image,Path(temp)/f'{i}.json',metadata(str(i),time.monotonic())
            def observe(image,meta,*args,tracking_epoch):
                state=resource_frame(meta['state_id'],meta['monotonic_ns']/1e9);state.update(input_sent=False,action_eligible=False)
                if meta['state_id']=='0':state['vehicles']['uncertain_candidates']=None
                return state
            def save(state):archive.append(state);first.set()
            with patch('pvz_agent.readonly_longrun.SkippingSchedule',ImmediateSchedule):
                result=run_long_readonly(capture_factory=lambda:SimpleNamespace(capture=capture),observe=observe,count=2,interval=.01,
                    max_input_age=1,max_result_age=2,max_duration=3,evidence_dir=Path(temp),archive_state=save,record_sink=records.append,
                    frame_registry=ThreatCheckedContinuityFrames(),bind_tracking_continuity=True)
        self.assertEqual(result['errors'],[]);self.assertEqual(len(archive),2)
        self.assertEqual(archive[0]['readonly_runtime']['publication_at_completion'],'continuity_rejected');self.assertEqual(archive[0]['scene']['value'],'playing')
        self.assertTrue(all(state['input_sent'] is False and state['action_eligible'] is False for state in archive))
        self.assertEqual(sum(e['kind']=='inferred' for e in records),2)

if __name__=='__main__':unittest.main()
