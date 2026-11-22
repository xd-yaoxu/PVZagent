"""Offline continuity binding through actual streams and read-only runner."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pvz_agent.continuity_frames import ContinuityFrames
from pvz_agent.readonly_longrun import run_long_readonly
from pvz_agent.vehicle_perception import VehicleObservationStream
from test_continuity_frames import metadata

CFG=json.loads((ROOT/'config/zombie-perception-bucket-v40.json').read_text(encoding='utf-8'))

class Detector:
    config=CFG
    def __init__(self):self.scene='playing';self.present=True
    def observe(self,image):
        return dict(scene=dict(value=self.scene),
            candidates=[dict(row=3,head_xy=[600,300],detection_id='h')] if self.present else [],
            vehicle_candidates=[dict(row=3,front_xy=[650,300],detection_id='v')] if self.present else [],
            uncertain_vehicle_candidates=[],vehicle_coverage='fake')

class BindingTests(unittest.TestCase):
    def setUp(self):self.detector=Detector();self.stream=VehicleObservationStream(self.detector)
    def obs(self,sid,t,epoch=None):
        return self.stream.observe(None,run_id='source',state_id=sid,capture_time='synthetic',
            frame_time_seconds=t,evidence=dict(image='fake',image_sha256='fake'),tracking_epoch=epoch)
    def ids(self,s):return (s['zombies']['candidates'][0]['track_id'],s['vehicles']['candidates'][0]['track_id'])
    def test_epoch_separates_head_vehicle_and_events(self):
        a=self.obs('a',10,0);b=self.obs('b',11,1)
        self.assertEqual(a['run_id'],'source');self.assertEqual(b['run_id'],'source')
        self.assertEqual(a['tracking_context']['tracking_run_id'],'source/continuity-0')
        self.assertEqual(b['tracking_context']['tracking_run_id'],'source/continuity-1')
        for old,new in zip(self.ids(a),self.ids(b)):self.assertNotEqual(old,new)
        self.assertTrue(self.ids(b)[0].startswith('source/continuity-1:'))
        self.assertTrue(self.ids(b)[1].startswith('source/continuity-1:vehicles:'))
        for channel in ('zombies','vehicles'):
            self.assertEqual({e['run_id'] for e in b[channel]['events']},{'source/continuity-1'})
            reset=next(e for e in b[channel]['events'] if e['type']=='run_reset')
            self.assertTrue(reset['previous_run_id'].startswith('source/continuity-0'))
    def test_same_epoch_short_loss_expiry_and_nonplaying(self):
        a=self.obs('a',0,0);ids=self.ids(a);self.detector.present=False
        lost=self.obs('lost',1,0)
        for group in ('zombies','vehicles'):self.assertEqual(lost[group]['tracks'][0]['status'],'lost')
        self.detector.present=True;b=self.obs('b',2.5,0);self.assertEqual(ids,self.ids(b))
        self.detector.present=False;self.obs('lost2',3,0);self.detector.present=True
        c=self.obs('c',5.0001,0)
        for old,new in zip(ids,self.ids(c)):self.assertNotEqual(old,new)
        self.detector.scene='pause';p=self.obs('pause',5.1,0)
        self.assertFalse(p['zombies']['tracks']);self.assertFalse(p['vehicles']['tracks'])
        self.detector.scene='playing';d=self.obs('d',5.2,0)
        for old,new in zip(self.ids(c),self.ids(d)):self.assertNotEqual(old,new)
    def test_none_legacy_and_malformed_rejected_before_detector(self):
        state=self.obs('a',0)
        self.assertNotIn('tracking_context',state);self.assertEqual(self.ids(state),('source:z1','source:vehicles:z1'))
        for value in (True,False,-1,0.0,'0',[],{}):
            with self.subTest(value=value),self.assertRaises(ValueError):self.obs('b',1,value)
        for run_id in (None,'','  ',17):
            before=copy.deepcopy((self.stream.tracker.__dict__,self.stream.vehicle_tracker.geometry.__dict__))
            with self.subTest(run_id=run_id),patch.object(self.detector,'observe',side_effect=AssertionError('detector called')) as detector:
                with self.assertRaises(ValueError):
                    self.stream.observe(None,run_id=run_id,state_id='b',capture_time='fake',frame_time_seconds=1,
                        evidence=dict(image='fake',image_sha256='fake'),tracking_epoch=1)
                detector.assert_not_called()
            self.assertEqual(before,(self.stream.tracker.__dict__,self.stream.vehicle_tracker.geometry.__dict__))
        self.assertEqual(self.stream.tracker.last_time,0)
    def test_omitted_and_none_have_identical_legacy_output(self):
        other=VehicleObservationStream(Detector())
        kwargs=dict(run_id='source',state_id='same',capture_time='synthetic',frame_time_seconds=0,
                    evidence=dict(image='fake',image_sha256='fake'))
        a=self.stream.observe(None,**kwargs);b=other.observe(None,tracking_epoch=None,**kwargs)
        for value in (a,b):value.pop('processing_ms');value.pop('perceived_at')
        self.assertEqual(a,b)

    def test_real_registry_breaks_on_session_and_focus(self):
        for reason in ('session','focus'):
            with self.subTest(reason=reason):
                now=[10.];registry=ContinuityFrames(clock=lambda:now[0]);stream=VehicleObservationStream(Detector())
                def frame(sid):
                    registry.put(dict(metadata=metadata(sid,now[0])));ticket=registry.take()
                    state=stream.observe(None,run_id='source',state_id=str(sid),capture_time='fake',
                        frame_time_seconds=now[0],evidence=dict(image='fake',image_sha256='fake'),
                        tracking_epoch=ticket.item['_continuity_epoch'])
                    return registry.publish_observation(ticket,state,captured_at=now[0],now=now[0],max_age=2,input_age_ms=0)
                old=frame('a');now[0]=10.5
                if reason=='session':registry.session_changed()
                else:
                    bad=metadata('bad',now[0]);bad['foreground_game_or_scaler']=False
                    registry.put(dict(metadata=bad));registry.take()
                now[0]=11.;new=frame('b')
                self.assertEqual(new['readonly_runtime']['publication_at_completion'],'published')
                self.assertEqual(new['continuity']['capture_epoch'],1)
                for before,after in zip(self.ids(old),self.ids(new)):self.assertNotEqual(before,after)

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.dir=Path(self.temp.name);self.image=self.dir/'frame.png';Image.new('RGB',(10,10)).save(self.image)
    def run_args(self,registry,capture,observe,**extra):
        args=dict(capture_factory=lambda:SimpleNamespace(capture=capture),observe=observe,count=1,interval=.02,
            max_input_age=1,max_result_age=2,max_duration=2,evidence_dir=self.dir,archive_state=lambda s:None,
            record_sink=lambda r:None,frame_registry=registry)
        args.update(extra);return args
    def frame(self,sid,**kw):
        now=time.monotonic();m=metadata(str(sid),now);m.update(kw)
        return self.image,self.dir/'meta.json',m
    def test_inflight_old_ticket_rejected_new_ticket_gets_new_namespace(self):
        first_entered=threading.Event();new_put=threading.Event();second_seen=threading.Event()
        class Registry(ContinuityFrames):
            def put(self,item):
                result=super().put(item)
                if item['metadata']['state_id']=='1':new_put.set()
                return result
        registry=Registry();stream=VehicleObservationStream(Detector());archived=[];i=[0]
        def capture(*a,**kw):
            n=i[0];i[0]+=1
            if n==1:
                self.assertTrue(first_entered.wait(2));registry.session_changed()
            if n==2:self.assertTrue(second_seen.wait(2))
            # Capture metadata cannot override the registry's epoch stamp.
            return self.frame(n,_continuity_epoch=999,**({'foreground_stable':False} if n==2 else {}))
        def observe(image,meta,path,image_file,*,tracking_epoch):
            if meta['state_id']=='0':
                first_entered.set();self.assertTrue(new_put.wait(2))
            result=stream.observe(image,run_id='source',state_id=meta['state_id'],capture_time='fake',
                frame_time_seconds=meta['monotonic_ns']/1e9,evidence=dict(image='fake',image_sha256='fake'),tracking_epoch=tracking_epoch)
            return result
        def archive(state):
            archived.append(state)
            if state['state_id']=='1':second_seen.set()
        result=run_long_readonly(**self.run_args(registry,capture,observe,count=3,archive_state=archive,bind_tracking_continuity=True))
        self.assertFalse(result['errors']);self.assertEqual(len(archived),2)
        self.assertNotEqual(archived[0]['readonly_runtime']['publication_at_completion'],'published')
        self.assertEqual(archived[1]['readonly_runtime']['publication_at_completion'],'published')
        self.assertEqual([s['tracking_context']['epoch'] for s in archived],[0,1])
        for group in ('zombies','vehicles'):
            self.assertNotEqual(archived[0][group]['candidates'][0]['track_id'],archived[1][group]['candidates'][0]['track_id'])
        self.assertEqual(archived[1]['continuity']['capture_epoch'],1)
    def test_missing_or_malformed_ticket_epoch_fails_closed(self):
        class BadRegistry(ContinuityFrames):
            def __init__(self,value):super().__init__();self.value=value
            def take(self):
                ticket=super().take()
                if ticket is not None:
                    if self.value=='missing':ticket.item.pop('_continuity_epoch')
                    else:ticket.item['_continuity_epoch']=self.value
                return ticket
        for value in ('missing',None,True,-1,1.0,'1'):
            seen=[];reg=BadRegistry(value)
            result=run_long_readonly(**self.run_args(reg,lambda *a,**k:self.frame(0),lambda *a,**k:seen.append(1),bind_tracking_continuity=True))
            self.assertTrue(result['errors'],str(value));self.assertEqual(seen,[]);self.assertIsNone(reg.current(time.monotonic()))
    def test_default_four_argument_callback_and_strict_switch(self):
        seen=[]
        def observe(a,b,c,d):
            seen.append(4);return dict(input_sent=False,action_eligible=False,scene=dict(value='playing'))
        from pvz_agent.snapshot_frames import SnapshotFrames
        result=run_long_readonly(**self.run_args(SnapshotFrames(),lambda *a,**k:self.frame(0),observe))
        self.assertFalse(result['errors']);self.assertEqual(seen,[4])
        for invalid in (None,0,1,'true'):
            with self.assertRaises(ValueError):run_long_readonly(**self.run_args(SnapshotFrames(),lambda *a,**k:self.frame(0),observe,bind_tracking_continuity=invalid))

class ProductEntryTests(unittest.TestCase):
    def test_real_factory_constructs_without_capture_or_provider(self):
        from pvz_agent.resource_stream import create_resource_stream
        with patch('pvz_agent.pinned_capture.PinnedWindowsCapture',side_effect=AssertionError('capture')), \
             patch('pvz_agent.strategy_profile.ProfiledStrategyProvider',side_effect=AssertionError('provider')):
            self.assertIsNotNone(create_resource_stream(ROOT))
    def test_entry_enables_and_passes_epoch_to_stream(self):
        module=importlib.import_module('day06_product_resource_live');received=[]
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as tmp:
            base=Path(tmp);(base/'evidence/day04').mkdir(parents=True);(base/'evidence/day04/STOP').touch()
            from pvz_agent.product_startup import REQUIRED,OLD_FREEZE,MANIFEST
            for rel in (*REQUIRED,OLD_FREEZE,MANIFEST):
                target=base/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
            cfg=dict(image_authorization_record='auth.json',image_transfer_authorized=True,base_url='fake',model='fake')
            class Stream:
                def observe(self,image,**kwargs):received.append(kwargs);return {}
            def load(path):
                name=Path(path).name
                if name=='freeze.json':return {'artifacts':{}}
                if name=='resource-attention-v1.json':return {'policy':{}}
                if name=='auth.json':return {'user_reply':'允许','base_url':'fake'}
                if name=='environment.json':return {}
                raise AssertionError(name)
            def runner(**kwargs):
                self.assertIs(kwargs['bind_tracking_continuity'],True)
                kwargs['observe'](None,dict(state_id='1',capture_time='fake',monotonic_ns=1000000000,image_sha256='hash'),
                    base/'m.json',base/'i.png',tracking_epoch=7)
                return dict(records=[],errors=[],strategy=dict(requests=0,workers_finished=True))
            with patch.multiple(module,ROOT=base,load=load,resolve_product_strategy=lambda:cfg,
                create_product_strategy=lambda _:object(),create_bucket_stream=lambda _:Stream(),
                run_goal_readonly=runner,write=lambda *args:None),patch.object(sys,'argv',['test','--out','out']):module.main()
        self.assertEqual(received[0]['tracking_epoch'],7)

if __name__=='__main__':unittest.main()
