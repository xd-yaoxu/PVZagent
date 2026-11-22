"""Offline regression through the product resource supervisor and snapshot registry."""
import copy
import importlib
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pvz_agent.resource_threat_v2 import ThreatCheckedContinuityFrames
from pvz_agent.resource_runtime import ResourceStrategySupervisor
from test_continuity_frames import metadata
from test_resource_review import resource_frame
from test_strategy_review import strategy


class ProductThreatGateTests(unittest.TestCase):
    def setUp(self):
        self.now=[10.];self.registry=ThreatCheckedContinuityFrames(clock=lambda:self.now[0]);self.calls=[];self.events=[]
        def backend(s):
            self.calls.append(s['state_id'])
            return dict(strategy=strategy(s['state_id'],goal='collect_resources',focus_row=None))
        self.supervisor=ResourceStrategySupervisor(self.registry,backend,policy={'left_warning_x':200,'other_lane_margin_px':80},
            record_sink=self.events.append,clock=lambda:self.now[0],max_requests=1)
        self.addCleanup(self.supervisor.stop)

    def publish(self,sid,state):
        self.registry.put({'metadata':metadata(sid,self.now[0])});ticket=self.registry.take()
        return self.registry.publish_observation(ticket,state,captured_at=self.now[0],now=self.now[0],max_age=2,input_age_ms=0)

    def make(self,sid): return resource_frame(sid,self.now[0])

    def test_unknown_source_never_dispatches(self):
        source=self.make('old');source['vehicles']['uncertain_candidates']=None
        self.publish('old',source);self.supervisor.poll()
        self.assertEqual(self.supervisor.requests,0);self.assertEqual(self.calls,[])

    def test_unknown_current_then_recovery_cannot_revive_old_review(self):
        source=self.make('old');self.publish('old',source);self.supervisor.poll()
        self.supervisor.reviewer._future.result(timeout=2)
        self.now[0]=10.5;bad=self.make('bad');bad['zombies']['uncertain_edge_presence']=None
        self.publish('bad',bad);self.supervisor.poll()
        self.now[0]=11.;self.publish('new',self.make('new'));self.supervisor.poll()
        matched=[e for e in self.events if e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match']
        self.assertEqual(matched,[])

    def test_known_empty_channels_preserve_readonly_review(self):
        source=self.make('old');self.publish('old',source);self.supervisor.poll()
        self.supervisor.reviewer._future.result(timeout=2)
        self.now[0]=10.5;current=self.make('new');self.publish('new',current);self.supervisor.poll()
        matches=[e for e in self.events if e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match']
        self.assertEqual(len(matches),1);self.assertFalse(matches[0]['review']['action_eligible'])

    def test_all_required_channels_fail_closed_for_source_and_current(self):
        channels=[('vehicles','candidates'),('vehicles','tracks'),('vehicles','uncertain_candidates'),
                  ('zombies','candidates'),('zombies','uncertain_edge_presence'),('zombies','uncertain_water_candidates')]
        for side in ('source','current'):
            for group,key in channels:
                for malformed in ('missing',None,{},False,[None]):
                    with self.subTest(side=side,group=group,key=key,malformed=malformed):
                        now=[10.];registry=ThreatCheckedContinuityFrames(clock=lambda:now[0]);calls=[];events=[]
                        def backend(s):
                            calls.append(s['state_id'])
                            return dict(strategy=strategy(s['state_id'],goal='collect_resources',focus_row=None))
                        supervisor=ResourceStrategySupervisor(registry,backend,policy={'left_warning_x':200,'other_lane_margin_px':80},
                            record_sink=events.append,clock=lambda:now[0],max_requests=1)
                        def publish(sid,s):
                            registry.put({'metadata':metadata(sid,now[0])});ticket=registry.take()
                            return registry.publish_observation(ticket,s,captured_at=now[0],now=now[0],max_age=2,input_age_ms=0)
                        try:
                            if side=='current':
                                publish('old',resource_frame('old',now[0]));supervisor.poll();supervisor.reviewer._future.result(timeout=2)
                                now[0]=10.5
                            bad=resource_frame('bad',now[0])
                            if malformed=='missing':del bad[group][key]
                            else:bad[group][key]=copy.deepcopy(malformed)
                            value=publish('bad',bad)
                            self.assertEqual(value['readonly_runtime']['publication_at_completion'],'continuity_rejected')
                            supervisor.poll()
                            if side=='source':self.assertEqual(calls,[])
                            else:
                                now[0]=11.;publish('new',resource_frame('new',now[0]));supervisor.poll()
                                self.assertFalse(any(e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match' for e in events))
                        finally:supervisor.stop()

    def test_published_and_consumed_snapshots_are_isolated(self):
        source=self.make('old');self.publish('old',source)
        source['vehicles']['uncertain_candidates']=None
        copy_from_registry=self.registry.current(self.now[0]);copy_from_registry['zombies']['uncertain_edge_presence']=None
        self.supervisor.poll();self.supervisor.reviewer._future.result(timeout=2)
        self.now[0]=10.5;self.publish('new',self.make('new'));self.supervisor.poll()
        self.assertTrue(any(e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match' for e in self.events))

    def test_review_in_progress_does_not_emit_match_after_invalidating_current(self):
        import pvz_agent.resource_review as review_module
        self.publish('old',self.make('old'));self.supervisor.poll();self.supervisor.reviewer._future.result(timeout=2)
        self.now[0]=10.5;self.publish('good',self.make('good'))
        entered=threading.Event();release=threading.Event();invalidated=threading.Event();errors=[]
        original_review=review_module.resource_evidence;original_notify=self.registry._notify
        def paused(*args):
            entered.set()
            if not release.wait(3):raise TimeoutError('review test barrier')
            return original_review(*args)
        def notify(listener,reason):
            if reason=='threat_context_invalid':invalidated.set()
            return original_notify(listener,reason)
        def invalidate():
            try:
                self.now[0]=10.6;bad=self.make('bad');bad['vehicles']['uncertain_candidates']=None
                self.publish('bad',bad)
            except Exception as exc:errors.append(type(exc).__name__)
        with patch.object(review_module,'resource_evidence',side_effect=paused),patch.object(self.registry,'_notify',side_effect=notify):
            consumer=threading.Thread(target=self.supervisor.poll);consumer.start()
            self.assertTrue(entered.wait(2));producer=threading.Thread(target=invalidate);producer.start()
            try:
                self.assertTrue(invalidated.wait(2));self.assertIsNone(self.registry.current(10.6))
            finally:
                release.set();consumer.join(3);producer.join(3)
        self.assertFalse(consumer.is_alive());self.assertFalse(producer.is_alive());self.assertEqual(errors,[])
        matched=[e for e in self.events if e.get('kind')=='strategy_review' and e['review']['disposition']=='selected_observations_match']
        self.assertEqual(matched,[])

    def test_product_entry_keeps_legacy_and_v2_is_disabled(self):
        from pvz_agent.resource_runtime import run_resource_readonly
        from pvz_agent.resource_runtime_v2 import run_resource_readonly as disabled_v2
        with patch('pvz_agent.pinned_capture.PinnedWindowsCapture') as capture, \
             patch('pvz_agent.strategy_profile.ProfiledStrategyProvider') as provider, \
             patch('pvz_agent.resource_stream.create_resource_stream') as stream, \
             patch.object(ResourceStrategySupervisor,'start') as start:
            module=importlib.import_module('day06_product_resource_live')
            self.assertIs(module.run_goal_readonly,run_resource_readonly)
            with self.assertRaisesRegex(RuntimeError,'v2_disabled_pending_review_invalidation_protocol'): disabled_v2()
            capture.assert_not_called();provider.assert_not_called();stream.assert_not_called();start.assert_not_called()


if __name__=='__main__':unittest.main()
