"""Resource attention requires current bound cues, not an unchanged counter."""
import copy
from pathlib import Path
import sys
import threading
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pvz_agent.resource_review import ResourceStrategicReviewer,resource_evidence,ReviewBlocked
from pvz_agent.resource_runtime import ResourceStrategySupervisor
from pvz_agent.continuity_frames import ContinuityFrames
from pvz_agent.resource_threat_v2 import ThreatCheckedContinuityFrames
from test_goal_review import frame,POLICY
from test_strategy_review import strategy
from test_continuity_frames import metadata


def resource_frame(sid='old',now=10.):
    state=frame(sid,now);state['capture_time']='test:'+sid
    evidence=dict(image='test.png',image_sha256='a'*64);state['evidence']=evidence
    cue=dict(cue_id=sid+':sun1',kind='visible_sun_appearance_candidate',state_id=sid,capture_time=state['capture_time'],
        evidence=copy.deepcopy(evidence),visible_reference_xy=[100,100],box_xyxy=[84,84,116,116],action_eligible=False)
    state['resources']=dict(sun_appearance=dict(schema='pvz-sun-appearance/1',status='observed',
        state_id=sid,capture_time=state['capture_time'],evidence=copy.deepcopy(evidence),candidates=[cue]))
    return state


class ResourcePredicateTests(unittest.TestCase):
    def test_current_cues_allow_counter_card_and_ordinary_motion_changes_only_for_attention(self):
        before=resource_frame();after=resource_frame('new',20.);after['fields']['sun']['value']=None
        after['fields']['cards'][0]['visual_state']='unknown';after['zombies']['candidates'][0]['head_xy'][0]=450
        evidence=resource_evidence(before,after,POLICY)
        self.assertEqual(evidence['current_cue_ids'],['new:sun1']);self.assertFalse(evidence['interactable_verified'])

    def test_counter_increase_alone_never_proves_a_pickup(self):
        before=resource_frame();after=resource_frame('new',20.);after['fields']['sun']['value']=125
        after['resources']['sun_appearance']['candidates']=[]
        with self.assertRaisesRegex(ReviewBlocked,'current_sun_cue_unconfirmed'):resource_evidence(before,after,POLICY)
        before['resources']['sun_appearance']['candidates']=[]
        with self.assertRaisesRegex(ReviewBlocked,'source_sun_cue_unconfirmed'):resource_evidence(before,resource_frame('new'),POLICY)

    def test_foreign_frame_cues_and_missing_evidence_fail_closed(self):
        for mutate in [lambda s:s['resources']['sun_appearance'].update(state_id='stale'),
            lambda s:s['resources']['sun_appearance']['candidates'][0]['evidence'].update(image_sha256='b'*64),
            lambda s:s['resources']['sun_appearance']['candidates'][0].update(visible_reference_xy=[float('nan'),100]),
            lambda s:s['resources'].clear(),lambda s:s['resources']['sun_appearance'].update(status='scene_unverified')]:
            after=resource_frame('new');mutate(after)
            with self.assertRaisesRegex(ReviewBlocked,'sun_cue_context_unknown'):resource_evidence(resource_frame(),after,POLICY)

    def test_new_left_or_rapid_nearer_warning_requires_replan(self):
        for x,reason in [(200,'resource_left_warning'),(350,'resource_new_nearer_warning')]:
            after=resource_frame('new');after['zombies']['candidates'][0]['head_xy'][0]=x
            with self.assertRaisesRegex(ReviewBlocked,reason):resource_evidence(resource_frame(),after,POLICY)

    def test_vehicle_unlocalized_and_nonplaying_still_block(self):
        for mutate in [lambda s:s['vehicles']['tracks'].append(dict(row=1)),
            lambda s:s['zombies']['uncertain_water_candidates'].append(dict(row=None)),lambda s:s['scene'].update(value='nonplaying')]:
            after=resource_frame('new');mutate(after)
            with self.assertRaises(ReviewBlocked):resource_evidence(resource_frame(),after,POLICY)


class ResourceReviewerTests(unittest.TestCase):
    def run_review(self,mutate=lambda s:None,*,pause=False,age=20.,goal='collect_resources'):
        clock=[10.];live=[resource_frame()];entered=threading.Event();release=threading.Event()
        def backend(source):
            entered.set();release.wait(2)
            return dict(strategy=strategy(source['state_id'],goal=goal,focus_row=None))
        worker=ResourceStrategicReviewer(backend,current_state=lambda:live[0],clock=lambda:clock[0],policy=POLICY)
        try:
            self.assertTrue(worker.submit());self.assertTrue(entered.wait(1))
            clock[0]=age;live[0]=resource_frame('new',age);mutate(live[0])
            if pause:worker.invalidate('observed_pause')
            release.set();worker._future.result(timeout=2);result=worker.review()
            if result and result['disposition']!='selected_observations_match':
                live[0]=resource_frame('later',age);self.assertIsNone(worker.review())
            return result,worker.events()
        finally:release.set();worker.close()

    def test_fresh_current_cue_is_nonactionable_and_counter_independent(self):
        result,_=self.run_review(lambda s:s['fields']['sun'].update(value=125))
        self.assertEqual(result['disposition'],'selected_observations_match');self.assertEqual(result['review_policy'],'resource_attention/v1')
        self.assertFalse(result['action_eligible']);self.assertTrue(result['requires_fresh_pickup_proposal'])
        self.assertNotIn('visible_reference_xy',result['evidence'])

    def test_cue_disappearance_rejects_and_cannot_revive(self):
        result,_=self.run_review(lambda s:s['resources']['sun_appearance'].update(candidates=[]))
        self.assertEqual(result['disposition'],'current_sun_cue_unconfirmed')

    def test_expiry_epoch_and_current_freshness_guards_unchanged(self):
        result,events=self.run_review(pause=True);self.assertIsNone(result);self.assertEqual(events[-1]['kind'],'invalidated_response')
        result,events=self.run_review(age=41.);self.assertIsNone(result);self.assertEqual(events[-1]['kind'],'historical_window_expired')
        result,_=self.run_review(lambda s:s['readonly_runtime'].update(expires_monotonic=19.))
        self.assertEqual(result['disposition'],'fresh_continuous_state_unavailable')

    def test_hold_and_develop_economy_keep_original_rules(self):
        for goal in ['hold','develop_economy']:
            result,_=self.run_review(lambda s:s['fields']['sun'].update(value=125),goal=goal)
            self.assertEqual(result['disposition'],'observed_context_changed')

    def test_supervisor_uses_same_registry_and_request_cap(self):
        clock=[10.];registry=ThreatCheckedContinuityFrames(clock=lambda:clock[0]);events=[]
        supervisor=ResourceStrategySupervisor(registry,lambda s:dict(strategy=strategy(s['state_id'],goal='collect_resources',focus_row=None)),
            policy=POLICY,record_sink=events.append,clock=lambda:clock[0],max_requests=1)
        try:
            registry.put(dict(metadata=metadata('old',10.)));ticket=registry.take()
            registry.publish_observation(ticket,resource_frame(),captured_at=10.,now=10.,max_age=2,input_age_ms=0)
            supervisor.poll();supervisor.reviewer._future.result(timeout=2)
            clock[0]=11.;registry.put(dict(metadata=metadata('new',11.)));ticket=registry.take();state=resource_frame('new',11.)
            state['fields']['sun']['value']=125
            registry.publish_observation(ticket,state,captured_at=11.,now=11.,max_age=2,input_age_ms=0)
            supervisor.poll();self.assertEqual(events[-1]['review']['review_policy'],'resource_attention/v1');self.assertEqual(supervisor.requests,1)
            registry.session_changed();self.assertIsNone(supervisor.reviewer.review())
        finally:supervisor.stop()


if __name__=='__main__':unittest.main()
