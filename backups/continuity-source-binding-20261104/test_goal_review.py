"""Scoped attention can survive irrelevant changes, but not new risk or stale evidence."""
import copy
from pathlib import Path
import sys
import threading
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pvz_agent.goal_review import GoalStrategicReviewer,ReviewBlocked,defense_evidence,validate_policy
from pvz_agent.goal_runtime import GoalStrategySupervisor
from pvz_agent.continuity_frames import ContinuityFrames
from test_strategy_review import state,strategy
from test_continuity_frames import metadata

POLICY=dict(left_warning_x=200,other_lane_margin_px=80)


def frame(sid='old',now=10.):
    result=state(sid,now);result['vehicles']['uncertain_candidates']=[];return result


class DefensePredicateTests(unittest.TestCase):
    def test_unrelated_resources_unknown_cards_and_normal_motion_do_not_block_attention(self):
        before=frame();after=frame('new',20.)
        after['fields']['sun']['value']=None;after['fields']['cards'][0]['visual_state']='unknown'
        after['zombies']['candidates'][0]['head_xy'][0]=350
        evidence=defense_evidence(before,after,1,POLICY)
        self.assertEqual(evidence['current_visible_count'],1);self.assertFalse(evidence['identity_verified'])

    def test_vanished_target_is_unconfirmed_and_never_called_dead(self):
        after=frame();after['zombies']['candidates']=[]
        with self.assertRaisesRegex(ReviewBlocked,'focus_lane_unconfirmed'):defense_evidence(frame(),after,1,POLICY)

    def test_count_growth_or_shrinkage_requires_new_model_review(self):
        before=frame();after=frame();after['zombies']['candidates'].append(dict(row=1,head_xy=[700,50]))
        with self.assertRaisesRegex(ReviewBlocked,'focus_count_changed'):defense_evidence(before,after,1,POLICY)
        with self.assertRaisesRegex(ReviewBlocked,'focus_count_changed'):defense_evidence(after,before,1,POLICY)

    def test_focus_crossing_left_warning_boundary_requires_replan(self):
        after=frame();after['zombies']['candidates'][0]['head_xy'][0]=200
        with self.assertRaisesRegex(ReviewBlocked,'focus_warning_boundary_crossed'):defense_evidence(frame(),after,1,POLICY)

    def test_new_nearer_other_lane_requires_replan(self):
        after=frame();after['zombies']['candidates'].append(dict(row=2,head_xy=[400,100]))
        with self.assertRaisesRegex(ReviewBlocked,'new_other_lane_warning'):defense_evidence(frame(),after,1,POLICY)

    def test_new_farther_other_lane_does_not_claim_safety_or_optimality(self):
        after=frame();after['zombies']['candidates'].append(dict(row=2,head_xy=[750,100]))
        evidence=defense_evidence(frame(),after,1,POLICY);self.assertFalse(evidence['full_board_verified'])

    def test_vehicle_and_unlocalized_water_or_edge_evidence_block(self):
        for field in ['candidates','tracks','uncertain_candidates']:
            after=frame();after['vehicles'][field]=[dict(row=None)]
            with self.assertRaisesRegex(ReviewBlocked,'vehicle_context_unsupported'):defense_evidence(frame(),after,1,POLICY)
        for field in ['uncertain_edge_presence','uncertain_water_candidates']:
            after=frame();after['zombies'][field]=[dict(row=None)]
            with self.assertRaisesRegex(ReviewBlocked,'unlocalized_threat_context'):defense_evidence(frame(),after,1,POLICY)

    def test_unknown_row_nonfinite_or_boolean_positions_are_not_evidence(self):
        for updates in [dict(row=None),dict(row=True),dict(head_xy=[float('nan'),20]),dict(head_xy=[True,20])]:
            after=frame();after['zombies']['candidates'][0].update(updates)
            with self.assertRaisesRegex(ReviewBlocked,'head_position_unknown'):defense_evidence(frame(),after,1,POLICY)

    def test_missing_required_channels_and_nonplaying_fail_closed(self):
        after=frame();del after['vehicles']['uncertain_candidates']
        with self.assertRaisesRegex(ReviewBlocked,'vehicle_context_unknown'):defense_evidence(frame(),after,1,POLICY)
        after=frame();after['scene']['value']='nonplaying'
        with self.assertRaisesRegex(ReviewBlocked,'nonplaying_scene'):defense_evidence(frame(),after,1,POLICY)

    def test_policy_cannot_silently_expand_to_whole_board_or_bad_types(self):
        for policy in [None,dict(POLICY,execute=True),dict(left_warning_x=True,other_lane_margin_px=80),dict(left_warning_x=800,other_lane_margin_px=80)]:
            with self.assertRaises(ValueError):validate_policy(policy)


class GoalReviewerTests(unittest.TestCase):
    def run_review(self,*,goal='defend_lane',mutate=lambda s:None,now=20.,invalidate=False):
        clock=[10.];live=[frame()];entered=threading.Event();release=threading.Event()
        def backend(source):
            entered.set();release.wait(2)
            return dict(strategy=strategy(source['state_id'],goal=goal,focus_row=1 if goal=='defend_lane' else None))
        worker=GoalStrategicReviewer(backend,current_state=lambda:live[0],clock=lambda:clock[0],policy=POLICY)
        try:
            self.assertTrue(worker.submit());self.assertTrue(entered.wait(1))
            clock[0]=now;live[0]=frame('new',now);mutate(live[0])
            if invalidate:worker.invalidate('pause_observed')
            release.set();worker._future.result(timeout=2)
            result=worker.review()
            if result and result['disposition']=='selected_observations_match':
                result['strategy']['goal']='mutated'
                self.assertEqual(worker.review()['strategy']['goal'],goal)
                result['strategy']['goal']=goal
            return result,worker.events()
        finally:release.set();worker.close()

    def test_attention_matches_after_resources_change_but_never_becomes_actionable(self):
        result,_=self.run_review(mutate=lambda s:s['fields']['sun'].update(value=75))
        self.assertEqual(result['disposition'],'selected_observations_match');self.assertEqual(result['review_policy'],'defense_attention/v1')
        self.assertTrue(result['requires_current_target_and_legality_check']);self.assertFalse(result['action_eligible'])

    def test_nondefense_goals_keep_original_full_field_rejection(self):
        for goal in ['hold','collect_resources','develop_economy']:
            result,_=self.run_review(goal=goal,mutate=lambda s:s['fields']['sun'].update(value=75))
            self.assertEqual(result['disposition'],'observed_context_changed')

    def test_stale_snapshot_session_change_and_inflight_pause_remain_blocked(self):
        for mutate in [lambda s:s.update(run_id='restarted'),lambda s:s['readonly_runtime'].update(expires_monotonic=19.)]:
            result,_=self.run_review(mutate=mutate);self.assertEqual(result['disposition'],'fresh_continuous_state_unavailable')
        result,events=self.run_review(invalidate=True);self.assertIsNone(result);self.assertEqual(events[-1]['kind'],'invalidated_response')

    def test_historical_plan_window_is_not_extended(self):
        result,events=self.run_review(now=41.)
        self.assertIsNone(result);self.assertEqual(events[-1]['kind'],'historical_window_expired')

    def test_new_risk_clears_candidate_instead_of_reusing_older_goal(self):
        def mutate(s):s['zombies']['candidates'].append(dict(row=2,head_xy=[100,100]))
        result,_=self.run_review(mutate=mutate);self.assertEqual(result['disposition'],'new_other_lane_warning')

    def test_goal_supervisor_uses_continuity_hook_and_preserves_bounded_dispatch(self):
        clock=[10.];registry=ContinuityFrames(clock=lambda:clock[0]);events=[]
        supervisor=GoalStrategySupervisor(registry,lambda s:dict(strategy=strategy(s['state_id'])),
            policy=POLICY,record_sink=events.append,clock=lambda:clock[0],max_requests=1)
        try:
            registry.put(dict(metadata=metadata(0,10.)));ticket=registry.take()
            registry.publish_observation(ticket,frame(),captured_at=10.,now=10.,max_age=2,input_age_ms=0)
            supervisor.poll();supervisor.reviewer._future.result(timeout=2)
            clock[0]=11.;registry.put(dict(metadata=metadata(1,11.)));ticket=registry.take()
            next_state=frame('new',11.);next_state['fields']['sun']['value']=75
            registry.publish_observation(ticket,next_state,captured_at=11.,now=11.,max_age=2,input_age_ms=0)
            supervisor.poll();self.assertEqual(events[-1]['review']['review_policy'],'defense_attention/v1')
            registry.session_changed();self.assertIsNone(supervisor.reviewer.review());self.assertEqual(supervisor.requests,1)
        finally:supervisor.stop()


if __name__=='__main__':unittest.main()
