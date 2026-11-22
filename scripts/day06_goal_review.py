"""Freeze and compare scoped review on recorded API responses and live archives."""
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import sys
import threading
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pvz_agent.goal_review import GoalStrategicReviewer
from pvz_agent.strategy_review import StrategicReviewer
from day05 import load,write,fingerprint
BASE=ROOT/'evidence/day06/goal-review-v1'
SOURCE=ROOT/'evidence/day06/continuity-v1/live-02-pinned'


def prepare():
    assert not BASE.exists();BASE.mkdir(parents=True)
    grid=load(ROOT/'config/calibration.json')['grid'];cell_width=grid['rect']['width']/grid['columns']
    assert cell_width==80 and grid['rect']['x']==40
    target=ROOT/'config/strategy-goal-review-v1.json';assert not target.exists()
    config=dict(schema='pvz-goal-review/1',policy=dict(left_warning_x=200,other_lane_margin_px=80),
        geometry_basis='Historical canonical grid x40, width720,9columns; left warning is x40+2*80, margin80. Read-only warning heuristic, not an action threshold or calibrated danger score.',
        supported_scoped_goal='defend_lane',other_goals='original full-field comparison',
        source_dispatch_gate='unchanged original full known observed_basis',
        live_config='config/llm-strategy-v1.json',max_result_age_seconds=2,historical_review_window_seconds=30,
        input_sent=False,action_eligible=False,experimental=True)
    write(target,config)
    upstream='evidence/day06/continuity-v1/manifest-SHA256.json'
    inputs=['src/pvz_agent/goal_review.py','src/pvz_agent/goal_runtime.py','tests/test_goal_review.py',
        'scripts/day06_goal_review.py','config/strategy-goal-review-v1.json']
    write(BASE/'freeze.json',dict(created_at=datetime.now(timezone.utc).isoformat(),artifacts={p:fingerprint(ROOT/p) for p in inputs},
        upstream_manifest=upstream,upstream_manifest_sha256=fingerprint(ROOT/upstream),
        design_before_replay_metrics=True,known_source_used_for_development=True,
        evaluation_plan='Two saved real API outputs, original first review and subsequent published states, source binding and recorded monotonic clock reconstructed. Compare original vs scoped attention. No new API/capture/CV, no independent accuracy.',
        input_sent=False))
    print(dict(prepared=True))


def check():
    for rel,digest in load(BASE/'freeze.json')['artifacts'].items():assert fingerprint(ROOT/rel)==digest,rel
    assert (ROOT/'evidence/day04/STOP').exists()


def tests():
    check();assert not (BASE/'tests.json').exists()
    with (BASE/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    record=dict(tests_run=result.testsRun,successful=result.wasSuccessful(),skipped=len(result.skipped),failures=len(result.failures),errors=len(result.errors))
    write(BASE/'tests.json',record);print(record);raise SystemExit(0 if result.wasSuccessful() else 1)


def evaluate(kind,source,result,first,request_time,reviews,policy):
    now=[request_time];current=[copy.deepcopy(source)];entered=threading.Event();release=threading.Event()
    def backend(value):
        assert value['state_id']==result['source_state_id'];entered.set();release.wait(2)
        return copy.deepcopy(result)
    cls=GoalStrategicReviewer if kind=='scoped' else StrategicReviewer
    worker=cls(backend,current_state=lambda:current[0],clock=lambda:now[0],**(dict(policy=policy) if kind=='scoped' else {}))
    rows=[]
    try:
        assert worker.submit() and entered.wait(1)
        now[0]=first['monotonic'];current[0]=load(SOURCE/'states'/(first['reviewed_state_id']+'.json'))
        release.set();worker._future.result(timeout=2)
        rows.append(dict(state_id=current[0]['state_id'],monotonic=now[0],review=worker.review()))
        for row in reviews:
            clock=row['monotonic_ns']/1e9
            if clock<=first['monotonic']:continue
            if row['publication']!='published':continue
            now[0]=clock;current[0]=load(SOURCE/'states'/(row['state_id']+'.json'))
            rows.append(dict(state_id=row['state_id'],monotonic=clock,review=worker.review()))
        return dict(rows=rows,events=worker.events())
    finally:release.set();worker.close()


def replay():
    check();assert not (BASE/'comparison.json').exists()
    raw=load(SOURCE/'summary.json');records=[]
    for path in raw['provider_results']:
        result=load(ROOT/path);sid=result['source_state_id'];source_path=SOURCE/'states'/(sid+'.json');source=load(source_path)
        assert result['success'] and result['strategy']['source_state_id']==sid
        request=next(r for r in raw['strategy']['records'] if r['kind']=='strategy_requested' and r['source_state_id']==sid)
        first=next(r for r in raw['strategy']['records'] if r['kind']=='strategy_review' and r['review']['source_state_id']==sid)
        inferences=[r for r in raw['records'] if r['kind']=='inferred']
        values={kind:evaluate(kind,source,result,first,request['monotonic'],inferences,
            load(ROOT/'config/strategy-goal-review-v1.json')['policy']) for kind in ['original','scoped']}
        records.append(dict(provider_result=path,provider_result_sha256=fingerprint(ROOT/path),
            source_archive=source_path.relative_to(ROOT).as_posix(),source_archive_sha256=fingerprint(source_path),
            original_first_review=first,**values))
    counts={kind:sum(r[kind]['rows'][0]['review']['disposition']=='selected_observations_match' for r in records) for kind in ['original','scoped']}
    write(BASE/'comparison.json',dict(created_at=datetime.now(timezone.utc).isoformat(),records=records,
        first_review_matches=counts,first_review_pairs=len(records),
        archived_source_only=True,simulated_historical_clock=True,live_registry_read=False,
        api_requests_this_phase=0,current_capture_performed=False,cv_inference_performed=False,
        input_sent=False,action_eligible=False,independent_validation=False,
        limits=['Known-source development replay; original real API outputs existed before this policy was designed.',
            'Original archive timestamps are evaluated with a simulated clock, not current live state.',
            'Each response lifetime is isolated; original overlapping request schedule is not a new live supervisor run.',
            'No negative-label accuracy or actual target/planting legality claim.']))
    print(dict(first_review_matches=counts,pairs=len(records)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','tests','replay']);args=parser.parse_args()
    globals()[args.command]()
