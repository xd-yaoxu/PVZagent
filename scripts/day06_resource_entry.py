"""Create opt-in resource review entry and freeze before tests/replay."""
from datetime import datetime,timezone
from pathlib import Path
from day05 import load,write,fingerprint
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'evidence/day06/resource-review-v1'

if __name__=='__main__':
    config=ROOT/'config/resource-attention-v1.json';entry=ROOT/'scripts/day06_resource_live.py'
    assert not config.exists() and not entry.exists()
    settings=load(ROOT/'config/strategy-goal-review-v1.json')
    settings.update(schema='pvz-resource-attention/1',resource_contract='Source and current bound visible sun appearance cues, no left warning or new nearer warning; attention only.',
        unrelated_resource_attention_changes=['stored_sun_counter','card_appearance'],sun_cue_config='config/sun-cues-v2.json',
        source_dispatch_gate='unchanged original full known observed_basis',other_goals='defend_lane retains frozen scoped rule; hold/develop_economy retain full selected-basis equality')
    write(config,settings)
    source=(ROOT/'scripts/day06_bucket_live.py').read_text(encoding='utf-8')
    source=source.replace('from pvz_agent.goal_runtime import run_goal_readonly','from pvz_agent.resource_runtime import run_resource_readonly as run_goal_readonly')
    source=source.replace('from pvz_agent.strategy_review import StrategyProvider,recommend_strategy','from pvz_agent.strategy_profile import ProfiledStrategyProvider as StrategyProvider,recommend_profiled as recommend_strategy')
    source=source.replace('from pvz_agent.bucket_readonly import create_bucket_stream','from pvz_agent.resource_stream import create_resource_stream as create_bucket_stream')
    source=source.replace("config/strategy-goal-review-v1.json","config/resource-attention-v1.json")
    source=source.replace("run_id='bucket-attention-'","run_id='resource-attention-'")
    source=source.replace("mode='bucket_v40_goal_scoped_live_readonly',object_version=40","mode='resource_attention_live_readonly',object_version=40,sun_cue_version=2")
    entry.write_text(source,encoding='utf-8')
    paths=[config,entry,ROOT/'config/sun-cues-v2.json',ROOT/'data/day06/resource-review-v1/sun-cue-v2.npz',
        ROOT/'src/pvz_agent/sun_cues.py',ROOT/'src/pvz_agent/resource_review.py',ROOT/'src/pvz_agent/resource_runtime.py',
        ROOT/'src/pvz_agent/resource_stream.py',ROOT/'src/pvz_agent/strategy_profile.py',Path(__file__)]
    write(BASE/'resource-freeze.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        artifacts={p.relative_to(ROOT).as_posix():fingerprint(p) for p in paths},
        input_age_seconds=1.5,result_age_seconds=2,historical_strategy_seconds=30,
        source_dispatch_changed=False,defense_rule_changed=False,input_sent=False))
    print(dict(entry=entry.relative_to(ROOT).as_posix(),live_run_performed=False))
