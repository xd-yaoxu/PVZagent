"""Opt-in v40 scoped-review live entry; no game input or automatic activation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pvz_agent.resource_runtime import run_resource_readonly as run_goal_readonly
from pvz_agent.pinned_capture import PinnedWindowsCapture
from pvz_agent.strategy_profile import ProfiledStrategyProvider as StrategyProvider,recommend_profiled as recommend_strategy
from pvz_agent.resource_stream import create_resource_stream as create_bucket_stream
from day05 import load,write,fingerprint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--seconds',type=int,default=60)
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists() or not 30<=args.seconds<=180:
        raise ValueError('New project output and30..180 seconds required')
    if not (ROOT/'evidence/day04/STOP').exists():raise ValueError('STOP must remain latched')
    for rel,digest in load(ROOT/'evidence/day06/goal-review-v1/freeze.json')['artifacts'].items():
        if fingerprint(ROOT/rel)!=digest:raise ValueError('Frozen goal review changed')
    settings=load(ROOT/'config/resource-attention-v1.json');cfg=load(ROOT/settings['live_config'])
    auth=load(ROOT/cfg['image_authorization_record'])
    if not cfg['image_transfer_authorized'] or auth['user_reply']!='允许' or auth['base_url']!=cfg['base_url']:
        raise ValueError('Existing game-client image authorization required')
    env=load(ROOT/'config/environment.json');provider=StrategyProvider(cfg)
    stream=create_bucket_stream(ROOT)
    out.mkdir(parents=True);calls=[];run_id='resource-attention-'+datetime.now(timezone.utc).isoformat()
    def backend(source):
        tick=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
        try:record=dict(success=True,**recommend_strategy(provider,ROOT,source))
        except Exception as exc:record=dict(success=False,error=getattr(exc,'code',type(exc).__name__))
        record.update(started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),duration_ms=(time.perf_counter()-tick)*1000,
            source_state_id=source['state_id'],source_run_id=source['run_id'],image=source['evidence']['image'],
            image_sha256=source['evidence']['image_sha256'],input_sent=False,action_eligible=False)
        path=out/f'provider-result-{len(calls):02d}.json';write(path,record);calls.append(path.relative_to(ROOT).as_posix())
        if not record['success']:raise RuntimeError('strategy_request_failed')
        return record
    def observe(image,meta,path,image_file):
        return stream.observe(image,run_id=run_id,state_id=meta['state_id'],capture_time=meta['capture_time'],
            frame_time_seconds=meta['monotonic_ns']/1e9,evidence=dict(image=image_file.relative_to(ROOT).as_posix(),
                image_sha256=meta['image_sha256'],metadata=path.relative_to(ROOT).as_posix(),pinned_pair=True))
    with (out/'journal.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
        lock=threading.Lock()
        def sink(record):
            with lock:journal.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
            if record.get('kind')=='captured' and record['index']%15==0:print(dict(captured_index=record['index']),flush=True)
        result=run_goal_readonly(backend=backend,policy=settings['policy'],strategy_record_sink=sink,
            max_requests=2,min_request_interval=15,
            capture_factory=lambda:PinnedWindowsCapture(env['game_exe'],env['scaler_exe'],env['scaler_config']),observe=observe,
            count=args.seconds,interval=1.,max_input_age=1.5,max_result_age=2.,max_duration=args.seconds+1,
            evidence_dir=out/'captures',archive_state=lambda s:write(out/'states'/(s['state_id']+'.json'),s),record_sink=sink)
    result.update(created_at=datetime.now(timezone.utc).isoformat(),mode='resource_attention_live_readonly',object_version=40,sun_cue_version=2,provider_results=calls,
        current_game_captured=True,cv_inference_performed=True,input_sent=False,action_eligible=False,stop_latched=True)
    write(out/'summary.json',result)
    print(dict(captured=sum(r['kind']=='captured' for r in result['records']),inferred=sum(r['kind']=='inferred' for r in result['records']),
        requests=result['strategy']['requests'],errors=result['errors'],workers_finished=result['strategy']['workers_finished']))


if __name__=='__main__':main()
