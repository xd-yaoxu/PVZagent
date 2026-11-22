"""Freeze a stricter tilted-only interior gate; preserve the v38 parent."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from day05 import load,write,fingerprint
from pvz_agent.bucket_water_interior import interior_tilted_contexts
BASE=ROOT/'evidence/day06/bucket-angle-v1'

if __name__=='__main__':
    cfg=load(ROOT/'config/zombie-perception-bucket-v39.json');parent=load(ROOT/'config/zombie-perception-bucket-v38.json')['model']
    cfg.update(version=40,model='data/day06/bucket-angle-v1/model-v40.npz',bucket_orientation_component_right_max=744)
    config=ROOT/'config/zombie-perception-bucket-v40.json';output=ROOT/cfg['model']
    assert not config.exists() and not output.exists()
    with np.load(ROOT/parent,allow_pickle=False) as bundle:arrays={k:bundle[k].copy() for k in bundle.files}
    provenance=json.loads(str(arrays['provenance']));provenance.update(config=cfg,
        bucket_angle_parent=parent,bucket_angle_parent_sha256=fingerprint(ROOT/parent),
        bucket_angle_built_at=datetime.now(timezone.utc).isoformat(),
        bucket_angle_limits='Known sources: whole tilted component must end left of x744 pool-interior margin, excluding rim/entry. Original vertical v38 remains unchanged.')
    arrays['provenance']=np.array(json.dumps(provenance));np.savez_compressed(output,**arrays)
    write(output.with_suffix('.json'),provenance);write(config,cfg)
    paths=[output,output.with_suffix('.json'),config,ROOT/'src/pvz_agent/bucket_water_interior.py',
        ROOT/'src/pvz_agent/bucket_water_oriented.py',Path(__file__)]
    write(BASE/'freeze-v40.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        artifacts={p.relative_to(ROOT).as_posix():fingerprint(p) for p in paths},
        old_arrays_unchanged=True,new_template_pixels=False,inherited_thresholds_changed=False,
        added_tilted_interior_gate=True,known_v39_raw_proposals_seen=True,input_sent=False))
    states=sorted((load(p) for p in (ROOT/'evidence/day06/goal-live-v1/live-01/states').glob('*.json')),key=lambda s:s['frame_time_seconds'])
    records=[]
    for i,state in enumerate(states):
        ref=state['evidence'];assert fingerprint(ROOT/ref['image'])==ref['image_sha256']
        with Image.open(ROOT/ref['image']) as image:proposals,checks=interior_tilted_contexts(np.asarray(image.convert('RGB')),cfg)
        records.append(dict(index=i,state_id=state['state_id'],image=ref['image'],proposals=proposals,checks=checks))
        if proposals:print(dict(index=i,proposals=[p['head_xy'] for p in proposals]),flush=True)
    write(BASE/'raw-probe-v40.json',dict(records=records,scene_gate_applied=False,full_detector_run=False,accuracy_scored=False,input_sent=False))
