"""Conservative small-patch veto for newly aligned card answers."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .aligned_hud_perception import AlignedHudPerception
from .field_perception import card_vector
from .perception import fingerprint
from PIL import Image


def aligned_local_residual(engine,image,card,dx,dy):
    aligned=image.transform(image.size,Image.Transform.AFFINE,(1,0,dx,0,1,dy),resample=Image.Resampling.NEAREST)
    indexes,references=engine.card_references[card['slot']]
    sid=card['aligned_coverage_check']['reference_sample_id']
    j=next(j for j,index in enumerate(indexes) if engine.provenance['slot_card_refs'][int(index)]['sample_id']==sid)
    current=card_vector(aligned,card['slot']).reshape(30,18,3);reference=references[j]
    ratio=np.median(current.max(2)/np.maximum(reference.max(2),.1),axis=1)
    residual=np.abs(current-reference*np.clip(ratio,.15,1.3)[:,None,None]).mean(2)
    size=engine.config['aligned_residual_window_size']
    windows=cv2.boxFilter(residual,-1,(size,size),normalize=True,borderType=cv2.BORDER_REFLECT)
    y,x=np.unravel_index(windows.argmax(),windows.shape)
    return dict(maximum=float(windows[y,x]),window_size=size,patch_center_xy=[int(x),int(y)],
        threshold=engine.config['card_tile_residual_max'],reference_sample_id=sid,
        units='Mean absolute normalized RGB residual after scanline brightness fit, in resized card coordinates.')


def build_guarded_model(root,config,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Retain model')
    if fingerprint(root/config['aligned_guard_parent_model'])!=config['aligned_guard_parent_sha256']:
        raise ValueError('Aligned guard parent changed')
    with np.load(root/config['aligned_guard_parent_model'],allow_pickle=False) as parent:
        arrays={key:parent[key].copy() for key in parent.files}
    provenance=json.loads(str(arrays['provenance']))
    provenance.update(config=config,aligned_guard_parent_sha256=config['aligned_guard_parent_sha256'],
        aligned_guard_completed_at=datetime.now(timezone.utc).isoformat(),
        aligned_guard_limits='All arrays inherited unchanged from alignment22; only a stricter local residual veto. Known-source diagnosis, no new training.')
    arrays['provenance']=np.array(json.dumps(provenance))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance


class GuardedAlignedHudPerception(AlignedHudPerception):
    def observe(self,image,state_id,capture_time,*,card_costs=None,scene_hint=None):
        started=time.perf_counter()
        result=super().observe(image,state_id,capture_time,card_costs=card_costs,scene_hint=scene_hint)
        registration=result.get('card_alignment')
        if registration and registration['accepted'] and (registration['dx'] or registration['dy']):
            for card in result['cards']:
                if card['visual_state']=='unknown' or card['visual_evidence']!='header_registered_crop_with_joint_shape_price_and_brightness':continue
                evidence=aligned_local_residual(self,image,card,registration['dx'],registration['dy'])
                card['aligned_local_residual']=evidence
                if evidence['maximum']>evidence['threshold']:
                    insufficient=card['cost'] is not None and result['sun']['value'] is not None and result['sun']['value']<card['cost']
                    card.update(visual_state='unknown',reason='localized_aligned_patch_mismatch',similarity=0.,
                        usable_with_observed_resources=False if insufficient else None)
        result['processing_ms']=round((time.perf_counter()-started)*1000,3)
        return result
