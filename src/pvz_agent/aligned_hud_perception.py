"""Register card headers locally; supplement unknown card states only."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .hud_consensus_perception import HudConsensusPerception,card_candidate
from .field_perception import overlap_area
from .perception import fingerprint


def header_registration(image,templates,config):
    gray=np.asarray(image.convert('RGB'),dtype='float32').mean(2)
    rx,ry=config['header_search_radius_xy'];bank_maps=[];slot_maps=[]
    for bank in templates:
        maps=[]
        for slot,template in enumerate(bank,1):
            x=90+(slot-1)*51
            search=gray[5-ry:17+ry,x+2-rx:x+48+rx]
            maps.append(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))
        slot_maps.append(np.stack(maps));bank_maps.append(np.median(maps,axis=0))
    bank_maps=np.stack(bank_maps);combined=bank_maps.max(0)
    y,x=np.unravel_index(combined.argmax(),combined.shape);bank=int(bank_maps[:,y,x].argmax())
    other=combined.copy();other[y,x]=-1
    best=float(combined[y,x]);runner=float(other.max());scores=slot_maps[bank][:,y,x]
    support=int((scores>=config['header_slot_similarity_min']).sum())
    dx,dy=int(x)-rx,int(y)-ry
    boundary=abs(dx)==rx or abs(dy)==ry
    accepted=(best>=config['header_similarity_min'] and best-runner>=config['header_peak_margin_min']
              and support>=config['header_support_slots_min'] and not boundary)
    return dict(accepted=bool(accepted),dx=dx,dy=dy,reference_bank=bank,
        similarity=round(best,6),runner_up_similarity=round(runner,6),peak_margin=round(best-runner,6),
        supporting_slots=support,slot_similarities=[round(float(s),6) for s in scores],
        reason=None if accepted else 'header_translation_uncertain_or_at_search_boundary',
        scope='Canonical screenshot card crops only; not a desktop/input coordinate calibration.')


def build_aligned_model(root,config,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Retain model')
    if fingerprint(root/config['alignment_parent_model'])!=config['alignment_parent_sha256']:
        raise ValueError('Alignment parent changed')
    with np.load(root/config['alignment_parent_model'],allow_pickle=False) as parent:
        arrays={key:parent[key].copy() for key in parent.files}
    samples={s['id']:s for s in json.loads((root/config['header_dataset']).read_text(encoding='utf-8'))['samples']}
    banks=[];refs=[]
    for sid in config['header_reference_ids']:
        sample=samples[sid]
        if sample['split']!='development' or sample['source'] not in ('evidence/day02/gameplay-3-7.mp4','evidence/day06/new-games-20261023/game-a.mp4'):
            raise ValueError('Only old and A development header sources permitted')
        if sample['labels']['scene']!='playing':raise ValueError('Playing header reference required')
        if fingerprint(root/sample['image'])!=sample['sha256']:raise ValueError('Header source changed')
        with Image.open(root/sample['image']) as image:gray=np.asarray(image.convert('RGB'),dtype='float32').mean(2)
        if gray.shape!=(600,800):raise ValueError('Canonical screenshot required')
        bank=[]
        for slot in range(1,11):
            x=90+(slot-1)*51;bank.append(gray[5:17,x+2:x+48].copy())
        banks.append(np.stack(bank));refs.append(dict(sample_id=sid,image=sample['image'],sha256=sample['sha256'],source=sample['source'],split='development'))
    if not banks:raise ValueError('Header banks required')
    provenance=json.loads(str(arrays['provenance']))
    provenance.update(config=config,alignment_parent_sha256=config['alignment_parent_sha256'],header_refs=refs,
        header_dataset_sha256=fingerprint(root/config['header_dataset']),alignment_completed_at=datetime.now(timezone.utc).isoformat(),
        alignment_limits='Only old/A development headers. All prior numeric arrays retained. Known-source regression, B never templates. Local card crops only.')
    arrays.update(header_templates=np.stack(banks),provenance=np.array(json.dumps(provenance)))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance


class AlignedHudPerception(HudConsensusPerception):
    def observe(self,image,state_id,capture_time,*,card_costs=None,scene_hint=None):
        start=time.perf_counter()
        result=super().observe(image,state_id,capture_time,card_costs=card_costs,scene_hint=scene_hint)
        if result['scene']['status']!='playing':return result
        registration=header_registration(image,self.model['header_templates'],self.config)
        registration['reference_sample_id']=self.provenance['header_refs'][registration['reference_bank']]['sample_id']
        result['card_alignment']=registration
        if registration['accepted'] and (registration['dx'] or registration['dy']):
            dx,dy=registration['dx'],registration['dy']
            aligned=image.transform(image.size,Image.Transform.AFFINE,(1,0,dx,0,1,dy),resample=Image.Resampling.NEAREST)
            halo=self.config['card_glow_extra_margin']
            boxes=[[max(0,b[0]-halo),max(0,b[1]-halo),min(800,b[2]+halo),min(600,b[3]+halo)] for b in result['glow_occlusion_regions']]
            for card in result['cards']:
                if card['visual_state']!='unknown' or card['reason']=='glow_occlusion':continue
                slot=card['slot'];x=90+(slot-1)*51;roi=[x+4+dx,8+dy,x+43+dx,76+dy]
                if any(overlap_area(roi,box)>=self.config['card_glow_overlap_min'] for box in boxes):
                    card['aligned_coverage_check']=dict(value='unknown',reason='aligned_crop_glow_occlusion',source_roi_xyxy=roi)
                    continue
                candidate=card_candidate(self,aligned,slot);candidate['source_roi_xyxy']=roi
                card['aligned_coverage_check']=candidate
                if candidate['value']=='unknown':continue
                visual=candidate['value'];sun=result['sun']['value'];cost=card['cost']
                usable=False if visual=='dim' or cost is not None and sun is not None and sun<cost else (
                    True if visual=='bright' and cost is not None and sun is not None and sun>=cost else None)
                card['unshifted_reference_check']={key:card.pop(key) for key in ('reference_states','joint_reference_count','best_shape_distance','best_cost_strip_similarity','coverage_check') if key in card}
                card.update(visual_state=visual,reason=None,usable_with_observed_resources=usable,
                    similarity=round(max(0.,1-candidate['shape_distance']),4),reference_states=candidate['states'],
                    joint_reference_count=candidate['joint_reference_count'],reference_sample_id=candidate['reference_sample_id'],
                    visual_evidence='header_registered_crop_with_joint_shape_price_and_brightness')
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
