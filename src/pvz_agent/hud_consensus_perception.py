"""Spatial digit evidence and photometric card matching for fixed HUD layouts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .counter_field_perception import CounterFieldPerception
from .field_perception import card_vector
from .perception import classify,fingerprint


def glyph_evidence(image,config,threshold,color):
    rgb=np.asarray(image.convert('RGB').crop(config['sun_roi_xyxy'])).astype('int16')
    if ((rgb[:,:,0]>215)&(rgb[:,:,1]>190)&(rgb[:,:,2]<115)).mean()>.08:
        return []
    ink=(rgb.max(2)<threshold)&(np.ptp(rgb,axis=2)<50)
    if color:
        ink&=(rgb.max(2)<config['counter_dark_core_max'])|(rgb[:,:,0]-rgb[:,:,1]<=config['counter_red_green_max'])
    ink=ink.astype('uint8');ink[:config['sun_ink_top_rows_ignore']]=0
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=4)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h<=2 and area<=5:ink[labels==i]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=4)
    groups=[]
    for left,right in sorted((int(x),int(x+w)) for x,y,w,h,area in stats[1:] if area>0):
        if groups and left<groups[-1][1]:groups[-1][1]=max(right,groups[-1][1])
        else:groups.append([left,right])
    if not 1<=len(groups)<=4:return []
    boxes=[]
    for x1,x2 in groups:
        ys=np.flatnonzero(ink[:,x1:x2].any(1));y1,y2=int(ys[0]),int(ys[-1]+1)
        if x2-x1>16 or y2-y1<7 or ink[y1:y2,x1:x2].sum()<8:return []
        boxes.append([x1,y1,x2,y2])
    mask=Image.fromarray(ink*255)
    return [(box,np.asarray(mask.crop(box).resize((14,20),Image.Resampling.BILINEAR),
              dtype='float32').reshape(-1)/255) for box in boxes]


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def spatial_digit_consensus(readings,existing_texts,iou_min):
    """Match spatial glyphs, never assume the ordinal index means the same digit."""
    evidence=[g for reading in readings for g in reading['glyphs'] if g['match']['value'] is not None]
    hypotheses=[]
    for reading in readings:
        if not reading['glyphs']:continue
        supported=[]
        for anchor in reading['glyphs']:
            votes=[g for g in evidence if box_iou(anchor['box'],g['box'])>=iou_min]
            labels={v['match']['value'] for v in votes};thresholds={v['threshold'] for v in votes}
            digit=next(iter(labels)) if len(labels)==1 and len(thresholds)>=2 else None
            supported.append(dict(box=anchor['box'],digit=digit,votes=votes))
        text=''.join(s['digit'] for s in supported) if all(s['digit'] is not None for s in supported) else None
        if text and len(text)>1 and text.startswith('0'):text=None
        hypotheses.append(dict(mask=reading['mask'],threshold=reading['threshold'],text=text,support=supported))
    texts={h['text'] for h in hypotheses if h['text'] is not None}
    all_texts=texts|set(existing_texts)
    accepted=len(texts)==1 and len(all_texts)==1
    return dict(value=int(next(iter(texts))) if accepted else None,
        reason=None if accepted else 'spatial_digit_evidence_incomplete_or_conflicting',hypotheses=hypotheses)


def card_candidate(engine,image,slot):
    c=engine.config;indexes,refs=engine.card_references[slot]
    cur=card_vector(image,slot).reshape(30,18,3)
    ratios=np.median(cur.max(2)[None,:,:]/np.maximum(refs.max(3),.1),axis=2)
    adjusted=refs*np.clip(ratios,.15,1.3)[:,:,None,None]
    residual=np.abs(cur[None,:,:,:]-adjusted)
    distance=residual.mean((1,2,3));tile=residual.reshape(-1,6,5,3,6,3).mean((2,4,5)).max((1,2))
    minima=np.median(ratios.reshape(-1,6,5),axis=2).min(1)
    ref=adjusted[:,24:,:,:].mean(3).reshape(len(refs),-1);obs=cur[24:].mean(2).reshape(-1)
    ref-=ref.mean(1,keepdims=True);obs-=obs.mean()
    strip=(ref@obs)/np.maximum(np.linalg.norm(ref,axis=1)*np.linalg.norm(obs),1e-8)
    valid=(distance<=c['card_shape_distance_max'])&(tile<=c['card_tile_residual_max'])&(strip>=c['cost_strip_similarity_min'])
    close=valid&(distance<=distance[valid].min()+c['card_near_reference_distance_slack']) if valid.any() else valid
    states=np.where(minima<c['dim_ratio_max'],'dim',np.where(minima>=c['bright_ratio_min'],'bright','unknown'))
    plausible=set(states[close]);value=str(next(iter(plausible))) if len(plausible)==1 and 'unknown' not in plausible else 'unknown'
    best=int(np.argmin(np.where(close,distance,np.inf))) if close.any() else None
    return dict(value=value,states=sorted(plausible),nearby_reference_count=int(close.sum()),joint_reference_count=int(valid.sum()),
        reference_sample_id=engine.provenance['slot_card_refs'][int(indexes[best])]['sample_id'] if best is not None else None,
        shape_distance=float(distance[best]) if best is not None else None,
        adjusted_cost_strip_similarity=float(strip[best]) if best is not None else None,
        tile_residual_max=float(tile[best]) if best is not None else None)


def build_hud_model(root,config,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Retain model')
    if fingerprint(root/config['coverage_parent_model'])!=config['coverage_parent_sha256']:
        raise ValueError('Coverage parent changed')
    with np.load(root/config['coverage_parent_model'],allow_pickle=False) as parent:
        arrays={key:parent[key].copy() for key in parent.files}
    provenance=json.loads(str(arrays['provenance']))
    provenance.update(config=config,coverage_parent_sha256=config['coverage_parent_sha256'],
        coverage_completed_at=datetime.now(timezone.utc).isoformat(),
        coverage_limits='All numeric arrays inherited unchanged from fields20. Same known-source templates and fixed layout, no new training.')
    arrays['provenance']=np.array(json.dumps(provenance))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance


class HudConsensusPerception(CounterFieldPerception):
    def read_sun(self,image):
        original=super().read_sun(image)
        if original['value'] is not None or original['reason']=='counter_readers_disagree':return original
        readings=[]
        for color in (False,True):
            prefix='color_digit' if color else 'field_digit'
            for threshold in self.config['glyph_thresholds']:
                glyphs=[]
                for box,patch in glyph_evidence(image,self.config,threshold,color):
                    match=classify(patch,self.model[prefix+'_x'],self.model[prefix+'_y'],
                        self.config['field_digit_distance_max'],self.config['field_digit_margin_min'])
                    limit=self.config.get('digit_distance_overrides',{}).get(match['value'])
                    if limit is not None and match['distance']>limit:match.update(value=None,reason='digit_specific_shape_distance')
                    glyphs.append(dict(box=box,match=match,threshold=threshold,mask='color' if color else 'neutral'))
                readings.append(dict(mask='color' if color else 'neutral',threshold=threshold,glyphs=glyphs))
        existing={r['text'] for reader in original['readers'].values() for r in reader['threshold_reads'] if r['text'] is not None}
        consensus=spatial_digit_consensus(readings,existing,self.config['glyph_match_iou_min'])
        original['spatial_consensus']=consensus
        if consensus['value'] is not None:
            scores=[v['match']['similarity'] for h in consensus['hypotheses'] if h['text'] is not None for s in h['support'] for v in s['votes']]
            original.update(value=consensus['value'],reason=None,similarity=min(scores),
                method='spatially_aligned_digits_supported_at_two_distinct_thresholds')
        return original

    def observe(self,image,state_id,capture_time,*,card_costs=None,scene_hint=None):
        start=time.perf_counter()
        result=super().observe(image,state_id,capture_time,card_costs=card_costs,scene_hint=scene_hint)
        if result['scene']['status']!='playing':return result
        for card in result['cards']:
            if card['visual_state']!='unknown' or card['reason']=='glow_occlusion':continue
            candidate=card_candidate(self,image,card['slot']);card['coverage_check']=candidate
            if candidate['value']=='unknown':continue
            visual=candidate['value'];sun=result['sun']['value'];cost=card['cost']
            usable=False if visual=='dim' or cost is not None and sun is not None and sun<cost else (
                True if visual=='bright' and cost is not None and sun is not None and sun>=cost else None)
            card['prior_reference_check']={key:card.pop(key) for key in ('reference_states','joint_reference_count','best_shape_distance','best_cost_strip_similarity')}
            card.update(visual_state=visual,reason=None,usable_with_observed_resources=usable,
                similarity=round(max(0.,1-candidate['shape_distance']),4),
                reference_states=candidate['states'],joint_reference_count=candidate['joint_reference_count'],
                reference_sample_id=candidate['reference_sample_id'],
                visual_evidence='joint_shape_tile_and_scanline_adjusted_price_in_near_reference_neighborhood')
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
