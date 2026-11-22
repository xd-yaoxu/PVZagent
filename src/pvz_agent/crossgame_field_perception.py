"""Cross-game HUD atlas with clean ink and consistent multi-reference cards."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .perception import classify,fingerprint
from .field_perception import FieldPerception,card_vector,glow_regions,overlap_area


def clean_glyphs(image,roi,threshold,ignore_top_rows=0,grouping='columns',speck_max_y=None):
    rgb=np.asarray(image.convert('RGB').crop(roi));channels=rgb.astype('int16')
    glow=(rgb[:,:,0]>215)&(rgb[:,:,1]>190)&(rgb[:,:,2]<115)
    if glow.mean()>.08:return [],'sun_glow_occlusion'
    ink=((channels.max(axis=2)<threshold)&(np.ptp(channels,axis=2)<50)).astype('uint8')
    ink[:ignore_top_rows]=0  # Counter tray shadow above the registered text band.
    # Discard only isolated tiny horizontal specks, never a broken final glyph.
    n,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=4)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h<=2 and area<=5 and (speck_max_y is None or y+h<=speck_max_y):ink[labels==i]=0
    groups=[]
    if grouping=='overlapping_component_intervals':
        # Adjacent columns can belong to separate glyphs without touching ink.
        # Broken parts of one glyph overlap in x and should stay together.
        _,_,parts,_=cv2.connectedComponentsWithStats(ink,connectivity=4)
        intervals=sorted((int(x),int(x+w)) for x,y,w,h,area in parts[1:] if area>0)
        for left,right in intervals:
            if groups and left<groups[-1][-1]+1:
                groups[-1]=list(range(groups[-1][0],max(groups[-1][-1]+1,right)))
            else:groups.append(list(range(left,right)))
    else:
        columns=np.flatnonzero(ink.any(axis=0))
        for x in columns:
            if not groups or x>groups[-1][-1]+1:groups.append([int(x)])
            else:groups[-1].append(int(x))
    boxes=[]
    for group in groups:
        x1,x2=group[0],group[-1]+1;ys=np.flatnonzero(ink[:,x1:x2].any(axis=1));y1,y2=int(ys[0]),int(ys[-1]+1)
        if x2-x1>16 or y2-y1<7 or ink[y1:y2,x1:x2].sum()<8:return [],'incomplete_or_merged_glyph'
        boxes.append((x1,y1,x2,y2))
    if not 1<=len(boxes)<=4:return [],'digit_component_count'
    mask=Image.fromarray(ink*255)
    return [np.asarray(mask.crop(b).resize((14,20),Image.Resampling.BILINEAR),dtype='float32').reshape(-1)/255 for b in boxes],None


def build_crossgame_fields(root,config,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Retain model')
    if fingerprint(root/config['parent_field_model'])!=config['parent_field_sha256']:raise ValueError('Field parent changed')
    with np.load(root/config['parent_field_model'],allow_pickle=False) as bundle:arrays={k:bundle[k].copy() for k in bundle.files}
    data=json.loads((root/config['crossgame_field_dataset']).read_text(encoding='utf-8'))
    dx=[];dy=[];dr=[];cx=[];cy=[];slots=[];cr=[];ids=[]
    permitted={'evidence/day02/gameplay-3-7.mp4','evidence/day06/new-games-20261023/game-a.mp4'}
    legacy_ids=set(json.loads(str(arrays['provenance']))['training_ids'])
    # Source restriction is enforced before writing any artifact.
    for sample in data['samples']:
        if sample['split']!='development':continue
        if sample['source'] not in permitted and not (sample['labels']['scene']!='playing' and sample['id'] in legacy_ids):
            raise ValueError('Only old and A development sources permitted')
        if fingerprint(root/sample['image'])!=sample['sha256']:raise ValueError('Development image changed')
        ids.append(sample['id'])
        if sample['labels']['scene']!='playing':continue
        with Image.open(root/sample['image']) as im:image=im.convert('RGB')
        if image.size!=(800,600):raise ValueError('Canonical screenshot required')
        sun=sample['labels']['sun']
        if sun is not None:
            for threshold in config['glyph_thresholds']:
                patches,reason=clean_glyphs(image,config['sun_roi_xyxy'],threshold,config.get('sun_ink_top_rows_ignore',0),config.get('glyph_grouping','columns'),config.get('speck_removal_max_y'))
                if reason is None and len(patches)==len(str(sun)):
                    for j,(patch,label) in enumerate(zip(patches,str(sun))):
                        dx.append(patch);dy.append(label);dr.append(dict(sample_id=sample['id'],sha256=sample['sha256'],threshold=threshold,digit_index=j))
        for slot,label in enumerate(sample['labels']['cards'],1):
            if label in ('bright','dim'):
                cx.append(card_vector(image,slot));cy.append(label);slots.append(slot)
                cr.append(dict(sample_id=sample['id'],sha256=sample['sha256'],slot=slot,label=label))
    if not dx or not cx:raise ValueError('Development templates required')
    provenance=json.loads(str(arrays['provenance']))
    provenance.update(config=config,training_ids=ids,digit_refs=dr,slot_card_refs=cr,
        crossgame_field_dataset_sha256=fingerprint(root/config['crossgame_field_dataset']),
        parent_field_sha256=config['parent_field_sha256'],crossgame_completed_at=datetime.now(timezone.utc).isoformat(),
        limits='Old and A development only; B excluded from all field templates. Known-source regression, fixed art layout.')
    arrays.update(field_digit_x=np.stack(dx),field_digit_y=np.array(dy),slot_card_x=np.stack(cx),slot_card_y=np.array(cy),
        slot_card_slots=np.array(slots),provenance=np.array(json.dumps(provenance)))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance


class CrossgameFieldPerception(FieldPerception):
    def __init__(self,model_file,config,root=None):
        super().__init__(model_file,config,root)
        self.card_references={}
        for slot in range(1,11):
            indexes=np.flatnonzero((self.model['slot_card_slots']==slot)&(self.model['slot_card_y']=='bright'))
            if not len(indexes):raise ValueError('Clear bright reference missing')
            brightness=np.array([self.model['slot_card_x'][j].reshape(30,18,3)[4:23].mean() for j in indexes])
            indexes=indexes[brightness>=brightness.max()*config['bright_reference_fraction_min']]
            self.card_references[slot]=(indexes,self.model['slot_card_x'][indexes].reshape(-1,30,18,3))

    def read_sun(self,image):
        reads=[]
        for threshold in self.config['glyph_thresholds']:
            glyphs,reason=clean_glyphs(image,self.config['sun_roi_xyxy'],threshold,self.config.get('sun_ink_top_rows_ignore',0),self.config.get('glyph_grouping','columns'),self.config.get('speck_removal_max_y'))
            matches=[classify(g,self.model['field_digit_x'],self.model['field_digit_y'],self.config['field_digit_distance_max'],self.config['field_digit_margin_min']) for g in glyphs]
            for match in matches:
                limit=self.config.get('digit_distance_overrides',{}).get(match['value'])
                if limit is not None and match['distance']>limit:
                    match.update(value=None,reason='digit_specific_shape_distance')
            text=''.join(m['value'] for m in matches) if matches and all(m['value'] is not None for m in matches) else None
            if text is not None and len(text)>1 and text.startswith('0'):text=None;reason='leading_zero_or_artifact'
            reads.append(dict(threshold=threshold,text=text,reason=reason,matches=matches))
        texts={r['text'] for r in reads if r['text'] is not None};accepted=len(texts)==1 and sum(r['text'] is not None for r in reads)>=2
        return dict(value=int(next(iter(texts))) if accepted else None,
            similarity=min((m['similarity'] for r in reads if r['text'] is not None for m in r['matches']),default=0.) if accepted else 0.,
            reason=None if accepted else 'glyph_consensus_missing_or_conflicting',threshold_reads=reads,
            method='old_and_A_digit_templates_tiny_specks_rejected',number_template_required=False)

    def observe(self,image,state_id,capture_time,*,card_costs=None,scene_hint=None):
        start=time.perf_counter();result=super().observe(image,state_id,capture_time,card_costs=card_costs,scene_hint=scene_hint)
        if result['scene']['status']!='playing':return result
        for card in result['cards']:
            slot=card['slot'];indexes,references=self.card_references[slot];current=card_vector(image,slot).reshape(30,18,3)
            ratios=np.median(current.max(axis=2)[None,:,:]/np.maximum(references.max(axis=3),.1),axis=2)
            residual=np.abs(current[None,:,:,:]-references*np.clip(ratios,.15,1.3)[:,:,None,None])
            distances=residual.mean(axis=(1,2,3));tile_max=residual.reshape(-1,6,5,3,6,3).mean(axis=(2,4,5)).max(axis=(1,2))
            minima=np.median(ratios.reshape(-1,6,5),axis=2).min(axis=1)
            ref=references[:,24:,:,:].mean(axis=3).reshape(len(references),-1)
            obs=current[24:].mean(axis=2).reshape(-1);ref=ref-ref.mean(axis=1,keepdims=True);obs=obs-obs.mean()
            strip=(ref@obs)/np.maximum(np.linalg.norm(ref,axis=1)*np.linalg.norm(obs),1e-8)
            valid=(distances<=self.config['card_shape_distance_max'])&(tile_max<=self.config['card_tile_residual_max'])&(strip>=self.config['cost_strip_similarity_min'])
            states=np.where(minima<self.config['dim_ratio_max'],'dim',np.where(minima>=self.config['bright_ratio_min'],'bright','unknown'))
            if 'strict_bright_scanline_min' in self.config:
                # A thin gray cover must not disappear inside a five-row median.
                thin_cover=ratios[:,:24].min(axis=1)<self.config['strict_bright_scanline_min']
                states[(states=='bright')&thin_cover]='unknown'
            plausible=set(states[valid]);visual=next(iter(plausible)) if len(plausible)==1 and 'unknown' not in plausible else 'unknown'
            reason=None if visual!='unknown' else 'reference_disagreement_or_no_joint_match'
            x=90+(slot-1)*51
            halo=self.config.get('card_glow_extra_margin',0)
            glow_boxes=[[max(0,b[0]-halo),max(0,b[1]-halo),min(800,b[2]+halo),min(600,b[3]+halo)] for b in result['glow_occlusion_regions']]
            if any(overlap_area([x+4,8,x+43,76],box)>=self.config['card_glow_overlap_min'] for box in glow_boxes):
                visual='unknown';reason='glow_occlusion'
            selected=int(np.argmin(distances+.05*(1-strip)))
            cost=card['cost'];sun=result['sun']['value']
            usable=False if visual=='dim' or cost is not None and sun is not None and sun<cost else (
                True if visual=='bright' and cost is not None and sun is not None and sun>=cost else None)
            for key in ('distance','brightness_ratio_min','cost_strip_similarity','tile_residual_max','reference_sample_id'):
                card.pop(key,None)
            card.update(visual_state=visual,reason=reason,usable_with_observed_resources=usable,
                similarity=round(max(0.,1-float(distances[selected])),4) if visual!='unknown' else 0.,
                reference_count=len(indexes),joint_reference_count=int(valid.sum()),reference_states=sorted(plausible),
                best_shape_distance=round(float(distances.min()),5),best_cost_strip_similarity=round(float(strip.max()),5),
                reference_sample_id=self.provenance['slot_card_refs'][int(indexes[selected])]['sample_id'],
                visual_evidence='consistent_joint_shape_cost_and_brightness_across_clear_development_references')
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
