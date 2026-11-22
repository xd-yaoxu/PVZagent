"""Experimental compositional HUD reading with explicit occlusion abstention."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .perception import Perception,feature,classify,fingerprint,cell_patch
from .zombie_perception import ZombiePerception


def glyphs_at_components(image,roi,threshold):
    rgb=np.asarray(image.convert('RGB').crop(roi));channels=rgb.astype('int16')
    glow=(rgb[:,:,0]>215)&(rgb[:,:,1]>190)&(rgb[:,:,2]<115)
    if glow.mean()>.08:return [],'sun_glow_occlusion'
    ink=((np.max(channels,axis=2)<threshold)&(np.ptp(channels,axis=2)<50)).astype('uint8')
    n,_,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=4)
    boxes=[]
    for x,y,w,h,area in stats[1:]:
        if h>=7 and area>=8:
            if w>16 or h>21:return [],'merged_or_large_glyph'
            boxes.append((int(x),int(y),int(x+w),int(y+h)))
    boxes.sort()
    if not 1<=len(boxes)<=4:return [],'digit_component_count'
    mask=Image.fromarray(ink*255)
    return [np.asarray(mask.crop(box).resize((14,20),Image.Resampling.BILINEAR),dtype='float32').reshape(-1)/255 for box in boxes],None


def glyphs_at(image,roi,threshold):
    rgb=np.asarray(image.convert('RGB').crop(roi));channels=rgb.astype('int16')
    glow=(rgb[:,:,0]>215)&(rgb[:,:,1]>190)&(rgb[:,:,2]<115)
    if glow.mean()>.08:return [],'sun_glow_occlusion'
    ink=(np.max(channels,axis=2)<threshold)&(np.ptp(channels,axis=2)<50)
    # Join broken strokes sharing columns; do not discard a small final digit.
    columns=np.flatnonzero(ink.any(axis=0));groups=[]
    for x in columns:
        if not groups or x>groups[-1][-1]+1:groups.append([int(x)])
        else:groups[-1].append(int(x))
    boxes=[]
    for group in groups:
        x1,x2=group[0],group[-1]+1;ys=np.flatnonzero(ink[:,x1:x2].any(axis=1))
        y1,y2=int(ys[0]),int(ys[-1]+1)
        if x2-x1>16 or y2-y1<7 or int(ink[y1:y2,x1:x2].sum())<8:
            return [],'incomplete_or_merged_glyph'
        boxes.append((x1,y1,x2,y2))
    if not 1<=len(boxes)<=4:return [],'digit_component_count'
    mask=Image.fromarray(ink.astype('uint8')*255)
    return [np.asarray(mask.crop(b).resize((14,20),Image.Resampling.BILINEAR),dtype='float32').reshape(-1)/255 for b in boxes],None


def card_image(image,slot):
    x=90+(slot-1)*51
    return image.crop((x+4,8,x+43,76))


def card_vector(image,slot):return feature(card_image(image,slot),(18,30))


def glow_regions(image):
    rgb=np.asarray(image.convert('RGB'))
    bright=((rgb[:,:,0]>=235)&(rgb[:,:,1]>=230)&(rgb[:,:,2]<=160)&
            (rgb[:,:,0].astype('int16')-rgb[:,:,1]<=20)).astype('uint8')
    # The fixed sun counter icon is not a floating occluder.
    bright[:55,:80]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(bright,connectivity=4)
    boxes=[]
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<70 or not (9<=w<=42 and 9<=h<=42 and .65<=w/h<=1.5):continue
        cx,cy=round(x+(w-1)/2),round(y+(h-1)/2)
        core=labels[max(0,cy-3):cy+4,max(0,cx-3):cx+4]
        if (core==i).sum()<40:continue
        boxes.append([max(0,int(x)-12),max(0,int(y)-12),min(800,int(x+w)+12),min(600,int(y+h)+12)])
    return boxes


def overlap_area(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def announcement_regions(image):
    rgb=np.asarray(image.convert('RGB'))
    red=((rgb[:,:,0]>180)&(rgb[:,:,1]<90)&(rgb[:,:,2]<90)).astype('uint8')
    red[:240]=0;red[440:]=0;red[:,:100]=0;red[:,750:]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(red)
    black=np.max(rgb,axis=2)<75;boxes=[]
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<35 or w<8 or h<12:continue
        left,top,right,bottom=max(0,x-4),max(0,y-4),min(800,x+w+4),min(600,y+h+4)
        support=(labels[top:bottom,left:right]==i).astype('uint8')
        ring=(cv2.dilate(support,np.ones((7,7),np.uint8))>0)&(support==0)
        fraction=float(black[top:bottom,left:right][ring].mean()) if ring.any() else 0.
        if fraction>=.3:boxes.append([max(0,int(left)-4),max(0,int(top)-4),min(800,int(right)+4),min(600,int(bottom)+4)])
    return boxes


def build_field_model(root,config,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Retain existing model')
    if fingerprint(root/config['parent_model'])!=config['parent_sha256']:raise ValueError('Parent changed')
    with np.load(root/config['parent_model'],allow_pickle=False) as bundle:arrays={k:bundle[k].copy() for k in bundle.files}
    dataset=json.loads((root/config['field_dataset']).read_text(encoding='utf-8'))
    digit_x=[];digit_y=[];digit_refs=[];card_x=[];card_y=[];card_slots=[];card_refs=[]
    for s in dataset['samples']:
        if s['split']!='development':continue
        if s['source']!='evidence/day02/gameplay-3-7.mp4' and s['labels']['scene']=='playing':
            raise ValueError('Only registered old development source permitted')
        if fingerprint(root/s['image'])!=s['sha256']:raise ValueError('Development frame changed')
        with Image.open(root/s['image']) as source:image=source.convert('RGB')
        if image.size!=(800,600):raise ValueError('Canonical image required')
        if s['labels']['scene']!='playing':continue
        sun=s['labels']['sun']
        if sun is not None:
            for threshold in config['glyph_thresholds']:
                glyphs,reason=glyphs_at(image,config['sun_roi_xyxy'],threshold)
                if reason is None and len(glyphs)==len(str(sun)):
                    for j,(glyph,label) in enumerate(zip(glyphs,str(sun))):
                        digit_x.append(glyph);digit_y.append(label);digit_refs.append(dict(sample_id=s['id'],sha256=s['sha256'],threshold=threshold,digit_index=j))
        for slot,label in enumerate(s['labels']['cards'],1):
            if label in ('bright','dim'):
                card_x.append(card_vector(image,slot));card_y.append(label);card_slots.append(slot)
                card_refs.append(dict(sample_id=s['id'],sha256=s['sha256'],slot=slot,label=label))
    if not digit_x or not card_x:raise ValueError('Development exemplars required')
    provenance=json.loads(str(arrays['provenance']))
    provenance.update(config=config,field_dataset_sha256=fingerprint(root/config['field_dataset']),
        digit_refs=digit_refs,slot_card_refs=card_refs,parent_sha256=config['parent_sha256'],
        field_completed_at=datetime.now(timezone.utc).isoformat(),
        limits='Old development only; known-source diagnostic regression, fixed ten-slot art layout. No readiness or input guarantee.')
    arrays.update(field_digit_x=np.stack(digit_x),field_digit_y=np.array(digit_y),
        slot_card_x=np.stack(card_x),slot_card_y=np.array(card_y),slot_card_slots=np.array(card_slots),
        provenance=np.array(json.dumps(provenance)))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance


class FieldPerception(Perception):
    def __init__(self,model_file,config,root=None):
        super().__init__(model_file,config);root=Path(root) if root else Path(__file__).resolve().parents[2]
        if fingerprint(root/config['guard_model'])!=config['guard_sha256']:raise ValueError('Scene guard changed')
        guard_config=json.loads((root/config['guard_config']).read_text(encoding='utf-8'))
        self.guard=ZombiePerception(root/config['guard_model'],guard_config)
        self.guard.model['templates']=np.empty((0,*self.guard.model['templates'].shape[1:]),dtype='uint8')

    def read_sun(self,image):
        reads=[]
        for threshold in self.config['glyph_thresholds']:
            glyphs,reason=glyphs_at(image,self.config['sun_roi_xyxy'],threshold)
            matches=[classify(g,self.model['field_digit_x'],self.model['field_digit_y'],
                self.config['field_digit_distance_max'],self.config['field_digit_margin_min']) for g in glyphs]
            text=''.join(m['value'] for m in matches) if matches and all(m['value'] is not None for m in matches) else None
            if text is not None and len(text)>1 and text.startswith('0'):text=None;reason='leading_zero_or_artifact'
            reads.append(dict(threshold=threshold,text=text,reason=reason,matches=matches))
        texts={r['text'] for r in reads if r['text'] is not None}
        accepted=len(texts)==1 and sum(r['text'] is not None for r in reads)>=2
        return dict(value=int(next(iter(texts))) if accepted else None,
            similarity=min((m['similarity'] for r in reads if r['text'] is not None for m in r['matches']),default=0.) if accepted else 0.,
            reason=None if accepted else 'glyph_consensus_missing_or_conflicting',threshold_reads=reads,
            method='development_digit_templates_with_threshold_consensus',number_template_required=False)

    def observe(self,image,state_id,capture_time,*,card_costs=None,scene_hint=None):
        start=time.perf_counter()
        if image.size!=(800,600):raise ValueError('Canonical image required')
        scene_guard=self.guard.observe(image)['scene']
        hint=scene_hint if scene_guard['value']=='playing' else 'nonplaying'
        result=super().observe(image,state_id,capture_time,card_costs=card_costs,scene_hint=hint)
        result['schema']='pvz-observation/2';result['scene_guard']=scene_guard
        result['scope']='Fixed reviewed ten-slot art layout; visual brightness is not cooldown or planting legality.'
        if result['scene']['status']!='playing':
            if scene_guard['value']!='playing':result['scene'].update(value=scene_guard['value'],reason=scene_guard.get('reason'))
            result['processing_ms']=round((time.perf_counter()-start)*1000,3);return result
        result['sun']=self.read_sun(image);occluders=glow_regions(image);result['glow_occlusion_regions']=occluders
        announcements=announcement_regions(image);result['announcement_occlusion_regions']=announcements
        for card in result['cards']:
            slot=card['slot'];x=90+(slot-1)*51;box=[x+4,12,x+43,76]
            selector=self.model['slot_card_slots']==slot
            vectors=self.model['slot_card_x'][selector];current=card_vector(image,slot).reshape(30,18,3)
            clear=np.argmax([v.reshape(30,18,3)[4:23].mean() for v in vectors])
            reference_index=int(np.flatnonzero(selector)[clear])
            vectors=vectors[clear:clear+1]
            # Fit a brightness multiplier per scanline against bright art only.
            # Partial gray covers remain dim even when most of the card is bright.
            fits=[]
            for vector in vectors:
                reference=vector.reshape(30,18,3)
                ratio=np.median(current.max(axis=2)/np.maximum(reference.max(axis=2),.1),axis=1)
                residual=np.abs(current-reference*np.clip(ratio,.15,1.3)[:,None,None])
                tile=residual.reshape(6,5,3,6,3).mean(axis=(1,3,4))
                fits.append((float(residual.mean()),float(tile.max()),ratio))
            if not fits:
                match=dict(value=None,reason='no_bright_art_reference',similarity=0.);tile_max=1.;minimum=None
            else:
                distance,tile_max,ratio=min(fits,key=lambda f:f[0]);minimum=float(np.min(np.median(ratio.reshape(6,5),axis=1)))
                visual='dim' if minimum<self.config['dim_ratio_max'] else 'bright' if minimum>=self.config['bright_ratio_min'] else None
                accepted=distance<=self.config['card_shape_distance_max']
                match=dict(value=visual if accepted else None,reason=None if accepted and visual else 'art_mismatch_or_partial_transition',
                    similarity=round(max(0.,1-distance),4),distance=round(distance,5),brightness_ratio_min=round(minimum,5))
                ref=vectors[0].reshape(30,18,3)[24:].mean(axis=2).astype('float32')
                observed=current[24:].mean(axis=2).astype('float32')
                strip_score=float(cv2.matchTemplate(observed,ref,cv2.TM_CCOEFF_NORMED)[0,0])
                match['cost_strip_similarity']=round(strip_score,5)
                if strip_score<self.config['cost_strip_similarity_min']:
                    match.update(value=None,reason='cost_strip_occlusion_or_mismatch')
            occluded=any(overlap_area(box,b)>=self.config['card_glow_overlap_min'] for b in occluders)
            if occluded or tile_max>self.config['card_tile_residual_max']:
                match.update(value=None,reason='glow_occlusion' if occluded else 'localized_card_patch_mismatch')
            visual=match.pop('value');cost=card['cost'];sun=result['sun']['value']
            card.pop('margin',None)  # Legacy pooled-atlas margins do not describe this reader.
            usable=False if visual=='dim' or cost is not None and sun is not None and sun<cost else (
                True if visual=='bright' and cost is not None and sun is not None and sun>=cost else None)
            card.update(match,visual_state=visual or 'unknown',usable_with_observed_resources=usable,
                tile_residual_max=round(tile_max,5),visual_evidence='scanline_brightness_relative_to_registered_bright_art',
                reference_sample_id=self.provenance['slot_card_refs'][reference_index]['sample_id'],
                layout_scope='Registered fixed ten-slot layout; novel art or reordering unvalidated.')
        for cell in result['cells']:
            x=40+(cell['column']-.5)*80;y=80+(cell['row']-.5)*85
            box=[round(x-31),round(y-43),round(x+31),round(y+26)]
            if any(overlap_area(box,b)>=self.config['cell_glow_overlap_min'] for b in occluders):
                cell.update(occupancy='unknown',occupied=None,carrier_present=None,upper_present=None,
                    reason='glow_occlusion',similarity=0.)
            whole_cell=[40+(cell['column']-1)*80,80+(cell['row']-1)*85,40+cell['column']*80,80+cell['row']*85]
            if any(overlap_area(whole_cell,b)>=30 for b in announcements):
                cell.update(occupancy='unknown',occupied=None,carrier_present=None,upper_present=None,
                    reason='announcement_text_occlusion',similarity=0.)
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
