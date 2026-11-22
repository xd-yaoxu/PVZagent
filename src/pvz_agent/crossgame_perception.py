"""Experimental cross-game template bank with explicit corpse-template abstention.

This remains a partial screenshot detector. Rejection is visual resemblance, never
proof of death, and does not permit input or claim a clear lane.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import cv2
import numpy as np
from PIL import Image
from .perception import fingerprint
from .zombie_perception import ZombiePerception, build_model


def build_crossgame_model(root,dataset,config,output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Retain frozen model; choose a new revision')
    base=output.with_name(output.stem+'-base.npz')
    provenance=build_model(root,dataset,config,base)
    negatives=[];refs=[]
    w,h=config['patch_wh'];ax,ay=config['anchor_xy']
    for sample in dataset['samples']:
        if sample['split']!='development':
            continue
        path=root/sample['image']
        if fingerprint(path)!=sample['sha256']:
            raise ValueError('Negative-source image hash mismatch')
        with Image.open(path) as im:
            gray=cv2.cvtColor(np.asarray(im.convert('RGB')),cv2.COLOR_RGB2GRAY)
        for x,y in sample['labels'].get('negative_heads',[]):
            patch=gray[y-ay:y-ay+h,x-ax:x-ax+w]
            if patch.shape==(h,w):
                negatives.append(patch.copy());refs.append({'sample_id':sample['id'],'head_xy':[x,y]})
    if not negatives:
        raise ValueError('Explicit development-only negatives required')
    with np.load(base,allow_pickle=False) as bundle:
        arrays={k:bundle[k].copy() for k in bundle.files}
    provenance.update(negative_refs=refs,scope='Old-video and new A development templates; B excluded.',
        completed_at=datetime.now(timezone.utc).isoformat())
    by_id={sample['id']:sample for sample in dataset['samples']}
    poses=[]
    for ref in provenance['template_refs']:
        truth=next(z for z in by_id[ref['sample_id']]['labels']['zombies'] if z['head_xy']==ref['head_xy'])
        poses.append(truth.get('pose','ordinary'))
    arrays.update(negative_templates=np.stack(negatives),template_pose=np.array(poses),
                  provenance=np.array(json.dumps(provenance)))
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return provenance


class CrossGamePerception(ZombiePerception):
    def __init__(self,model_file,config):
        super().__init__(model_file,config)
        # A separate immutable-in-use instance supplies the existing scene guards.
        # Its head bank is empty; actual search below keeps pose-specific lane bands.
        self.scene_gate=ZombiePerception(model_file,config)
        self.scene_gate.model['templates']=np.empty((0,*self.model['templates'].shape[1:]),dtype=np.uint8)

    def observe(self,image):
        start=time.perf_counter()
        result=self.scene_gate.observe(image)
        gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
        w,h=self.config['patch_wh'];ax,ay=self.config['anchor_xy']
        proposals=[]
        if result['scene']['value']=='playing':
            for i,template in enumerate(self.model['templates']):
                pose=str(self.model['template_pose'][i])
                bands=self.config['submerged_head_bands_y'] if pose=='submerged' else self.config['head_bands_y']
                for row,band in enumerate(bands,1):
                    if band is None:
                        continue
                    low,high=band;top=max(0,low-ay);bottom=min(600,high-ay+h)
                    scores=cv2.matchTemplate(gray[top:bottom,20:800],template,cv2.TM_CCOEFF_NORMED)
                    maxima=(scores>=self.config['match_threshold']) & (scores>=cv2.dilate(scores,np.ones((9,9),np.uint8)))
                    ys,xs=np.where(maxima)
                    for sy,sx in zip(ys,xs):
                        x,y=int(sx+20+ax),int(sy+top+ay)
                        if low<=y<=high:
                            proposals.append({'row':row,'head_xy':[x,y],
                                'box_xyxy':[x-ax,y-ay,x-ax+w,y-ay+h],
                                'similarity':round(float(scores[sy,sx]),5),'template_index':i,'pose':pose})
        accepted=[]
        for candidate in sorted(proposals,key=lambda c:c['similarity'],reverse=True):
            if not any(np.linalg.norm(np.subtract(candidate['head_xy'],a['head_xy']))<self.config['nms_radius_pixels'] for a in accepted):
                accepted.append(candidate)
        accepted.sort(key=lambda c:(c['row'],c['head_xy'][0]))
        for i,c in enumerate(accepted,1):
            c.update(detection_id=f'z{i}',kind='zombie_candidate',type_identity=None,
                     alive=None,health=None,threat_level=None,
                     position_semantics='visible_head_reference_in_canonical_pixels_not_feet')
        settings=self.config['negative_veto'];retained=[];rejected=[]
        for candidate in accepted:
            x,y=candidate['head_xy'];left,top=x-ax,y-ay
            radius=settings['search_radius_pixels']
            patch=gray[max(0,top-radius):min(600,top+h+radius),max(0,left-radius):min(800,left+w+radius)]
            score=max(float(cv2.minMaxLoc(cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED))[1])
                      for t in self.model['negative_templates'])
            candidate['negative_similarity']=round(score,5)
            if score>=settings['threshold'] and score>=candidate['similarity']+settings['margin']:
                candidate['rejection_reason']='resembles_labeled_detached_or_falling_head'
                rejected.append(candidate)
            else:
                retained.append(candidate)
        result.update(candidates=retained,rejected_candidates=rejected,
            rejection_semantics='Visual-template abstention; no death or empty-lane confirmation.',
            processing_ms=round((time.perf_counter()-start)*1000,3))
        return result
