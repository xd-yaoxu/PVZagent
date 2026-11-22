"""OpenCV screenshot candidates and scene vetoes. Never authorizes game input."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .perception import classify, feature, fingerprint


def scene_feature(image,config):
    if 'scene_feature_xyxy' in config:
        return feature(image.crop(config['scene_feature_xyxy']),tuple(config['scene_feature_wh']))
    return feature(image,(40,30))


def build_model(root, dataset, config, output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Model exists; retain frozen artifacts and select a new version')
    templates=[]; refs=[]; skipped=[]; scene_x=[]; scene_y=[]; training_ids=[]; reward=None
    w,h=config['patch_wh'];ax,ay=config['anchor_xy'];vetoes={}
    from PIL import Image
    for s in dataset['samples']:
        if s['split']!='development':
            continue
        path=root/s['image']
        if fingerprint(path)!=s['sha256']:
            raise ValueError('Development image hash mismatch')
        with Image.open(path) as f:
            im=f.convert('RGB')
        training_ids.append(s['id'])
        if s['id'] not in config.get('scene_exclude_ids',[]):
            scene_x.append(scene_feature(im,config));scene_y.append(s['labels']['scene'])
        for i,v in enumerate(config.get('scene_veto_templates',[])):
            if s['id']==v['sample_id']:
                vetoes[f'veto_{i}']=cv2.cvtColor(np.asarray(im.crop(v['crop_xyxy'])),cv2.COLOR_RGB2GRAY)
        if s['timestamp_seconds']==470:
            reward=cv2.cvtColor(np.asarray(im.crop(config['reward_template_xyxy'])),cv2.COLOR_RGB2GRAY)
        gray=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY)
        for z in s['labels']['zombies']:
            if not z['template']:
                continue
            x,y=z['head_xy'];left,top=x-ax,y-ay
            patch=gray[top:top+h,left:left+w]
            if patch.shape!=(h,w):
                skipped.append({'sample_id':s['id'],'head_xy':[x,y],
                                'reason':'template_clipped_by_image_edge'})
                continue
            templates.append(patch.copy());refs.append({'sample_id':s['id'],'head_xy':[x,y],'row':z['row']})
    if reward is None or not templates:
        raise ValueError('Missing development templates')
    if len(vetoes)!=len(config.get('scene_veto_templates',[])):
        raise ValueError('Every scene veto must reference a development image')
    provenance={'training_ids':training_ids,'source_split':'development','config':config,
                'template_refs':refs,'skipped_templates':skipped,
                'created_at':datetime.now(timezone.utc).isoformat(),
                'opencv':cv2.__version__,'scope':'Same-video local prototype; uncalibrated similarities.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,templates=np.stack(templates),scene_x=np.stack(scene_x),
                        scene_y=np.array(scene_y),reward=reward,
                        **vetoes,
                        provenance=np.array(json.dumps(provenance)))
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return provenance


class ZombiePerception:
    def __init__(self,model_file,config):
        with np.load(model_file,allow_pickle=False) as d:
            self.model={k:d[k].copy() for k in d.files}
        self.provenance=json.loads(str(self.model['provenance']))
        if config!=self.provenance['config']:
            raise ValueError('Zombie model/config mismatch')
        self.config=config;self.model_sha256=fingerprint(model_file)

    def observe(self,image):
        if image.size!=(800,600):
            raise ValueError('Need reviewed canonical 800x600 screenshot')
        start=time.perf_counter();rgb=np.asarray(image.convert('RGB'))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        scene=classify(scene_feature(image,self.config),self.model['scene_x'],self.model['scene_y'],
                       self.config['scene_distance_max'],self.config['scene_margin_min'])
        scene['veto_similarities']={}
        for i,v in enumerate(self.config.get('scene_veto_templates',[])):
            x1,y1,x2,y2=v['search_xyxy']
            score=float(cv2.minMaxLoc(cv2.matchTemplate(gray[y1:y2,x1:x2],self.model[f'veto_{i}'],cv2.TM_CCOEFF_NORMED))[1])
            scene['veto_similarities'][v['name']]=round(score,4)
            if score>=v['threshold']:
                scene.update(value='nonplaying',reason=v['name']+'_veto')
        x1,y1,x2,y2=self.config['reward_search_xyxy']
        reward_scores=[]
        for scale in self.config.get('reward_scales',[1.0]):
            template=cv2.resize(self.model['reward'],None,fx=scale,fy=scale)
            reward_scores.append(float(cv2.minMaxLoc(cv2.matchTemplate(gray[y1:y2,x1:x2],template,cv2.TM_CCOEFF_NORMED))[1]))
        reward_score=max(reward_scores)
        if reward_score>=self.config['reward_threshold']:
            scene={**scene,'value':'nonplaying','reason':'reward_card_veto'}
        scene['reward_similarity']=round(reward_score,4)
        candidates=[]
        if scene['value']=='playing':
            ax,ay=self.config['anchor_xy'];w,h=self.config['patch_wh']
            # Row-specific head bands are screenshot features, never hidden game positions.
            for row,(low,high) in enumerate(self.config['head_bands_y'],1):
                top=max(0,low-ay);bottom=min(600,high-ay+h)
                roi=gray[top:bottom,20:800]
                for i,template in enumerate(self.model['templates']):
                    scores=cv2.matchTemplate(roi,template,cv2.TM_CCOEFF_NORMED)
                    # Local maxima avoid enumerating every above-threshold neighboring pixel.
                    maxima=(scores>=self.config['match_threshold']) & (scores>=cv2.dilate(scores,np.ones((9,9),np.uint8)))
                    ys,xs=np.where(maxima)
                    for sy,sx in zip(ys,xs):
                        x,y=int(sx+20+ax),int(sy+top+ay)
                        if low<=y<=high:
                            candidates.append({'row':row,'head_xy':[x,y],
                                'box_xyxy':[x-ax,y-ay,x-ax+w,y-ay+h],
                                'similarity':round(float(scores[sy,sx]),5),'template_index':i})
        accepted=[];radius=self.config['nms_radius_pixels']
        for c in sorted(candidates,key=lambda z:z['similarity'],reverse=True):
            if any(np.linalg.norm(np.subtract(c['head_xy'],a['head_xy']))<radius for a in accepted):
                continue
            accepted.append(c)
        accepted.sort(key=lambda c:(c['row'],c['head_xy'][0]))
        for i,c in enumerate(accepted,1):
            c.update(detection_id=f'z{i}',kind='zombie_candidate',type_identity=None,
                     alive=None,health=None,threat_level=None,
                     position_semantics='visible_head_reference_in_canonical_pixels_not_feet')
        return {'scene':scene,'candidates':accepted,'coverage':'partial_unknown',
                'empty_lane_verified':False,'model_sha256':self.model_sha256,
                'confidence_semantics':'uncalibrated_template_similarity_not_probability',
                'processing_ms':round((time.perf_counter()-start)*1000,3)}
