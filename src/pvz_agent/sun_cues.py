"""Partial screenshot sun-appearance cues, separate from the stored sun counter."""
import json
from pathlib import Path
import time
import cv2
import numpy as np


class SunCuePerception:
    def __init__(self,root,config):
        self.root=Path(root);self.config=dict(config)
        with np.load(self.root/config['model'],allow_pickle=False) as bundle:
            self.template=bundle['template'].copy();provenance=json.loads(str(bundle['provenance']))
        if provenance['config']!=config:raise ValueError('Sun-cue model/config mismatch')

    def observe(self,image,*,scene,state_id,capture_time,evidence):
        if image.size!=(800,600):raise ValueError('Canonical screenshot required')
        if not state_id or not capture_time or not evidence.get('image') or not evidence.get('image_sha256'):
            raise ValueError('Current image binding required')
        started=time.perf_counter();candidates=[]
        if scene=='playing':
            rgb=np.asarray(image.convert('RGB'));gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            cfg=self.config;x1,y1,x2,y2=cfg['search_xyxy'];h,w=self.template.shape
            scores=cv2.matchTemplate(gray[y1:y2,x1:x2],self.template,cv2.TM_CCOEFF_NORMED)
            peaks=(scores>=cfg['match_threshold'])&(scores>=cv2.dilate(scores,np.ones((15,15),dtype='uint8')))
            ranked=sorted(((float(scores[sy,sx]),int(sx+x1),int(sy+y1)) for sy,sx in zip(*np.where(peaks))),reverse=True)
            for score,x,y in ranked:
                cx,cy=x+w//2,y+h//2
                if any(np.hypot(cx-c['visible_reference_xy'][0],cy-c['visible_reference_xy'][1])<cfg['cluster_radius'] for c in candidates):continue
                radius=cfg['core_radius'];core=rgb[cy-radius:cy+radius,cx-radius:cx+radius]
                bright=float(np.mean((core[:,:,0]>=230)&(core[:,:,1]>=200)&(core[:,:,2]>=45)))
                hsv=cv2.cvtColor(rgb[y:y+h,x:x+w],cv2.COLOR_RGB2HSV)
                yellow=float(np.mean((hsv[:,:,0]>=20)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=90)&(hsv[:,:,2]>=170)))
                if bright<cfg['bright_core_fraction_min'] or yellow<cfg['yellow_fraction_min']:continue
                candidates.append(dict(cue_id=f'{state_id}:sun{len(candidates)+1}',kind='visible_sun_appearance_candidate',
                    visible_reference_xy=[cx,cy],box_xyxy=[x,y,x+w,y+h],similarity=round(score,5),
                    support=dict(bright_core_fraction=bright,yellow_fraction=yellow),
                    state_id=state_id,capture_time=capture_time,evidence=dict(evidence),
                    interactable_verified=False,value_verified=False,identity_verified=False,action_eligible=False))
        return dict(schema='pvz-sun-appearance/1',state_id=state_id,capture_time=capture_time,evidence=dict(evidence),
            status='observed' if scene=='playing' else 'scene_unverified',candidates=candidates,
            coverage='Single known-source template, full visible context only, excluding HUD and clipped appearances; no absence or clickability guarantee.',
            empty_resource_area_verified=False,input_sent=False,action_eligible=False,processing_ms=round((time.perf_counter()-started)*1000,3))


class ResourceCueStream:
    def __init__(self,base_stream,sun_cues):self.base_stream=base_stream;self.sun_cues=sun_cues

    def observe(self,image,**kwargs):
        started=time.perf_counter();state=self.base_stream.observe(image,**kwargs)
        state['resources']=dict(sun_appearance=self.sun_cues.observe(image,scene=state['scene']['value'],
            **{k:kwargs[k] for k in ['state_id','capture_time','evidence']}))
        state['processing_ms']=round((time.perf_counter()-started)*1000,3)
        return state
