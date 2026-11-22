"""Separate visible vehicle-body channel layered on the frozen head detector."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .perception import fingerprint
from .water_pose_perception import WaterPosePerception
from .observation_stream import ObservationStream
from .tracking import ShortTermTracker


def build_vehicle_model(root, config, output):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Retain existing model')
    if fingerprint(root/config['vehicle_parent_model'])!=config['vehicle_parent_sha256']:
        raise ValueError('Vehicle parent changed')
    with np.load(root/config['vehicle_parent_model'],allow_pickle=False) as bundle:
        arrays={k:bundle[k].copy() for k in bundle.files}
    data=json.loads((root/config['vehicle_dataset']).read_text(encoding='utf-8'));refs=[]
    # Validate every source before constructing or saving a model.
    from PIL import Image
    for ref in data['template_refs']:
        if ref.get('split')!='development' or ref.get('run_id')!='a' or ref['row']!=1:
            raise ValueError('Only A development templates permitted')
        if fingerprint(root/ref['image'])!=ref['sha256']:
            raise ValueError('Template image changed')
        with Image.open(root/ref['image']) as image:
            if image.size!=(800,600):raise ValueError('Canonical image required')
            gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
        item=dict(ref,part_keys=[])
        if len(ref['parts'])<2:raise ValueError('Hood and chassis evidence required')
        for j,(x1,y1,x2,y2) in enumerate(ref['parts']):
            if not (0<=x1<x2<=800 and 0<=y1<y2<=600):raise ValueError('Invalid part')
            key=f'vehicle_{len(refs)}_{j}';arrays[key]=gray[y1:y2,x1:x2].copy();item['part_keys'].append(key)
        refs.append(item)
    provenance=json.loads(str(arrays['provenance']))
    provenance.update(config=config,vehicle_template_refs=refs,
        vehicle_dataset_sha256=fingerprint(root/config['vehicle_dataset']),
        vehicle_parent_sha256=config['vehicle_parent_sha256'],
        vehicle_built_at=datetime.now(timezone.utc).isoformat())
    arrays.update(provenance=np.array(json.dumps(provenance)),vehicle_refs=np.array(json.dumps(refs)))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return provenance


class VehiclePerception(WaterPosePerception):
    def __init__(self,model_file,config):
        super().__init__(model_file,config)
        self.vehicle_refs=json.loads(str(self.model['vehicle_refs']))

    def vehicle_proposals(self,image):
        if image.size!=(800,600):raise ValueError('Canonical image required')
        gray=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
        cfg=self.config['vehicle'];proposals=[]
        for index,ref in enumerate(self.vehicle_refs):
            seed=self.model[ref['part_keys'][0]];height,width=seed.shape
            rx,ry=ref['parts'][0][:2]
            xmin,ymin=cfg['search_x_min'],cfg['search_y_min']
            for visible in range(cfg['min_visible_width'],width+1):
                edge=visible<width
                left=800-visible if edge else xmin
                search=gray[ymin:600,left:800]
                scores=cv2.matchTemplate(search,seed[:,:visible],cv2.TM_CCOEFF_NORMED)
                threshold=cfg['edge_seed_threshold'] if edge else cfg['seed_threshold']
                maxima=(scores>=threshold)&(scores>=cv2.dilate(scores,np.ones((9,9),np.uint8)))
                ys,xs=np.where(maxima)
                for sy,sx in zip(ys,xs):
                    dx,dy=int(sx+left-rx),int(sy+ymin-ry)
                    parts=[float(scores[sy,sx])];fractions=[visible/width];boxes=[];valid=True
                    for j,(key,rect) in enumerate(zip(ref['part_keys'],ref['parts'])):
                        x1,y1,x2,y2=rect;x1+=dx;x2+=dx;y1+=dy;y2+=dy
                        cx1,cy1,cx2,cy2=max(0,x1),max(0,y1),min(800,x2),min(600,y2)
                        fraction=max(0,cx2-cx1)*max(0,cy2-cy1)/((x2-x1)*(y2-y1))
                        if fraction<cfg['min_part_fraction']:
                            valid=False;break
                        boxes.append([cx1,cy1,cx2,cy2])
                        if j:
                            template=self.model[key][cy1-y1:cy2-y1,cx1-x1:cx2-x1]
                            score=float(cv2.matchTemplate(gray[cy1:cy2,cx1:cx2],template,cv2.TM_CCOEFF_NORMED)[0,0])
                            parts.append(score);fractions.append(fraction)
                    if not valid or min(parts)<cfg['part_threshold'] or np.mean(parts)<cfg['context_threshold']:
                        continue
                    fx,fy=ref['front_xy'][0]+dx,ref['front_xy'][1]+dy
                    if not (0<=fx<800 and 0<=fy<600):continue
                    possible=[r for r in (1,2,5,6) if abs(fy-(ref['front_xy'][1]+(r-ref['row'])*cfg['row_spacing_pixels']))<=cfg['row_tolerance']]
                    row=possible[0] if len(possible)==1 else None
                    proposals.append(dict(kind='vehicle_body_candidate',vehicle_class_hypothesis='zamboni',
                        row=row,front_xy=[fx,fy],similarity=round(float(np.mean(parts)),5),
                        part_similarities=[round(v,5) for v in parts],visible_part_fractions=fractions,
                        vehicle_template_index=index,box_xyxy=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)],
                        box_semantics='visible matched hood and chassis parts; not full body',
                        truncated=any(f<1 for f in fractions),row_evidence='learned_hood_chassis_offset_to_land_row',
                        row_evidence_limits='Screenshot offset; no direct wheel contact or memory measurement.',
                        alive=None,health=None,threat_level=None,empty_lane_verified=False))
        kept=[]
        for p in sorted(proposals,key=lambda p:p['similarity'],reverse=True):
            if all(np.linalg.norm(np.subtract(p['front_xy'],q['front_xy']))>=cfg['nms_radius'] for q in kept):
                p['detection_id']=f'v{len(kept)+1}';kept.append(p)
        return kept

    def observe(self,image):
        start=time.perf_counter();result=super().observe(image)
        vehicles=self.vehicle_proposals(image) if result['scene']['value']=='playing' else []
        result.update(vehicle_candidates=[v for v in vehicles if v['row'] is not None],
            uncertain_vehicle_candidates=[v for v in vehicles if v['row'] is None],
            vehicle_coverage='Two A hood/chassis exemplars; no all-vehicle or empty-lane assurance.',
            processing_ms=round((time.perf_counter()-start)*1000,3))
        return result


class VehicleTracker:
    """Reuse geometry association in a separate namespace; expose front semantics."""
    def __init__(self,config):self.geometry=ShortTermTracker(config)

    def update(self,*,run_id,state_id,frame_time,candidates,scene):
        converted=[]
        for candidate in candidates:
            if 'front_xy' not in candidate or 'head_xy' in candidate:raise ValueError('Vehicle front geometry required')
            item=dict(candidate);item['head_xy']=item.pop('front_xy');converted.append(item)
        result=self.geometry.update(run_id=run_id+':vehicles',state_id=state_id,frame_time=frame_time,candidates=converted,scene=scene)
        for c in result['candidates']:c['front_xy']=c.pop('head_xy')
        for track in result['tracks']:
            track['front_xy']=track.pop('head_xy')
            track['velocity_semantics']='apparent_vehicle_front_x_change_not_measured_game_speed'
            for item in track['history']:item['front_xy']=item.pop('head_xy')
        for event in result['events']:
            event['object_channel']='vehicles';event['run_id']=run_id
        return result


class VehicleObservationStream(ObservationStream):
    def __init__(self,detector,field_perception=None):
        super().__init__(detector,field_perception)
        self.vehicle_tracker=VehicleTracker(detector.config['vehicle_tracker'])

    def observe(self,image,**kwargs):
        start=time.perf_counter();result=super().observe(image,**kwargs)
        zombie=result['zombies'];candidates=zombie.pop('vehicle_candidates')
        uncertain=zombie.pop('uncertain_vehicle_candidates');coverage=zombie.pop('vehicle_coverage')
        tracking_run_id=result.get('tracking_context',{}).get('tracking_run_id',kwargs['run_id'])
        assoc=self.vehicle_tracker.update(run_id=tracking_run_id,state_id=kwargs['state_id'],
            frame_time=kwargs['frame_time_seconds'],candidates=candidates,scene=result['scene']['value'])
        for c in assoc['candidates']:c['detection_id']=f'{kwargs["state_id"]}:{c["detection_id"]}'
        for c in uncertain:c['detection_id']=f'{kwargs["state_id"]}:{c["detection_id"]}'
        for i,event in enumerate(assoc['events'],1):
            event.update(event_id=f'{kwargs["state_id"]}:ve{i}',capture_time=kwargs['capture_time'],evidence=dict(kwargs['evidence']))
            if event.get('detection_id'):event['detection_id']=f'{kwargs["state_id"]}:{event["detection_id"]}'
        result.update(schema='pvz-state/3',vehicles=dict(assoc,uncertain_candidates=uncertain,coverage=coverage))
        result['limitations'].append('Vehicle body coverage is experimental; missed or ended tracks do not prove destruction.')
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
