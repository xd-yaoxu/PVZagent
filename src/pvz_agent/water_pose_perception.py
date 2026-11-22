"""Water-lane pose context from visible head/body/flipper parts, never input."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .edge_bank_perception import EdgeBankPerception
from .perception import fingerprint


def build_water_model(root, config, output):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Retain existing model')
    if fingerprint(root/config['parent_model']) != config['parent_model_sha256']:
        raise ValueError('Parent model changed')
    with np.load(root/config['parent_model'], allow_pickle=False) as bundle:
        arrays = {k:bundle[k].copy() for k in bundle.files}
    labels = json.loads((root/config['water_dataset']).read_text(encoding='utf-8'))
    refs = []
    for ref in labels['pose_refs']:
        if ref.get('split')!='development' or ref.get('run_id')!='human-20261023-a':
            raise ValueError('Water templates must be explicit A development records')
        if ref['row'] not in (3,4) or not ref['parts']:
            raise ValueError('Water row and visible parts required')
        path = root/ref['image']
        if fingerprint(path)!=ref['sha256']:
            raise ValueError('Water development image hash mismatch')
        with Image.open(path) as image:
            if image.size != (800,600):
                raise ValueError('Canonical screenshots required')
            gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        item = dict(ref, source_image_sha256=fingerprint(path), part_keys=[])
        for j, (x1,y1,x2,y2) in enumerate(ref['parts']):
            if not (0 <= x1 < x2 <= 800 and 0 <= y1 < y2 <= 600):
                raise ValueError('Invalid part rectangle')
            key = f'water_{len(refs)}_{j}'
            arrays[key] = gray[y1:y2,x1:x2].copy()
            item['part_keys'].append(key)
        refs.append(item)
    provenance = json.loads(str(arrays['provenance']))
    provenance.update(config=config, parent_model_sha256=config['parent_model_sha256'],
                      water_template_refs=refs, water_dataset_sha256=fingerprint(root/config['water_dataset']),
                      water_completed_at=datetime.now(timezone.utc).isoformat())
    provenance['training_ids'] = list(dict.fromkeys(provenance['training_ids']+[r['id'] for r in refs]))
    arrays.update(provenance=np.array(json.dumps(provenance)), water_refs=np.array(json.dumps(refs)))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return provenance


def water_fraction(rgb):
    if not rgb.size:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean((hsv[:,:,0]>=80)&(hsv[:,:,0]<=115)&(hsv[:,:,1]>=40)&(hsv[:,:,2]>=80)))


class WaterPosePerception(EdgeBankPerception):
    def __init__(self, model_file, config):
        super().__init__(model_file, config)
        self.water_refs = json.loads(str(self.model['water_refs']))

    def _water_proposals(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        settings = self.config['water_pose']
        proposals = []
        for index, ref in enumerate(self.water_refs):
            head_part = self.model[ref['part_keys'][0]]
            hx1,hy1,hx2,hy2 = ref['parts'][0]
            hh,hw = head_part.shape
            for row in (3,4):
                shift = (row-ref['row'])*settings['row_spacing_pixels']
                top = max(0,hy1+shift-settings['y_tolerance'])
                bottom = min(600,hy2+shift+settings['y_tolerance'])
                scores = cv2.matchTemplate(gray[top:bottom,settings['search_x_min']:800],head_part,cv2.TM_CCOEFF_NORMED)
                maxima = ((scores>=settings['head_threshold']) &
                          (scores>=cv2.dilate(scores,np.ones((7,7),np.uint8))))
                ys,xs = np.where(maxima)
                for sy,sx in zip(ys,xs):
                    dx = int(sx+settings['search_x_min']-hx1)
                    dy = int(sy+top-hy1)
                    part_scores = [float(scores[sy,sx])]
                    valid = True
                    for key, rect in zip(ref['part_keys'][1:],ref['parts'][1:]):
                        x1,y1,x2,y2 = rect
                        x1,x2,y1,y2 = x1+dx,x2+dx,y1+dy,y2+dy
                        if not (0<=x1<x2<=800 and 0<=y1<y2<=600):
                            valid = False
                            break
                        score = float(cv2.matchTemplate(gray[y1:y2,x1:x2],self.model[key],cv2.TM_CCOEFF_NORMED)[0,0])
                        part_scores.append(score)
                    if not valid or min(part_scores)<settings['part_threshold']:
                        continue
                    context_score = float(np.mean(part_scores))
                    if context_score<settings['context_threshold']:
                        continue
                    fraction = None
                    if 'water_check_xyxy' in ref:
                        x1,y1,x2,y2 = ref['water_check_xyxy']
                        x1,x2,y1,y2 = x1+dx,x2+dx,y1+dy,y2+dy
                        if not (0<=x1<x2<=800 and 0<=y1<y2<=600):
                            continue
                        fraction = water_fraction(rgb[y1:y2,x1:x2])
                        if fraction<settings['water_fraction_min']:
                            continue
                    x,y = ref['head_xy'][0]+dx,ref['head_xy'][1]+dy
                    if not (0<=x<800 and 0<=y<600):
                        continue
                    proposals.append({'row':row,'head_xy':[x,y], 'similarity':round(context_score,5),
                        'context_similarity':round(context_score,5),'part_similarities':[round(s,5) for s in part_scores],
                        'water_template_index':index,'water_pose':ref['pose'],'water_fraction':fraction,
                        'box_xyxy':[hx1+dx,hy1+dy,hx2+dx,hy2+dy],
                        'pose_context_xyxy':[min(p[0] for p in ref['parts'])+dx,min(p[1] for p in ref['parts'])+dy,
                                             max(p[2] for p in ref['parts'])+dx,max(p[3] for p in ref['parts'])+dy]})
        return proposals

    def observe(self,image):
        start=time.perf_counter()
        result=super().observe(image)
        poses=self._water_proposals(np.asarray(image.convert('RGB'))) if result['scene']['value']=='playing' else []
        settings=self.config['water_pose'];groups=[]
        for pose in sorted(poses,key=lambda p:p['context_similarity'],reverse=True):
            group=next((g for g in groups if np.linalg.norm(np.subtract(pose['head_xy'],g[0]['head_xy']))<settings['cluster_radius']),None)
            if group is None:
                groups.append([pose])
            else:
                group.append(pose)
        candidates=list(result['candidates']);uncertain=[];contexts=[]
        for candidate in candidates:
            candidate.setdefault('row_evidence','head_band_or_native_edge_template')
            candidate.setdefault('row_evidence_limits','Head height may cross rows during a water-entry animation.')
        for group in groups:
            best=group[0]
            other=max((p['context_similarity'] for p in group if p['row']!=best['row']),default=-1)
            nearby=[c for c in candidates if np.linalg.norm(np.subtract(c['head_xy'],best['head_xy']))<settings['cluster_radius']]
            if best['context_similarity']-other<settings['row_margin']:
                uncertain.append({'head_xy':best['head_xy'],'row':None,'possible_rows':[3,4],
                    'reason':'conflicting_water_pose_context','context_similarities':{r:max((p['context_similarity'] for p in group if p['row']==r),default=None) for r in (3,4)}})
                candidates=[c for c in candidates if c not in nearby]
                continue
            ranked=sorted((float(np.linalg.norm(np.subtract(c['head_xy'],best['head_xy']))),i,c)
                          for i,c in enumerate(nearby))
            if len(ranked)>1 and ranked[1][0]-ranked[0][0]<settings['association_margin']:
                uncertain.append({'head_xy':best['head_xy'],'row':None,'possible_rows':[3,4],
                                  'reason':'multiple_visible_candidates_near_pose'})
                continue
            # Keep the original screenshot reference point if already detected.
            if nearby:
                candidate=ranked[0][2]
                candidates=[c for c in candidates if c is not candidate]
                candidate.update(previous_head_band_row=candidate['row'],row=best['row'])
            else:
                candidate=dict(best,kind='zombie_candidate',type_identity=None,alive=None,health=None,threat_level=None,
                    truncated=False,visible_template_fraction=1.0,position_semantics='visible_head_reference_from_pose_parts')
            candidate.update(row_evidence='visible_water_pose_parts',water_pose=best['water_pose'],
                context_similarity=best['context_similarity'],part_similarities=best['part_similarities'],
                water_template_index=best['water_template_index'],pose_context_xyxy=best['pose_context_xyxy'],
                row_evidence_limits='Learned screenshot pose offset; not directly measured feet or future landing.')
            candidates.append(candidate);contexts.append(best)
        candidates.sort(key=lambda c:(c['row'],c['head_xy'][0],c['head_xy'][1]))
        for i,candidate in enumerate(candidates,1):
            candidate['detection_id']=f'z{i}'
        result.update(candidates=candidates,water_contexts=contexts,uncertain_water_candidates=uncertain,
            processing_ms=round((time.perf_counter()-start)*1000,3))
        return result
