"""Tilted fallback requires the whole supporting component inside the pool."""
import time
import cv2
import numpy as np
from .bucket_water_separated import SeparatedBucketWaterPerception
from .bucket_water_oriented import tilted_bucket_contexts


def interior_tilted_contexts(rgb,config):
    proposals,checks=tilted_bucket_contexts(rgb,config)
    right=config['bucket_orientation_component_right_max']
    retained=[]
    for check in checks:
        if check['helmet_box_xyxy'][2]>=right:
            check.update(orientation_gates_accepted=check['accepted'],accepted=False,
                reason='tilted_component_not_fully_inside_pool_interior')
    for proposal in proposals:
        if proposal['helmet_support']['accepted']:retained.append(proposal)
    return retained,checks


class InteriorBucketWaterPerception(SeparatedBucketWaterPerception):
    def observe(self,image):
        start=time.perf_counter();result=super().observe(image)
        proposals,checks=interior_tilted_contexts(np.asarray(image.convert('RGB')),self.config) if result['scene']['value']=='playing' else ([],[])
        kept=list(result['candidates'])
        for candidate in proposals:
            if any(np.linalg.norm(np.subtract(candidate['head_xy'],old['head_xy']))<self.config['water_pose']['cluster_radius'] for old in kept):continue
            kept.append(candidate)
        kept.sort(key=lambda c:(c['row'],c['head_xy'][0],c['head_xy'][1]))
        for i,candidate in enumerate(kept,1):candidate['detection_id']=f'z{i}'
        result.update(candidates=kept,tilted_bucket_context_checks=checks,processing_ms=round((time.perf_counter()-start)*1000,3))
        return result
