"""Keep inherited references before admitting additional occluded heads."""
import numpy as np
from .occlusion_shape import ShapeOcclusionPerception


class DistinctOcclusionPerception(ShapeOcclusionPerception):
    def observe(self,image):
        self.parent_reference_checks=[]
        return super().observe(image)

    def _search(self,gray):
        proposals=super()._search(gray)
        parents=[p for p in proposals if p.get('template_bank')!='visible_side_face_with_hat_context']
        retained=[];radius=self.config['preserve_parent_reference_radius']
        for candidate in proposals:
            if candidate.get('template_bank')!='visible_side_face_with_hat_context':
                retained.append(candidate);continue
            nearby=next((p for p in parents if p['row']==candidate['row'] and
                         np.linalg.norm(np.subtract(p['head_xy'],candidate['head_xy']))<radius),None)
            if nearby is not None:
                self.parent_reference_checks.append(dict(head_xy=candidate['head_xy'],row=candidate['row'],
                    parent_head_xy=nearby['head_xy'],reason='preserve_existing_reference_before_global_nms'))
            else:retained.append(candidate)
        return retained
