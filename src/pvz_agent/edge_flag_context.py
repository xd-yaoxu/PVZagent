"""Reject narrow red ties in the experimental edge-context channel."""
import time
from .edge_context import EdgeContextPerception


class FlagEdgeContextPerception(EdgeContextPerception):
    def observe(self,image):
        start=time.perf_counter();result=super().observe(image)
        settings=self.config['edge_flag_shape'];kept=[]
        for cue in result['uncertain_edge_presence']:
            x1,y1,x2,y2=cue['visible_flag_box_xyxy'];width,height=x2-x1,y2-y1
            if width<settings['width_min'] or width/height<settings['width_height_min']:continue
            # Even a broad red component is not confirmed semantic flag identity.
            cue['visible_red_component_box_xyxy']=cue.pop('visible_flag_box_xyxy')
            cue['shape_evidence']=dict(width=width,height=height,width_height_ratio=width/height)
            kept.append(cue)
        result['uncertain_edge_presence']=kept
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
