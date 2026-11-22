"""Reject plant colors using bounded hat components above a visible face."""
import cv2
import numpy as np
from .occlusion_support import OcclusionSupportPerception


def hat_component(rgb,x,y,kind,settings):
    x1,y1,x2,y2=max(0,x-30),max(0,y-60),min(800,x+45),min(600,y+25)
    roi=rgb[y1:y2,x1:x2]
    if not roi.size:return dict(accepted=False,components=[],reason='empty_roi')
    if kind=='cone':
        hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
        mask=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=120)&(hsv[:,:,2]>=135)
    else:mask=(np.min(roi,axis=2)>=95)&(np.ptp(roi.astype('int16'),axis=2)<=35)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype('uint8'))
    possible=[]
    for index in range(1,count):
        left,top,width,height,area=map(int,stats[index]);bottom=y1+top+height-1-y
        reltop=y1+top-y;cx=x1+float(centers[index,0])-x
        accepted=(area>=settings['area_min'] and settings['width_min']<=width<=settings['width_max'] and
            settings['height_min']<=height<=settings['height_max'] and reltop<=-18 and
            -25<=bottom<=3 and -15<=cx<=30)
        if accepted:possible.append(dict(box_xyxy=[x1+left,y1+top,x1+left+width,y1+top+height],area=area,
                                        top_offset=reltop,bottom_offset=bottom,center_dx=cx))
    return dict(accepted=bool(possible),components=possible,roi_xyxy=[x1,y1,x2,y2],
                reason=None if possible else 'no_bounded_hat_component_above_face')


class ShapeOcclusionPerception(OcclusionSupportPerception):
    def observe(self,image):
        self.shape_checks=[]
        return super().observe(image)

    def _search(self,gray):
        proposals=super()._search(gray);retained=[]
        for candidate in proposals:
            if candidate.get('template_bank')!='visible_side_face_with_hat_context':
                retained.append(candidate);continue
            kind=candidate['hat_hypothesis'];x,y=candidate['head_xy']
            support=hat_component(self.current_rgb,x,y,kind,self.config['occlusion_shape'][kind])
            self.shape_checks.append(dict(row=candidate['row'],head_xy=[x,y],kind=kind,
                template_index=candidate['template_index'],support=support))
            if support['accepted']:
                candidate['hat_component_support']=support;retained.append(candidate)
        return retained
