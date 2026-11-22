"""Visible head + duck-ring + adjacent water support, an explicit experiment."""
import cv2
import numpy as np
from .boundary_perception import row_at
from .vehicle_color_perception_v18 import VehicleColorPerception
from .water_pose_perception import water_fraction


def duck_support(rgb,x,y,config):
    dx1,dy1,dx2,dy2=config['duck_roi_offset']
    x1,y1,x2,y2=max(0,x+dx1),max(0,y+dy1),min(800,x+dx2),min(600,y+dy2)
    roi=rgb[y1:y2,x1:x2]
    if not roi.size:return dict(accepted=False,reason='duck_roi_empty')
    hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
    mask=(hsv[:,:,0]>=20)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=120)&(hsv[:,:,2]>=140)
    yellow=int(mask.sum());fraction=float(mask.mean())
    wx1,wy1,wx2,wy2=config['water_roi_offset']
    water=water_fraction(rgb[max(0,y+wy1):min(600,y+wy2),max(0,x+wx1):min(800,x+wx2)])
    accepted=yellow>=config['yellow_pixels_min'] and fraction>=config['yellow_fraction_min'] and water>=config['water_fraction_min']
    return dict(accepted=accepted,yellow_pixels=yellow,yellow_fraction=fraction,water_fraction=water,
                duck_roi_xyxy=[x1,y1,x2,y2],reason=None if accepted else 'duck_or_water_support_rejected')


class WaterHeadSupportPerception(VehicleColorPerception):
    def observe(self,image):
        self.water_head_checks=[]
        # No color or head proposal search when the frozen scene gate vetoes.
        self.current_rgb=np.asarray(image.convert('RGB'))
        return super().observe(image)

    def _search(self,gray):
        proposals=super()._search(gray);cfg=self.config['water_head_support']
        w,h=self.config['patch_wh'];ax,ay=self.config['anchor_xy'];height=cfg['head_rows']
        for index,template in enumerate(self.model['templates']):
            pose,bands=self._bands(index)
            for row in (3,4):
                band=bands[row-1]
                if band is None:continue
                top=max(0,band[0]-ay);bottom=min(600-h+height,band[1]-ay+height)
                scores=cv2.matchTemplate(gray[top:bottom,400:800],template[:height],cv2.TM_CCOEFF_NORMED)
                maxima=(scores>=cfg['head_threshold'])&(scores>=cv2.dilate(scores,np.ones((9,9),np.uint8)))
                for sy,sx in zip(*np.where(maxima)):
                    left,upper=int(sx+400),int(sy+top);x,y=left+ax,upper+ay
                    if row_at(y,bands)!=row:continue
                    patch=gray[upper:upper+h,left:left+w]
                    native=float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
                    check=dict(template_index=index,row=row,head_xy=[x,y],head_similarity=float(scores[sy,sx]),
                        native_similarity=native,head_threshold=cfg['head_threshold'],native_threshold=cfg['native_context_min'])
                    if native<cfg['native_context_min']:
                        check['gate']='native_context_rejected';self.water_head_checks.append(check);continue
                    support=duck_support(self.current_rgb,x,y,cfg);check.update(support=support,gate='passed' if support['accepted'] else support['reason'])
                    self.water_head_checks.append(check)
                    if not support['accepted']:continue
                    candidate=self._proposal(gray,index,x,y,w,scores[sy,sx],row,pose)
                    candidate.update(template_bank='water_head_and_duck_context',native_context_similarity=round(native,5),
                        duck_support=support,row_evidence='head_band_with_visible_duck_and_adjacent_water',
                        row_evidence_limits='Screenshot context only; not measured feet, actual death or a verified empty lane.',
                        score_semantics='Head-part NCC with separate native and duck/water gates; not probability')
                    proposals.append(candidate)
        return proposals
