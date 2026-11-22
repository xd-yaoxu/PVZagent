"""Vehicle body candidates supported by visible hood/chassis color geometry."""
import cv2
import numpy as np
from .vehicle_perception import VehiclePerception


class VehicleColorPerception(VehiclePerception):
    def vehicle_proposals(self,image):
        proposals=super().vehicle_proposals(image)
        cfg=self.config['vehicle_color'];rgb=np.asarray(image.convert('RGB'))
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=((hsv[:,:,0]>=cfg['blue_hue'][0])&(hsv[:,:,0]<=cfg['blue_hue'][1])&
              (hsv[:,:,1]>=cfg['blue_saturation_min'])&(hsv[:,:,2]>=cfg['blue_value_min'])).astype('uint8')
        mask[:self.config['vehicle']['search_y_min'],:]=0
        mask[:,:self.config['vehicle']['search_x_min']]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        # Fixed screenshot HUD can join a row-6 chassis; exclude those pixels explicitly.
        x1,y1,x2,y2=cfg['hud_exclusion_xyxy'];mask[y1:y2,x1:x2]=0
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        for index in range(1,count):
            x,y,w,h,area=map(int,stats[index])
            if w<cfg['width_min'] or h<cfg['height_min'] or area<cfg['area_min']:continue
            support=labels[y:y+h,x:min(800,x+32)]==index
            row_counts=support.sum(axis=1)
            starts=np.flatnonzero(row_counts>=min(cfg['blue_row_width_min'],w))
            if not len(starts):continue
            by=y+int(starts[0]);top=max(80,by-cfg['hood_height'])
            hood=rgb[top:by,x:min(800,x+min(w,cfg['hood_width_max']))]
            if not hood.size:continue
            white=float(np.mean((np.min(hood,axis=2)>=cfg['white_value_min'])&
                                (np.ptp(hood.astype('int16'),axis=2)<cfg['white_channel_spread_max'])))
            if white<cfg['white_fraction_min']:continue
            possible=[r for r in (1,2,5,6) if abs(by-(cfg['row1_reference_y']+(r-1)*85))<=cfg['row_tolerance']]
            row=possible[0] if len(possible)==1 else None
            # A matching template supplies the sharper reference; color support remains diagnostic.
            near=[p for p in proposals if p['row']==row and np.linalg.norm(np.subtract(p['front_xy'],[x,by]))<35]
            context=dict(blue_area=area,white_fraction=round(white,5),blue_reference_xy=[x,by],
                hud_pixels_excluded=cfg['hud_exclusion_xyxy'])
            if near:
                for p in near:p['color_context']=context
                continue
            proposals.append(dict(kind='vehicle_body_candidate',vehicle_class_hypothesis='zamboni',row=row,
                front_xy=[x,by],box_xyxy=[x,top,min(800,x+w),min(600,y+h)],
                box_semantics='visible hood and chassis color support, HUD excluded; not full vehicle extent',
                position_semantics='left visible chassis reference; hood nose may project farther forward',
                similarity=None,score_semantics='color geometry support; no template similarity or probability',
                color_context=context,vehicle_template_index=None,
                truncated=x+w==800 or y+h>=y1,
                row_evidence='visible_white_hood_over_blue_chassis_land_band',
                row_evidence_limits='Fixed screenshot color/height bands; no wheel contact, identity or speed guarantee.',
                alive=None,health=None,threat_level=None,empty_lane_verified=False))
        for i,p in enumerate(proposals,1):p['detection_id']=f'v{i}'
        return proposals
