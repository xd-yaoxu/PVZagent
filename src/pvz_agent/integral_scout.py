"""Vectorized necessary hat-pixel gate before expensive native refinement."""
import cv2
import numpy as np
from .boundary_perception import row_at
from .fast_occlusion import FastOcclusionPerception


def integral_counts(integral,x1,y1,x2,y2):
    return integral[y2,x2]-integral[y1,x2]-integral[y2,x1]+integral[y1,x1]


class IntegralScoutPerception(FastOcclusionPerception):
    def _search(self,gray):
        rgb=self.current_rgb;hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        masks=dict(cone=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=120)&(hsv[:,:,2]>=135),
            bucket=(np.min(rgb,axis=2)>=95)&(np.ptp(rgb.astype('int16'),axis=2)<=35))
        self.hat_integrals={k:cv2.integral(v.astype('uint8'),sdepth=cv2.CV_32S) for k,v in masks.items()}
        self.scout_stats['gated_peaks']=0
        return super()._search(gray)

    def _activate(self,gray,small,kind,index,rect,settings,parents):
        cfg=self.config['occlusion_scout'];a,b,c,d=rect;part=self.model['templates'][index,b:d,a:c]
        scores=cv2.matchTemplate(small[:,200:400],self.scout_templates[kind,index,tuple(rect)],cv2.TM_CCOEFF_NORMED)
        maxima=(scores>=cfg['threshold'])&(scores>=cv2.dilate(scores,np.ones((5,5),np.uint8)))
        ys,xs=np.where(maxima);self.scout_stats['coarse_peaks']+=len(xs)
        ax,ay=self.config['anchor_xy'];bands=self._bands(index)[1];radius=cfg['refine_radius'];slack=cfg['native_slack']
        hx,hy=xs*2+400-a+ax,ys*2-b+ay
        dx1,dy1,dx2,dy2=settings['hat']['roi_offset']
        # Every refined hat ROI lies inside this expanded ROI. Rejecting an
        # insufficient total pixel count is necessary, never a positive match.
        x1=np.clip(hx+dx1-radius,0,800);x2=np.clip(hx+dx2+radius,0,800)
        y1=np.clip(hy+dy1-radius,0,600);y2=np.clip(hy+dy2+radius,0,600)
        counts=integral_counts(self.hat_integrals[kind],x1,y1,x2,y2)
        valid=np.zeros(len(xs),dtype=bool)
        for band in bands:
            if band is not None:valid|=(hy>=band[0]-radius)&(hy<=band[1]+radius)
        valid&=counts>=settings['hat']['pixels_min']
        ys,xs=ys[valid],xs[valid];self.scout_stats['gated_peaks']+=len(xs)
        for sy,sx in zip(ys,xs):
            estimate_x,estimate_y=int(sx*2+400),int(sy*2)
            rx1,ry1=max(400,estimate_x-radius-4),max(0,estimate_y-radius-4)
            rx2,ry2=min(800,estimate_x+radius+4+part.shape[1]),min(600,estimate_y+radius+4+part.shape[0])
            if rx2-rx1<part.shape[1] or ry2-ry1<part.shape[0]:continue
            refined=cv2.matchTemplate(gray[ry1:ry2,rx1:rx2],part,cv2.TM_CCOEFF_NORMED)
            peaks=(refined>=settings['part_threshold']-slack)&(refined>=cv2.dilate(refined,np.ones((9,9),np.uint8))-slack)
            for ry,rx in zip(*np.where(peaks)):
                if abs(rx1+int(rx)-estimate_x)>radius or abs(ry1+int(ry)-estimate_y)>radius:continue
                x,y=rx1+int(rx)-a+ax,ry1+int(ry)-b+ay
                if row_at(y,bands) is None:continue
                self.scout_stats['refined_sites']+=1
                if self._context(gray,index,rect,x,y,kind,settings,parents) is not None:return True
        return False
