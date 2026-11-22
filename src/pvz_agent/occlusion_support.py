"""Visible side-face patches plus hat evidence, inherited template pixels only."""
import cv2
import numpy as np
from .boundary_perception import row_at
from .water_head_support import WaterHeadSupportPerception


def hat_support(rgb,x,y,kind,config):
    dx1,dy1,dx2,dy2=config['roi_offset'];x1,y1,x2,y2=max(0,x+dx1),max(0,y+dy1),min(800,x+dx2),min(600,y+dy2)
    roi=rgb[y1:y2,x1:x2]
    if not roi.size:return dict(accepted=False,reason='hat_roi_empty')
    if kind=='cone':
        hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
        mask=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=120)&(hsv[:,:,2]>=135)
    else:
        mask=(np.min(roi,axis=2)>=95)&(np.ptp(roi.astype('int16'),axis=2)<=35)
    count=int(mask.sum());fraction=float(mask.mean())
    return dict(accepted=count>=config['pixels_min'] and fraction>=config['fraction_min'],
                pixels=count,fraction=fraction,kind=kind,roi_xyxy=[x1,y1,x2,y2])


class OcclusionSupportPerception(WaterHeadSupportPerception):
    def observe(self,image):
        self.occlusion_checks=[]
        return super().observe(image)

    def _search(self,gray):
        proposals=super()._search(gray);cfg=self.config['occlusion_support']
        ax,ay=self.config['anchor_xy'];w,h=self.config['patch_wh']
        for kind,settings in cfg.items():
            indices=range(len(self.model['templates'])) if settings['template_indices']=='all' else settings['template_indices']
            for index in indices:
                template=self.model['templates'][index];pose,bands=self._bands(index)
                valid=[b for b in bands if b is not None]
                for rect in settings['parts']:
                    a,b,c,d=rect;part=template[b:d,a:c]
                    top=max(0,min(v[0] for v in valid)-ay+b)
                    bottom=min(600,max(v[1] for v in valid)-ay+d)
                    scores=cv2.matchTemplate(gray[top:bottom,400:800],part,cv2.TM_CCOEFF_NORMED)
                    maxima=(scores>=settings['part_threshold'])&(scores>=cv2.dilate(scores,np.ones((9,9),np.uint8)))
                    for sy,sx in zip(*np.where(maxima)):
                        left,upper=int(sx+400-a),int(sy+top-b);x,y=left+ax,upper+ay
                        row=row_at(y,bands)
                        if row is None:continue
                        check=dict(kind=kind,template_index=index,part=rect,head_xy=[x,y],row=row,part_similarity=float(scores[sy,sx]))
                        if not(0<=left and left+w<=800 and 0<=upper and upper+h<=600):
                            check['gate']='full_context_out_of_bounds';self.occlusion_checks.append(check);continue
                        patch=gray[upper:upper+h,left:left+w]
                        native=float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
                        head=float(cv2.matchTemplate(patch[:24],template[:24],cv2.TM_CCOEFF_NORMED)[0,0])
                        support=hat_support(self.current_rgb,x,y,kind,settings['hat'])
                        check.update(native_similarity=native,head_similarity=head,hat=support)
                        if native<settings['native_min'] or head<settings['head_min']:
                            check['gate']='native_or_head_context_rejected'
                        elif not support['accepted']:check['gate']='hat_support_rejected'
                        else:check['gate']='passed'
                        self.occlusion_checks.append(check)
                        if check['gate']!='passed':continue
                        candidate=self._proposal(gray,index,x,y,w,scores[sy,sx],row,pose)
                        candidate.update(template_bank='visible_side_face_with_hat_context',visible_face_part=rect,
                            native_context_similarity=round(native,5),hat_support=support,
                            hat_hypothesis=kind,score_semantics='Visible face-part NCC with separate whole-patch/head/hat gates; not probability or confirmed identity')
                        proposals.append(candidate)
        return proposals
