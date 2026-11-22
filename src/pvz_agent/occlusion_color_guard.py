"""Abstain on dark, skinless visual remnants in the new occlusion channel."""
import numpy as np
from .occlusion_distinct import DistinctOcclusionPerception


def face_color_context(rgb,x,y,settings):
    roi=rgb[max(0,y-8):min(600,y+17),max(0,x-12):min(800,x+12)].astype('int16')
    if not roi.size:return dict(accepted=False,reason='empty_face_context')
    r,g,b=roi[:,:,0],roi[:,:,1],roi[:,:,2]
    skin=(np.min(roi,axis=2)>=45)&(np.ptp(roi,axis=2)<=90)&(g-r>=0)&(g-r<=65)&(g-b>=0)&(g-b<=65)
    dark=np.max(roi,axis=2)<55;fraction=float(skin.mean());black=float(dark.mean())
    reject=black>=settings['dark_fraction_min'] and fraction<settings['skin_fraction_min']
    return dict(accepted=not reject,dark_fraction=black,skin_like_fraction=fraction,
        reason='dark_region_without_visible_skin_context' if reject else None,
        limits='Color-based visual abstention; not a death, health or identity determination.')


class GuardedOcclusionPerception(DistinctOcclusionPerception):
    def observe(self,image):
        self.face_color_checks=[]
        return super().observe(image)

    def _search(self,gray):
        proposals=super()._search(gray);retained=[]
        for candidate in proposals:
            if candidate.get('template_bank')!='visible_side_face_with_hat_context':
                retained.append(candidate);continue
            x,y=candidate['head_xy'];support=face_color_context(self.current_rgb,x,y,self.config['occlusion_face_color'])
            self.face_color_checks.append(dict(head_xy=[x,y],row=candidate['row'],support=support))
            if support['accepted']:
                candidate['face_color_context']=support;retained.append(candidate)
        return retained
