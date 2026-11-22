"""Coarse activation of side-face template pairs, with exact dense verification.

The scout never publishes a candidate. A pair with plausible visible support
is rerun over the original native ROI and all original guards are applied.
This preserves native scores for activated pairs; unseen scout misses remain
an experimental coverage risk and require regression, not an empty-lane claim.
"""
import cv2
import numpy as np
from .boundary_perception import row_at
from .occlusion_color_guard import GuardedOcclusionPerception,face_color_context
from .occlusion_shape import hat_component
from .occlusion_support import hat_support
from .water_head_support import WaterHeadSupportPerception


class FastOcclusionPerception(GuardedOcclusionPerception):
    def __init__(self,model_file,config):
        super().__init__(model_file,config)
        self.scout_templates={}
        for kind,settings in config['occlusion_support'].items():
            indices=range(len(self.model['templates'])) if settings['template_indices']=='all' else settings['template_indices']
            for index in indices:
                for rect in settings['parts']:
                    a,b,c,d=rect
                    self.scout_templates[kind,index,tuple(rect)]=cv2.resize(self.model['templates'][index,b:d,a:c],None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)

    def observe(self,image):
        self.scout_stats=dict(pairs=0,activated_pairs=0,coarse_peaks=0,refined_sites=0)
        return super().observe(image)

    def _context(self,gray,index,rect,x,y,kind,settings,parents):
        ax,ay=self.config['anchor_xy'];w,h=self.config['patch_wh'];left,top=x-ax,y-ay
        if not(0<=left and left+w<=800 and 0<=top and top+h<=600):return None
        patch=gray[top:top+h,left:left+w];template=self.model['templates'][index]
        native=float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
        head=float(cv2.matchTemplate(patch[:24],template[:24],cv2.TM_CCOEFF_NORMED)[0,0])
        if native<settings['native_min'] or head<settings['head_min']:return None
        hat=hat_support(self.current_rgb,x,y,kind,settings['hat'])
        if not hat['accepted']:return None
        shape=hat_component(self.current_rgb,x,y,kind,self.config['occlusion_shape'][kind])
        if not shape['accepted']:return None
        row=row_at(y,self._bands(index)[1])
        if any(p['row']==row and np.linalg.norm(np.subtract(p['head_xy'],[x,y]))<self.config['preserve_parent_reference_radius'] for p in parents):return None
        color=face_color_context(self.current_rgb,x,y,self.config['occlusion_face_color'])
        if not color['accepted']:return None
        return native,hat,shape,color

    def _activate(self,gray,small,kind,index,rect,settings,parents):
        cfg=self.config['occlusion_scout'];a,b,c,d=rect;part=self.model['templates'][index,b:d,a:c]
        scores=cv2.matchTemplate(small[:,200:400],self.scout_templates[kind,index,tuple(rect)],cv2.TM_CCOEFF_NORMED)
        maxima=(scores>=cfg['threshold'])&(scores>=cv2.dilate(scores,np.ones((5,5),np.uint8)))
        ax,ay=self.config['anchor_xy'];bands=self._bands(index)[1];radius=cfg['refine_radius'];slack=cfg['native_slack']
        ys,xs=np.where(maxima);self.scout_stats['coarse_peaks']+=len(xs)
        for sy,sx in zip(ys,xs):
            estimate_x,estimate_y=int(sx*2+400),int(sy*2)
            # Native score coordinates are top-left positions of the visible part.
            x1,y1=max(400,estimate_x-radius-4),max(0,estimate_y-radius-4)
            x2,y2=min(800,estimate_x+radius+4+part.shape[1]),min(600,estimate_y+radius+4+part.shape[0])
            if x2-x1<part.shape[1] or y2-y1<part.shape[0]:continue
            refined=cv2.matchTemplate(gray[y1:y2,x1:x2],part,cv2.TM_CCOEFF_NORMED)
            peaks=(refined>=settings['part_threshold']-slack)&(refined>=cv2.dilate(refined,np.ones((9,9),np.uint8))-slack)
            py,px=np.where(peaks)
            for ry,rx in zip(py,px):
                if abs(x1+int(rx)-estimate_x)>radius or abs(y1+int(ry)-estimate_y)>radius:continue
                x,y=x1+int(rx)-a+ax,y1+int(ry)-b+ay
                if row_at(y,bands) is None:continue
                self.scout_stats['refined_sites']+=1
                if self._context(gray,index,rect,x,y,kind,settings,parents) is not None:return True
        return False

    def _search(self,gray):
        parents=WaterHeadSupportPerception._search(self,gray);proposals=list(parents)
        small=cv2.resize(gray,(400,300),interpolation=cv2.INTER_AREA)
        ax,ay=self.config['anchor_xy'];w,h=self.config['patch_wh']
        for kind,settings in self.config['occlusion_support'].items():
            indices=range(len(self.model['templates'])) if settings['template_indices']=='all' else settings['template_indices']
            for index in indices:
                template=self.model['templates'][index];pose,bands=self._bands(index);valid=[b for b in bands if b is not None]
                for rect in settings['parts']:
                    self.scout_stats['pairs']+=1
                    if not self._activate(gray,small,kind,index,rect,settings,parents):continue
                    self.scout_stats['activated_pairs']+=1
                    # Exact original full native ROI, thresholds and local maxima.
                    a,b,c,d=rect;part=template[b:d,a:c]
                    top=max(0,min(v[0] for v in valid)-ay+b);bottom=min(600,max(v[1] for v in valid)-ay+d)
                    scores=cv2.matchTemplate(gray[top:bottom,400:800],part,cv2.TM_CCOEFF_NORMED)
                    maxima=(scores>=settings['part_threshold'])&(scores>=cv2.dilate(scores,np.ones((9,9),np.uint8)))
                    for sy,sx in zip(*np.where(maxima)):
                        x,y=int(sx+400-a+ax),int(sy+top-b+ay);row=row_at(y,bands)
                        if row is None:continue
                        context=self._context(gray,index,rect,x,y,kind,settings,parents)
                        if context is None:continue
                        native,hat,shape,color=context
                        candidate=self._proposal(gray,index,x,y,w,scores[sy,sx],row,pose)
                        candidate.update(template_bank='visible_side_face_with_hat_context',visible_face_part=rect,
                            native_context_similarity=round(native,5),hat_support=hat,hat_hypothesis=kind,
                            score_semantics='Visible face-part NCC with separate whole-patch/head/hat gates; not probability or confirmed identity',
                            hat_component_support=shape,face_color_context=color)
                        proposals.append(candidate)
        return proposals
