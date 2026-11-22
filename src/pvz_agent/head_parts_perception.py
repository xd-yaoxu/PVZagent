"""Experimental head-part matches with native full-patch context guards.

Templates are inherited development pixels. This expands pose/occlusion
coverage, without relaxing the frozen whole-patch detector or fabricating edge
pixels. New similarities are uncalibrated scores, not probabilities.
"""
import cv2
import numpy as np
from .boundary_perception import row_at
from .vehicle_color_perception_v18 import VehicleColorPerception


class HeadPartsPerception(VehicleColorPerception):
    def _search(self, gray):
        proposals = super()._search(gray)
        cfg = self.config['head_parts']; w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']; rows = cfg['head_rows']
        for index, template in enumerate(self.model['templates']):
            pose, bands = self._bands(index)
            valid = [b for b in bands if b is not None]
            top = max(0, min(b[0] for b in valid)-ay)
            bottom = min(600-h+rows, max(b[1] for b in valid)-ay+rows)
            scores = cv2.matchTemplate(gray[top:bottom,20:800], template[:rows], cv2.TM_CCOEFF_NORMED)
            maxima = (scores >= cfg['head_threshold']) & (scores >= cv2.dilate(scores,np.ones((9,9),np.uint8)))
            ys, xs = np.where(maxima)
            for sy, sx in zip(ys,xs):
                left, upper = int(sx+20), int(sy+top)
                x, y = left+ax, upper+ay; row = row_at(y,bands)
                if row is None:continue
                patch = gray[upper:upper+h,left:left+w]
                native = float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
                if native < cfg['native_context_min']:continue
                candidate = self._proposal(gray,index,x,y,w,scores[sy,sx],row,pose)
                candidate.update(template_bank='head_part_with_native_context',
                    head_part_rows=rows, native_context_similarity=round(native,5),
                    score_semantics='head-part NCC with separate native full-patch context gate; not probability')
                proposals.append(candidate)
        return proposals
