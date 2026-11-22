"""Coarse search followed by original-pixel verification, plus visible edge strips."""
import cv2
import numpy as np

from .boundary_perception import BoundaryPerception, row_at


class CoarseBoundaryPerception(BoundaryPerception):
    def __init__(self, model_file, config):
        super().__init__(model_file, config)
        w, h = config['patch_wh']
        ax, _ = config['anchor_xy']
        if config['coarse_search']['scale'] != .5 or w % 2 or h % 2:
            raise ValueError('Half-resolution coarse search needs even template dimensions')
        widths = config['right_edge_search']['visible_widths']
        if not widths or len(widths) != len(set(widths)) or any(type(v) is not int or not w-ax < v < w for v in widths):
            raise ValueError('Partial search must retain its anchor inside the screenshot')
        self.small_templates = [cv2.resize(t, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
                                for t in self.model['templates']]
        self.edge_templates = {}
        for i, template in enumerate(self.model['templates']):
            self.edge_templates[i] = [(width, template[:, :width].copy())
                                     for width in config['right_edge_search']['visible_widths']
                                     if float(np.std(template[:, :width])) >= config['right_edge_search']['template_std_min']]

    def _search(self, gray):
        w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']
        coarse = self.config['coarse_search']
        edge = self.config['right_edge_search']
        small = cv2.resize(gray, (400, 300), interpolation=cv2.INTER_AREA)
        proposals = []
        for i, template in enumerate(self.model['templates']):
            pose, bands = self._bands(i)
            valid = [b for b in bands if b is not None]
            scores = cv2.matchTemplate(small[:, 10:400], self.small_templates[i], cv2.TM_CCOEFF_NORMED)
            maxima = ((scores >= coarse['threshold']) &
                      (scores >= cv2.dilate(scores, np.ones((5, 5), np.uint8))))
            ys, xs = np.where(maxima)
            radius = coarse['refine_radius']
            for sy, sx in zip(ys, xs):
                left, top = int(sx*2+20), int(sy*2)
                if not any(b[0]-radius <= top+ay <= b[1]+radius for b in valid):
                    continue
                x1, y1 = max(20, left-radius), max(0, top-radius)
                x2, y2 = min(800, left+w+radius), min(600, top+h+radius)
                refined = cv2.matchTemplate(gray[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(refined)
                x, y = x1+loc[0]+ax, y1+loc[1]+ay
                row = row_at(y, bands)
                if row is not None and score >= self.config['match_threshold']:
                    proposals.append(self._proposal(gray, i, x, y, w, score, row, pose))
            if not edge['enabled']:
                continue
            top = max(0, min(b[0] for b in valid)-ay)
            bottom = min(600, max(b[1] for b in valid)-ay+h)
            for width, partial in self.edge_templates[i]:
                x = 800-width+ax
                scores = cv2.matchTemplate(gray[top:bottom, 800-width:800], partial, cv2.TM_CCOEFF_NORMED)
                maxima = ((scores >= edge['threshold']) &
                          (scores >= cv2.dilate(scores, np.ones((9, 1), np.uint8))))
                for sy in np.where(maxima[:, 0])[0]:
                    y = int(sy+top+ay)
                    row = row_at(y, bands)
                    if row is None:
                        continue
                    candidate = self._proposal(gray, i, x, y, width, scores[sy, 0], row, pose)
                    if candidate['head_similarity'] >= edge['head_support_min']:
                        proposals.append(candidate)
        return proposals

    def observe(self, image):
        result = super().observe(image)
        for i, rejected in enumerate(result['rejected_candidates'], 1):
            rejected.update(rejection_id=f'r{i}', kind='zombie_candidate', alive=None,
                            type_identity=None, health=None, threat_level=None)
        return result
