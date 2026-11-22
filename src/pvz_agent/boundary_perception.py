"""Read-only partial-window search and head-supported duplicate rejection.

Partial templates compare only pixels present in the screenshot. No padding,
off-screen pixels or inferred empty lanes are supplied to later stages.
"""
import time

import cv2
import numpy as np

from .crossgame_perception import CrossGamePerception


def row_at(y, bands):
    for row, band in enumerate(bands, 1):
        if band is not None and band[0] <= y <= band[1]:
            return row
    return None


def head_similarity(patch, template, rows):
    return float(cv2.matchTemplate(patch[:rows], template[:rows], cv2.TM_CCOEFF_NORMED)[0, 0])


class BoundaryPerception(CrossGamePerception):
    def _bands(self, index):
        pose = str(self.model['template_pose'][index])
        bands = self.config['submerged_head_bands_y'] if pose == 'submerged' else self.config['head_bands_y']
        return pose, bands

    def _proposal(self, gray, index, x, y, width, score, row, pose):
        w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']
        patch = gray[y-ay:y-ay+h, x-ax:x-ax+width]
        template = self.model['templates'][index, :, :width]
        truncated = width < w
        return {'row': row, 'head_xy': [x, y],
                'box_xyxy': [x-ax, y-ay, min(800, x-ax+w), y-ay+h],
                'similarity': round(float(score), 5), 'template_index': index, 'pose': pose,
                'head_similarity': round(head_similarity(patch, template, self.config['head_support_rows']), 5),
                'truncated': truncated, 'visible_template_fraction': round(width/w, 5),
                'visible_template_width': width,
                'position_semantics': ('partial_template_reference_with_unquantified_localization_uncertainty'
                                       if truncated else 'visible_head_reference_in_canonical_pixels_not_feet')}

    def _search(self, gray):
        w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']
        edge = self.config['right_edge_search']
        proposals = []
        for i, template in enumerate(self.model['templates']):
            pose, bands = self._bands(i)
            valid = [b for b in bands if b is not None]
            top = max(0, min(b[0] for b in valid)-ay)
            bottom = min(600, max(b[1] for b in valid)-ay+h)
            # One search per template, retaining the same allowed head bands.
            scores = cv2.matchTemplate(gray[top:bottom, 20:800], template, cv2.TM_CCOEFF_NORMED)
            maxima = ((scores >= self.config['match_threshold']) &
                      (scores >= cv2.dilate(scores, np.ones((9, 9), np.uint8))))
            ys, xs = np.where(maxima)
            for sy, sx in zip(ys, xs):
                x, y = int(sx+20+ax), int(sy+top+ay)
                row = row_at(y, bands)
                if row is not None:
                    proposals.append(self._proposal(gray, i, x, y, w, scores[sy, sx], row, pose))
            # Only the left portion of a full template is visible at the right edge.
            # The head reference itself remains inside the screenshot domain.
            if not edge['enabled']:
                continue
            for width in range(edge['min_visible_width'], w):
                x = 800-width+ax
                if not 0 <= x < 800:
                    continue
                partial = template[:, :width]
                if float(np.std(partial)) < edge['template_std_min']:
                    continue
                strip = gray[top:bottom, 800-width:800]
                scores = cv2.matchTemplate(strip, partial, cv2.TM_CCOEFF_NORMED)
                maxima = ((scores >= edge['threshold']) &
                          (scores >= cv2.dilate(scores, np.ones((9, 1), np.uint8))))
                ys = np.where(maxima[:, 0])[0]
                for sy in ys:
                    y = int(sy+top+ay)
                    row = row_at(y, bands)
                    if row is None:
                        continue
                    candidate = self._proposal(gray, i, x, y, width, scores[sy, 0], row, pose)
                    if candidate['head_similarity'] >= edge['head_support_min']:
                        proposals.append(candidate)
        return proposals

    def observe(self, image):
        start = time.perf_counter()
        result = self.scene_gate.observe(image)
        gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        proposals = self._search(gray) if result['scene']['value'] == 'playing' else []
        accepted = []
        radius = self.config['nms_radius_pixels']
        for candidate in sorted(proposals, key=lambda c: c['similarity'], reverse=True):
            if not any(np.linalg.norm(np.subtract(candidate['head_xy'], a['head_xy'])) < radius for a in accepted):
                accepted.append(candidate)
        w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']
        settings = self.config['negative_veto']
        retained, rejected = [], []
        for candidate in accepted:
            x, y = candidate['head_xy']
            left, top = x-ax, y-ay
            width = candidate['visible_template_width']
            radius = settings['search_radius_pixels']
            patch = gray[max(0, top-radius):min(600, top+h+radius),
                         max(0, left-radius):min(800, left+width+radius)]
            score = max(float(cv2.minMaxLoc(cv2.matchTemplate(patch, t[:, :width], cv2.TM_CCOEFF_NORMED))[1])
                        for t in self.model['negative_templates'])
            candidate['negative_similarity'] = round(score, 5)
            if score >= settings['threshold'] and score >= candidate['similarity']+settings['margin']:
                candidate['rejection_reason'] = 'resembles_labeled_detached_or_falling_head'
                rejected.append(candidate)
            else:
                retained.append(candidate)
        # A lower match on the chest must have weaker head evidence as well as
        # a stronger nearby upper head. Close real heads are not merged by radius alone.
        settings = self.config['vertical_duplicate']
        survivors = []
        for candidate in retained:
            x, y = candidate['head_xy']
            upper = next((a for a in retained if a is not candidate and a['row'] == candidate['row']
                          and abs(a['head_xy'][0]-x) <= settings['max_dx']
                          and settings['min_dy'] <= y-a['head_xy'][1] <= settings['max_dy']
                          and a['head_similarity'] >= candidate['head_similarity']+settings['head_margin']
                          and a['similarity'] >= candidate['similarity']+settings['score_margin']), None)
            if settings['enabled'] and upper is not None:
                candidate.update(rejection_reason='weaker_head_support_below_nearby_head',
                                 stronger_head_xy=upper['head_xy'])
                rejected.append(candidate)
            else:
                survivors.append(candidate)
        survivors.sort(key=lambda c: (c['row'], c['head_xy'][0], c['head_xy'][1]))
        for i, candidate in enumerate(survivors, 1):
            candidate.update(detection_id=f'z{i}', kind='zombie_candidate', type_identity=None,
                             alive=None, health=None, threat_level=None)
        result.update(candidates=survivors, rejected_candidates=rejected,
                      right_edge_coverage='partial_visible_templates_only; less-than-half and off-screen anchors remain uncovered',
                      rejection_semantics='Visual abstention only; no death or empty-lane confirmation.',
                      processing_ms=round((time.perf_counter()-start)*1000, 3))
        return result
