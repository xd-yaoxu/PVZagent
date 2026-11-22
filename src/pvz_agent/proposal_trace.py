"""Offline proposal provenance. Probes are labels for diagnosis, never runtime input.

The frozen detector runs unchanged. Hooks copy the actual pre-NMS and water
proposals; the parallel trace records why those proposals did or did not survive.
Dense probe searches are explicitly counterfactual and do not add detections.
"""
import copy
import cv2
import numpy as np

from .boundary_perception import row_at
from .vehicle_color_perception_v18 import VehicleColorPerception


def point_key(candidate):
    return candidate['row'], tuple(candidate['head_xy'])


def nms_trace(proposals, radius):
    kept, records = [], []
    for index, candidate in sorted(enumerate(proposals), key=lambda p: p[1]['similarity'], reverse=True):
        blocker = next(((j, c) for j, c in kept
                        if np.linalg.norm(np.subtract(candidate['head_xy'], c['head_xy'])) < radius), None)
        record = dict(proposal_id=index, candidate=copy.deepcopy(candidate),
                      stage='nms_suppressed' if blocker else 'nms_passed')
        if blocker:
            record.update(blocker_proposal_id=blocker[0], blocker_head_xy=blocker[1]['head_xy'],
                          distance_pixels=float(np.linalg.norm(np.subtract(candidate['head_xy'], blocker[1]['head_xy']))))
        else:
            kept.append((index, candidate))
        records.append(record)
    return records


class TracedVehiclePerception(VehicleColorPerception):
    def _search(self, gray):
        proposals = super()._search(gray)
        self.raw_head_proposals = copy.deepcopy(proposals)
        return proposals

    def _water_proposals(self, rgb):
        proposals = super()._water_proposals(rgb)
        self.raw_water_proposals = copy.deepcopy(proposals)
        return proposals

    def observe(self, image):
        self.raw_head_proposals, self.raw_water_proposals = [], []
        return super().observe(image)

    def trace(self, image, probes):
        result = self.observe(image)
        records = nms_trace(self.raw_head_proposals, self.config['nms_radius_pixels'])
        rejected = {point_key(c): c for c in result['rejected_candidates']}
        final = {point_key(c): c for c in result['candidates']}
        for record in records:
            key = point_key(record['candidate'])
            if record['stage'] == 'nms_passed':
                if key in rejected:
                    record.update(stage='visual_veto', reason=rejected[key]['rejection_reason'])
                elif key in final:
                    record['stage'] = 'final_candidate'
                else:
                    record['stage'] = 'water_association_or_row_change'
        gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        cases = []
        for probe in probes:
            x, y = probe['head_xy']
            nearby = [r for r in records if r['candidate']['row'] == probe['row']
                      and abs(r['candidate']['head_xy'][0]-x) <= 25
                      and abs(r['candidate']['head_xy'][1]-y) <= 35]
            cases.append(dict(probe=probe, actual_nearby_proposals=nearby,
                              counterfactual_dense_search=self.dense_probe(gray, probe)))
        return result, dict(schema='pvz-head-proposal-trace/1', model_sha256=self.model_sha256,
            scene=result['scene'], raw_proposals=records, water_proposals=self.raw_water_proposals,
            probes=cases, input_sent=False,
            limits='Dense local searches use manual probe points for offline diagnosis only; no new candidates or runtime guarantees.')

    def dense_probe(self, gray, probe):
        """Native scores and actual coarse-refinement gates around a manual point."""
        x, y = probe['head_xy']; row = probe['row']; w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']; cfg = self.config['coarse_search']
        small = cv2.resize(gray, (400, 300), interpolation=cv2.INTER_AREA)
        dense, coarse_events, partial_events, centered = [], [], [], []
        for i, template in enumerate(self.model['templates']):
            pose, bands = self._bands(i); band = bands[row-1]
            if band is None:
                continue
            x1, x2 = max(20, x-25-ax), min(800-w, x+25-ax)
            y1, y2 = max(0, y-35-ay, band[0]-ay), min(600-h, y+35-ay, band[1]-ay)
            if x1 <= x2 and y1 <= y2:
                scores = cv2.matchTemplate(gray[y1:y2+h, x1:x2+w], template, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(scores)
                dense.append(dict(template_index=i, pose=pose, similarity=score,
                    head_xy=[x1+loc[0]+ax, y1+loc[1]+ay], threshold=self.config['match_threshold']))
                cx1, cx2 = max(x1,x-8-ax), min(x2,x+8-ax)
                if cx1 <= cx2:
                    local=cv2.matchTemplate(gray[y1:y2+h,cx1:cx2+w],template,cv2.TM_CCOEFF_NORMED)
                    _,value,_,where=cv2.minMaxLoc(local)
                    centered.append(dict(template_index=i,similarity=value,
                        head_xy=[cx1+where[0]+ax,y1+where[1]+ay],threshold=self.config['match_threshold']))
            for width,partial in self.edge_templates[i]:
                px=800-width+ax
                if abs(px-x)>25:continue
                py1=max(0,y-35-ay,band[0]-ay);py2=min(600-h,y+35-ay,band[1]-ay)
                if py1>py2:continue
                values=cv2.matchTemplate(gray[py1:py2+h,800-width:800],partial,cv2.TM_CCOEFF_NORMED)
                _,value,_,where=cv2.minMaxLoc(values)
                partial_events.append(dict(template_index=i,visible_width=width,similarity=value,
                    head_xy=[px,py1+where[1]+ay],threshold=self.config['right_edge_search']['threshold']))
            scores = cv2.matchTemplate(small[:, 10:400], self.small_templates[i], cv2.TM_CCOEFF_NORMED)
            maxima = (scores >= cfg['threshold']) & (scores >= cv2.dilate(scores, np.ones((5, 5), np.uint8)))
            ys, xs = np.where(maxima)
            for sy, sx in zip(ys, xs):
                left, top = int(sx*2+20), int(sy*2); radius = cfg['refine_radius']
                if abs(left+ax-x) > 25+radius or abs(top+ay-y) > 35+radius:
                    continue
                valid = [b for b in bands if b is not None]
                if not any(b[0]-radius <= top+ay <= b[1]+radius for b in valid):
                    continue
                rx1, ry1 = max(20, left-radius), max(0, top-radius)
                rx2, ry2 = min(800, left+w+radius), min(600, top+h+radius)
                refined = cv2.matchTemplate(gray[ry1:ry2, rx1:rx2], template, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(refined)
                px, py = rx1+loc[0]+ax, ry1+loc[1]+ay
                assigned = row_at(py, bands)
                coarse_events.append(dict(template_index=i, coarse_similarity=float(scores[sy, sx]),
                    head_xy=[px, py], native_similarity=score, assigned_row=assigned,
                    native_threshold=self.config['match_threshold'],
                    gate='passed' if assigned is not None and score >= self.config['match_threshold'] else 'native_score_or_band_rejected'))
        # These are summaries; actual above-threshold partial proposals are in raw_proposals.
        return dict(full_native_best=sorted(dense, key=lambda r:r['similarity'], reverse=True)[:8],
                    centered_native_best=sorted(centered,key=lambda r:r['similarity'],reverse=True)[:8],
                    visible_partial_best=sorted(partial_events,key=lambda r:r['similarity'],reverse=True)[:8],
                    coarse_refinements=sorted(coarse_events, key=lambda r:r['native_similarity'], reverse=True)[:16],
                    full_native_search_window=dict(x_tolerance=25,y_tolerance=35),
                    partial_score_maps_included=True,
                    partial_limits='Inherited full-bank visible strips only; native-edge and water proposals are separately preserved, not exhaustive below-threshold maps.')
