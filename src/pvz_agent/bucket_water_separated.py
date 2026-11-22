"""Separate thick helmet support from connected thin pool trim before conjunction."""
import time
import cv2
import numpy as np
from .edge_flag_context import FlagEdgeContextPerception
from .water_head_support import duck_support


def separated_bucket_contexts(rgb, config):
    if rgb.shape != (600, 800, 3):
        raise ValueError('Canonical RGB screenshot required')
    cfg = config['bucket_water_context']
    shape = dict(config['occlusion_shape']['bucket'], **cfg['shape'])
    mask = (np.min(rgb, axis=2) >= 95) & (np.ptp(rgb.astype('int16'), axis=2) <= 35)
    x1,y1,x2,y2 = cfg['search_xyxy']
    bounded = np.zeros((600,800), dtype='uint8')
    bounded[y1:y2,x1:x2] = mask[y1:y2,x1:x2]
    # Remove thin connections to the pool trim; retain thick neutral support.
    bounded = cv2.morphologyEx(bounded, cv2.MORPH_OPEN, np.ones(tuple(cfg['opening_hw']), dtype='uint8'))
    count, _, stats, centers = cv2.connectedComponentsWithStats(bounded, 8)
    proposals = []; checks = []
    for index in range(1, count):
        x,y,width,height,area = map(int, stats[index])
        if not (area >= shape['area_min'] and shape['width_min'] <= width <= shape['width_max'] and
                shape['height_min'] <= height <= shape['height_max']):
            continue
        # Clipped components cannot establish a complete helmet outline.
        if x <= x1 or y <= y1 or x+width >= x2 or y+height >= y2:
            continue
        hx = int(round(float(centers[index,0]) + cfg['reference_dx']))
        hy = y + height + cfg['reference_below_component']
        rows = [row for row, ref in zip((3,4), cfg['floating_reference_y'])
                if abs(hy-ref) <= cfg['row_tolerance']]
        check = dict(helmet_box_xyxy=[x,y,x+width,y+height], area=area, head_xy=[hx,hy],
            possible_rows=rows, accepted=False)
        checks.append(check)
        if len(rows) != 1:
            check['reason'] = 'floating_row_unconfirmed'; continue
        dx1,dy1,dx2,dy2 = cfg['face_roi_offset']
        fx1,fy1,fx2,fy2 = hx+dx1,hy+dy1,hx+dx2,hy+dy2
        if not (0 <= fx1 < fx2 <= 800 and 0 <= fy1 < fy2 <= 600):
            check['reason'] = 'face_out_of_bounds'; continue
        face = rgb[fy1:fy2,fx1:fx2].astype('int16')
        r,g,b = face[:,:,0], face[:,:,1], face[:,:,2]
        skin = ((np.min(face,axis=2) >= 45) & (np.ptp(face,axis=2) <= 90) &
            (g-r >= 0) & (g-r <= 65) & (g-b >= 0) & (g-b <= 65))
        pixels = int(skin.sum()); fraction = float(skin.mean())
        check.update(face_roi_xyxy=[fx1,fy1,fx2,fy2], skin_like_pixels=pixels, skin_like_fraction=fraction)
        if pixels < cfg['skin_pixels_min'] or fraction < cfg['skin_fraction_min']:
            check['reason'] = 'visible_face_support_rejected'; continue
        support = duck_support(rgb,hx,hy,config['water_head_support'])
        check['duck_water_support'] = support
        if not support['accepted']:
            check['reason'] = 'duck_or_water_support_rejected'; continue
        check.update(accepted=True, reason='conjunctive_visual_support')
        proposals.append(dict(row=rows[0], head_xy=[hx,hy],
            box_xyxy=[x,min(y,fy1),max(x+width,fx2),max(y+height,fy2)],
            kind='zombie_candidate', template_bank='floating_helmet_face_duck_water_geometry',
            similarity=None, score_semantics='Conjunctive color/shape gates; no NCC score, probability or confirmed bucket identity.',
            row_evidence='floating_helmet_face_duck_and_water_context',
            row_evidence_limits='Fixed screenshot floating-pose calibration; airborne/submerged poses and actual feet are not inferred.',
            position_semantics='component_relative_visible_face_reference_not_precise_head_center',
            hat_hypothesis='bucket', helmet_support=check, truncated=False,
            type_identity=None, alive=None, health=None, threat_level=None,
            empty_lane_verified=False, action_eligible=False))
    return proposals, checks


class SeparatedBucketWaterPerception(FlagEdgeContextPerception):
    def observe(self, image):
        start = time.perf_counter()
        result = super().observe(image)
        proposals, checks = separated_bucket_contexts(np.asarray(image.convert('RGB')), self.config) if result['scene']['value']=='playing' else ([], [])
        kept = list(result['candidates'])
        for candidate in proposals:
            # Existing observations retain their reference, row and confidence semantics.
            # Nearby geometry is overlap evidence, never a second counted entity.
            if any(np.linalg.norm(np.subtract(candidate['head_xy'], old['head_xy'])) <
                   self.config['water_pose']['cluster_radius'] for old in kept):
                continue
            kept.append(candidate)
        kept.sort(key=lambda c:(c['row'],c['head_xy'][0],c['head_xy'][1]))
        for i, candidate in enumerate(kept, 1):
            candidate['detection_id'] = f'z{i}'
        result.update(candidates=kept, bucket_context_checks=checks,
            processing_ms=round((time.perf_counter()-start)*1000,3))
        return result
