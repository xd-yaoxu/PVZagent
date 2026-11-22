"""Offline-only supplemental partial-head evidence; never mutates v40 output."""
import copy
import cv2
import numpy as np


def _eye_components(rgb, x, y, cfg):
    left, top = max(0, x - 35), max(0, y - 25)
    right, bottom = min(800, x + 55), min(600, y + 25)
    roi = rgb[top:bottom, left:right].astype('int16')
    mask = ((roi.min(2) >= cfg['eye_rgb_min']) & (np.ptp(roi, axis=2) <= cfg['eye_channel_spread_max'])).astype('uint8')
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    lo, hi = cfg['eye_component_area']; wlo, whi = cfg['eye_component_width_height']
    amin, amax = cfg['eye_aspect_range']; result = []
    for stat, center in zip(stats[1:], centers[1:]):
        sx, sy, w, h, area = map(int, stat)
        if lo <= area <= hi and wlo <= w <= whi and wlo <= h <= whi and amin <= w / h <= amax:
            result.append({'center_xy': [float(center[0] + left), float(center[1] + top)],
                           'box_xyxy': [sx+left, sy+top, sx+left+w, sy+top+h], 'area': area})
    return result


def _contour_rows(rgb, eye, cfg):
    ex, ey = round(eye[0]), round(eye[1]); rows = []
    for y in range(max(0, ey+cfg['contour_y_offset'][0]), min(600, ey+cfg['contour_y_offset'][1]+1)):
        for x in range(max(0, ex+cfg['contour_x_offset'][0]), min(792, ex+cfg['contour_x_offset'][1]+1)):
            def scalp(pixel):
                return int(pixel.min()) >= cfg['scalp_min_rgb'] and int(np.ptp(pixel.astype('int16'))) <= cfg['scalp_max_channel_spread']
            if not (scalp(rgb[y,x]) and scalp(rgb[y,x+1])):
                continue
            for dark_length in range(cfg['dark_run_length'][0], cfg['dark_run_length'][1]+1):
                dark = rgb[y,x+2:x+2+dark_length]
                if len(dark) != dark_length or np.any(dark.max(axis=1) >= cfg['dark_max_rgb_exclusive']):
                    continue
                for gap in range(cfg['grass_max_gap_after_dark']+1):
                    pos = x+2+dark_length+gap
                    if pos >= 800: continue
                    r,g,b = map(int,rgb[y,pos])
                    if g >= cfg['grass_g_min'] and g-r >= cfg['grass_g_minus_r_min'] and g-b >= cfg['grass_g_minus_b_min']:
                        rows.append(y); break
                if rows and rows[-1] == y: break
            if rows and rows[-1] == y: break
    return rows


def _rejection_reason(parent, candidate_xy, eye, parent_eyes, negative, parents, bands, cfg):
    px,py=parent['head_xy']; x,y=candidate_xy; ex,ey=eye['center_xy']; row=parent['row']
    if row is None or not (1 <= row <= len(bands)) or bands[row-1] is None:
        return 'unknown_row'
    low,high=bands[row-1]
    if not (low <= y <= high and low <= ey <= high): return 'different_row_band'
    if not (16 <= x <= 784 and 16 <= y <= 568): return 'template_out_of_bounds'
    if not (cfg['parent_dx_inclusive'][0] <= x-px <= cfg['parent_dx_inclusive'][1]) or abs(y-py)>cfg['parent_abs_dy_max']:
        return 'outside_parent_geometry'
    if len(parent_eyes)<cfg['parent_eye_count_min']: return 'too_few_parent_eyes'
    if ex-px<cfg['extra_eye_min_dx'] or any(abs(ex-e['center_xy'][0])<cfg['extra_eye_parent_eye_min_abs_dx'] for e in parent_eyes):
        return 'extra_eye_not_distinct'
    if abs(ex-x)>cfg['extra_eye_abs_dx_from_partial_max'] or abs(ey-y)>cfg['extra_eye_abs_dy_from_partial_max']:
        return 'extra_eye_outside_partial'
    if any(p['row']==row and np.linalg.norm(np.subtract([ex,ey],p['head_xy'])) < cfg['dedupe_partial_dx_dy_euclidean_less_than'] for p in parents):
        return 'duplicate_of_v40_candidate'
    if negative >= cfg['negative_template_absolute_veto']: return 'negative_template_veto'
    return None


def supplemental_evidence(image, state, model, cfg, bands):
    """Return local supplemental evidence, not normal heads or input permission."""
    if image.size != (800,600): raise ValueError('Canonical image required')
    if state['scene']['value'] != 'playing': return []
    rgb = np.asarray(image.convert('RGB')); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    before = copy.deepcopy(state['zombies']['candidates']); proposals = []
    for parent_index,parent in enumerate(before):
        px,py = parent['head_xy']; row = parent['row']
        if row is None: continue
        eyes = _eye_components(rgb,px,py,cfg)
        parent_eyes = [e for e in eyes if cfg['parent_eye_dx'][0] <= e['center_xy'][0]-px <= cfg['parent_eye_dx'][1]
                       and abs(e['center_xy'][1]-py) <= cfg['parent_eye_abs_dy_max']]
        if len(parent_eyes) < cfg['parent_eye_count_min']: continue
        extra = [e for e in eyes if e not in parent_eyes and e['center_xy'][0]-px >= cfg['extra_eye_min_dx']
                 and all(abs(e['center_xy'][0]-p['center_xy'][0]) >= cfg['extra_eye_parent_eye_min_abs_dx'] for p in parent_eyes)]
        if not extra: continue
        x1,x2 = max(16,px+cfg['parent_dx_inclusive'][0]),min(784,px+cfg['parent_dx_inclusive'][1])
        y1,y2 = max(16,py-cfg['parent_abs_dy_max']),min(568,py+cfg['parent_abs_dy_max'])
        if x1>x2 or y1>y2: continue
        a,b,c,d = cfg['part_xyxy']; search = gray[y1-16+b:y2-16+d,x1-16+a:x2-16+c]
        local = []
        for index, template in enumerate(model['templates']):
            if str(model['template_pose'][index]) != cfg['template_pose']: continue
            scores = cv2.matchTemplate(search,template[b:d,a:c],cv2.TM_CCOEFF_NORMED)
            for sy,sx in zip(*np.where(scores >= cfg['score_min'])):
                x,y=x1+int(sx),y1+int(sy)
                for eye in extra:
                    ex,ey=eye['center_xy']
                    if abs(ex-x)>cfg['extra_eye_abs_dx_from_partial_max'] or abs(ey-y)>cfg['extra_eye_abs_dy_from_partial_max']: continue
                    contour=_contour_rows(rgb,eye['center_xy'],cfg)
                    if len(contour)<cfg['contour_rows_min']: continue
                    radius=cfg['negative_search_radius']; left,top=x-16,y-16
                    patch=gray[max(0,top-radius):min(600,top+48+radius),max(0,left-radius):min(800,left+32+radius)]
                    negative=max(float(cv2.minMaxLoc(cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED))[1]) for t in model['negative_templates'])
                    if _rejection_reason(parent,[x,y],eye,parent_eyes,negative,before,bands,cfg): continue
                    local.append({'row':row,'visible_reference_xy':[round(ex),round(ey)],'template_partial_reference_xy':[x,y],
                        'parent_index':parent_index,'parent_head_xy':[px,py], 'partial_template_index':index,
                        'partial_similarity':float(scores[sy,sx]),'negative_template_max':negative,
                        'parent_eye_components':parent_eyes,'extra_eye_component':eye,'outer_contour_rows':contour,
                        'semantics':'supplemental_uncertain_partial_visible_eye_not_full_head_center',
                        'alive':None,'health':None,'action_eligible':False})
        local.sort(key=lambda v: v['partial_similarity'], reverse=True)
        for candidate in local:
            if any(np.linalg.norm(np.subtract(candidate['visible_reference_xy'], v['visible_reference_xy'])) < cfg['dedupe_partial_dx_dy_euclidean_less_than'] for v in proposals): continue
            proposals.append(candidate)
    if before != state['zombies']['candidates']: raise RuntimeError('Parent candidate mutation')
    return proposals
