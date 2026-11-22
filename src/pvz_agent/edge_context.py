"""Untracked right-edge color/context evidence, never an accepted entity."""
import time
import cv2
import numpy as np
from .integral_scout import IntegralScoutPerception
from .vehicle_perception import VehicleObservationStream
from .water_pose_perception import water_fraction


def edge_presence_cues(rgb, settings):
    if rgb.shape != (600, 800, 3):
        raise ValueError('Canonical RGB screenshot required')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 170)) &
           (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 70))
    red[:, :settings['search_x_min']] = False
    red[:85] = False
    red[550:] = False
    count, _, stats, _ = cv2.connectedComponentsWithStats(red.astype('uint8'), 8)
    cues = []
    for x, y, width, height, area in stats[1:count]:
        x, y, width, height, area = map(int, (x, y, width, height, area))
        if not (10 <= width <= 80 and 15 <= height <= 85 and area >= 80 and
                x + width >= settings['flag_right_min']):
            continue
        bottom = y + height
        skin_box = [max(x, 776), max(0, bottom-5), 800, min(600, bottom+50)]
        duck_box = [max(0, x-8), min(600, bottom+25), 800, min(600, bottom+105)]
        water_box = [max(0, x-80), min(600, bottom+10), min(x-15, 760), min(600, bottom+105)]
        sx1, sy1, sx2, sy2 = skin_box
        part = rgb[sy1:sy2, sx1:sx2].astype('int16')
        if not part.size:
            continue
        skin = ((np.min(part, axis=2) >= 45) & (np.ptp(part, axis=2) <= 90) &
                (part[:, :, 1]-part[:, :, 0] >= 0) & (part[:, :, 1]-part[:, :, 0] <= 65) &
                (part[:, :, 1]-part[:, :, 2] >= 0) & (part[:, :, 1]-part[:, :, 2] <= 65))
        dx1, dy1, dx2, dy2 = duck_box
        part = hsv[dy1:dy2, dx1:dx2]
        if not part.size:
            continue
        yellow = ((part[:, :, 0] >= 20) & (part[:, :, 0] <= 38) &
                  (part[:, :, 1] >= 120) & (part[:, :, 2] >= 140))
        wx1, wy1, wx2, wy2 = water_box
        water = water_fraction(rgb[wy1:wy2, wx1:wx2])
        if not (int(skin.sum()) >= settings['skin_pixels_min'] and
                float(skin.mean()) >= settings['skin_fraction_min'] and
                int(yellow.sum()) >= settings['yellow_pixels_min'] and
                float(yellow.mean()) >= settings['yellow_fraction_min'] and
                water >= settings['water_fraction_min']):
            continue
        cues.append(dict(kind='uncertain_right_edge_water_context',
            visible_flag_box_xyxy=[x, y, x+width, bottom],
            box_semantics='visible red component only; not a head or body box',
            row=None, possible_rows=[3, 4],
            row_semantics='water-lane possibilities, not observed landing or feet',
            track_id=None, alive=None, health=None, type_identity=None, confidence=None,
            score_semantics='uncalibrated color and spatial cues, not probability',
            relationship_to_detected_heads='may overlap; not an additional entity count',
            fresh_observation=True, expires_after_frame=True, action_eligible=False,
            support=dict(skin_like=dict(roi_xyxy=skin_box, pixels=int(skin.sum()), fraction=float(skin.mean())),
                         yellow_like=dict(roi_xyxy=duck_box, pixels=int(yellow.sum()), fraction=float(yellow.mean())),
                         adjacent_water=dict(roi_xyxy=water_box, fraction=water)),
            limitations='Color coincidence possible; no confirmed zombie, flag identity, death, or empty-lane assurance.'))
    return cues


class EdgeContextPerception(IntegralScoutPerception):
    def observe(self, image):
        start = time.perf_counter()
        result = super().observe(image)
        result['uncertain_edge_presence'] = edge_presence_cues(
            np.asarray(image.convert('RGB')), self.config['edge_presence']) if result['scene']['value'] == 'playing' else []
        result['processing_ms'] = round((time.perf_counter()-start)*1000, 3)
        return result


class EdgeObservationStream(VehicleObservationStream):
    def observe(self, image, **kwargs):
        start = time.perf_counter()
        result = super().observe(image, **kwargs)
        for index, cue in enumerate(result['zombies']['uncertain_edge_presence'], 1):
            cue.update(context_id=f'{kwargs["state_id"]}:edge{index}', state_id=kwargs['state_id'],
                       capture_time=kwargs['capture_time'], evidence=dict(kwargs['evidence']))
        # This separate channel never enters either association tracker.
        result['processing_ms'] = round((time.perf_counter()-start)*1000, 3)
        return result
