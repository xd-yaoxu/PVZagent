"""Identify plants from occupied cells in a canonical 800×600 BGR frame."""
from __future__ import annotations

from collections import Counter, deque
from math import isfinite
from pathlib import Path
from time import monotonic, perf_counter

import cv2
import numpy as np

from .zombie_intel import WATER_ROWS, is_plant_legal_for_row

PLANT_NAMES_ZH = {
    "sunflower": "向日葵", "peashooter": "豌豆", "repeater": "双发",
    "wallnut": "坚果", "spikeweed": "地刺", "potato_mine": "土豆雷",
    "tangle_kelp": "海草", "squash": "窝瓜", "lily_pad": "睡莲",
    "unknown_plant": "未知植物",
}
SHOOTER_DPS = {"peashooter": 13.5, "repeater": 27.0}
_OCCUPIED = ("ground_occupied", "base_only", "base_and_upper")
_FEET_Y = (159, 241, 333, 418, 500, 583)
_MEMORY_SECONDS = 1.0
_REFERENCE = Path("evidence/day06/new-material-a-20261107-v1/canonical/frame-0000.png")


def compact_plants(counts):
    return "+".join(f"{PLANT_NAMES_ZH[kind]}x{counts[kind]}" for kind in PLANT_NAMES_ZH if counts.get(kind))


def lane_entities(intel):
    """Yield (row, entity) from classified lanes or the frozen head candidates."""
    if isinstance(intel, dict):
        if "by_lane" in intel:
            for row, lane in intel["by_lane"].items():
                for entity in lane.get("entities", ()):
                    yield int(row), entity
            return
        intel = intel.get("candidates", intel.get("entities", intel.get("zombies", ())))
        if isinstance(intel, dict):
            yield from lane_entities(intel)
            return
    for entity in intel or ():
        row = entity.get("row")
        if row in range(1, 7):
            yield int(row), entity


def entity_x(entity):
    point = entity.get("head_xy") or entity.get("center_xy") or entity.get("vehicle_center_xy")
    return entity.get("x_px", entity.get("x", point[0] if point else None))


def _components(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return stats[1:]


def _largest(mask):
    stats = _components(mask)
    return tuple(map(int, max(stats, key=lambda box: box[4]))) if len(stats) else (0, 0, 0, 0, 0)


def _shooter(hsv, water):
    # Six canonical A frames: 49–56 px lime head, 27–34 px high.
    # The rear crest measured 83–114 dark pixels on the repeater vs 22–39 on peas.
    end = 40 if water else 47
    lime = cv2.inRange(hsv[4:end], (33, 103, 119), (55, 255, 255))
    x, y, w, h, area = _largest(lime)
    y += 4
    if not (39 <= w <= 61 and 22 <= h <= 37 and 680 <= area <= 1460 and 12 <= x <= 34):
        return None
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 98))
    right = x + w
    muzzle = dark[max(3, y - 4):y + h + 4, right - 12:right + 5]
    eye = dark[y + 3:y + h - 3, x + w // 3:x + 2 * w // 3]
    if not any(2 <= bw <= 8 and 12 <= bh <= 23 and a >= 36
               for _, _, bw, bh, a in _components(muzzle)):
        return None
    if not any(8 <= a <= 45 and bh >= 4 for _, _, _, bh, a in _components(eye)):
        return None
    # Water pads may merge with the foot leaves. On land require the separate
    # near-ground leaf cluster so a lime projectile cannot stand in for a plant.
    if not water:
        _, _, fw, fh, fa = _largest(cv2.inRange(hsv[50:], (33, 85, 75), (58, 255, 255)))
        if fw < 27 or fh < 9 or fa < 155:
            return None
    crest = dark[max(3, y - 5):y + h, max(0, x - 17):x + 2]
    _, _, cw, ch, ca = _largest(crest)
    if ca >= 78 and cw >= 12 and ch >= 17:
        return "repeater", 0.91
    return "peashooter", 0.88


def _kelp(hsv):
    red = cv2.inRange(hsv[48:, 12:66], (0, 135, 120), (12, 255, 255))
    if cv2.countNonZero(red) < 35:
        return False
    dark = cv2.inRange(hsv[22:], (0, 0, 0), (179, 255, 83))
    _, _, dw, dh, da = _largest(dark)
    return dw >= 44 and dh >= 27 and da >= 680


def _land_trap(hsv, amber, *, rescue=False):
    floor_amber = amber[38:]
    if cv2.countNonZero(floor_amber) < (480 if rescue else 210):
        return None
    ax, ay, aw, ah, aa = _largest(floor_amber)
    if ay >= 1 and 40 <= aw <= 74 and 13 <= ah <= 35 and 480 <= aa <= 1550:
        button = hsv[max(28, ay + 17):ay + 40, 23:57]
        red = cv2.bitwise_or(cv2.inRange(button, (161, 111, 157), (179, 255, 255)),
                             cv2.inRange(button, (0, 111, 157), (9, 255, 255)))
        armed = any(48 <= a <= 300 and 7 <= bw <= 23 and 6 <= bh <= 23
                    for _, _, bw, bh, a in _components(red))
        if armed:
            return "potato_mine", 0.89, 0, True
        if not rescue:
            gray = cv2.inRange(button, (0, 0, 116), (179, 74, 251))
            if any(32 <= a <= 240 and 7 <= bw <= 19 and 5 <= bh <= 19
                   for _, _, bw, bh, a in _components(gray)):
                return "potato_mine", 0.89, 0, False
    if cv2.countNonZero(amber[48:]) >= 575:
        sx, sy, sw, sh, sa = _largest(amber[48:])
        if sw >= 59 and 14 <= sh <= 28 and sa >= 575 and sy >= 3:
            tips = amber[48 + sy:48 + sy + sh // 2, sx:sx + sw]
            if sum(a >= 10 and bh >= 4 for _, _, _, bh, a in _components(tips)) >= 3:
                if not rescue or cv2.countNonZero(cv2.inRange(
                    hsv[63:], (0, 0, 0), (179, 255, 95),
                )) >= 190:
                    return "spikeweed", 0.88, 0, False
    if not rescue and ay >= 22 and 22 <= aw <= 52 and 8 <= ah <= 19 and 210 <= aa <= 620:
        eye_roi = hsv[38 + ay:38 + ay + ah, ax + 6:ax + aw - 6]
        eyes = cv2.inRange(eye_roi, (0, 0, 0), (179, 255, 83))
        if sum(5 <= a <= 48 and 2 <= bh <= 9 for _, _, _, bh, a in _components(eyes)) >= 2:
            return "potato_mine", 0.80, 0, False
    return None


def _identify(roi, *, row, occupancy):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    water = row in WATER_ROWS
    if water and occupancy == "ground_occupied":
        if _kelp(hsv):
            return "tangle_kelp", 0.90, 0, False
        return "unknown_plant", 0.0, 0, False

    amber = cv2.inRange(hsv, (5, 72, 63), (33, 255, 255))
    x, y, w, h, area = _largest(amber) if cv2.countNonZero(amber) >= 850 else (0, 0, 0, 0, 0)
    # Sunflower face (+ petals) measures 1,800–1,900 amber pixels before glare.
    # Amber in the head band vetoes a shooter even when the flower is obscured.
    amber_head = cv2.countNonZero(amber[4:49, 9:69])
    if w >= 39 and 30 <= h <= 54 and y <= 23 and area >= 850 and amber_head >= 600:
        return "sunflower", 0.88, 0, False
    if 29 <= w <= 68 and 38 <= h <= 73 and 1000 <= area <= 3400 and y + h >= 62:
        white = cv2.inRange(hsv[y + 10:y + h // 2 + 6, x:x + w], (0, 0, 175), (179, 60, 255))
        eyes = sum(35 <= a <= 190 and 6 <= bw <= 16 and 6 <= bh <= 16
                   for _, _, bw, bh, a in _components(white))
        if eyes:
            if eyes >= 2 and w >= 52 and h >= 62 and area >= 2500:
                hp = 3600
            elif min(w / 56, h / 67) <= 0.75:
                hp = 900
            elif min(w / 56, h / 67) <= 0.91:
                hp = 2200
            else:
                hp = 2400
            return "wallnut", 0.90, hp, False

    if not water:
        trap = _land_trap(hsv, amber)
        if trap:
            return trap

    # Squash's tall, pale olive body has very little saturation in its highlights.
    pale = cv2.inRange(hsv[4:74, 5:73], (23, 30, 115), (52, 125, 255))
    px, py, pw, ph, pa = _largest(pale)
    if 39 <= pw <= 66 and ph >= 51 and pa >= 1850 and py <= 12:
        face = cv2.inRange(
            hsv[py + 17:py + 39, px + 22:px + pw - 3], (0, 0, 0), (179, 255, 103),
        )
        if any(a >= 14 and bh >= 4 for _, _, _, bh, a in _components(face)):
            return "squash", 0.88, 0, False

    if amber_head < 145:
        shooter = _shooter(hsv, water)
        if shooter:
            return *shooter, 0, False
    return "unknown_plant", 0.0, 0, False


def _cell_roi(frame, row, x):
    foot = _FEET_Y[row - 1]
    return frame[foot - 78:foot + 2, x - 39:x + 39]


def _trap_color_pixels(roi, water):
    # Quick BGR test on the floor before HSV or connected-component work.
    if water:
        return cv2.inRange(roi[32:78], (0, 0, 0), (96, 96, 96)), 430
    return cv2.inRange(roi[48:78], (0, 0, 104), (150, 220, 255)), 210


def _trap_reference_rois(reference):
    # The actual canonical reference includes plants. Where the floor color test
    # finds foreground in the reference, borrow the cleanest strip in that lane.
    rois = {}
    for row in range(1, 7):
        strips = [_cell_roi(reference, row, 80 * col) for col in range(1, 10)]
        masks = [_trap_color_pixels(strip, row in WATER_ROWS) for strip in strips]
        counts = [cv2.countNonZero(mask) for mask, _ in masks]
        clear = strips[counts.index(min(counts))]
        for col, (strip, count, (_, threshold)) in enumerate(zip(strips, counts, masks), start=1):
            rois[row, col] = clear if count >= threshold else strip
    return rois


def _rescue_trap(roi, reference_roi, water):
    color, threshold = _trap_color_pixels(roi, water)
    if cv2.countNonZero(color) < threshold:
        return None
    floor = slice(32 if water else 48, 78)
    diff = cv2.absdiff(roi[floor], reference_roi[floor])
    same = cv2.inRange(diff, (0, 0, 0), (28, 28, 28))
    changed = cv2.bitwise_and(color, cv2.bitwise_not(same))
    if cv2.countNonZero(changed) < threshold:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if water:
        yellow_green = cv2.inRange(hsv[30:], (17, 90, 130), (62, 255, 255))
        tendrils = cv2.inRange(hsv[25:77], (23, 105, 26), (88, 255, 132))
        if cv2.countNonZero(yellow_green) < 800 and cv2.countNonZero(tendrils) >= 650 and _kelp(hsv):
            return "tangle_kelp", 0.90, 0, False
        return None
    amber = cv2.inRange(hsv, (5, 72, 63), (33, 255, 255))
    if cv2.countNonZero(amber[10:48, 8:70]) >= 420:
        return None
    return _land_trap(hsv, amber, rescue=True)


def _plant_cell(row, column, occupancy, identification):
    kind, confidence, hp, armed = identification
    if not is_plant_legal_for_row(kind, row):
        kind, confidence, hp, armed = "unknown_plant", 0.0, 0, False
    return {
        "row": row, "column": column, "x_px": 80 * column, "plant_id": kind,
        "occupancy": occupancy, "confidence": confidence, "dps": SHOOTER_DPS.get(kind, 0.0),
        "blocker_hp": hp, "armed": armed,
        "carrier": "lily_pad" if row in WATER_ROWS and occupancy == "base_and_upper" else None,
    }


class PlantBoardClassifier:
    """Stateful board reader. Reset at a match boundary to discard recent votes."""

    def __init__(self, root=None):
        path = Path(root) / _REFERENCE if root is not None else Path(__file__).resolve().parents[3] / _REFERENCE
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        reference = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        self._reference_rois = _trap_reference_rois(reference) if (
            reference is not None and reference.shape == (600, 800, 3)
        ) else {}
        self.reset()

    def reset(self):
        self._history = {}
        self._last_time = None

    def _classify_cell(self, frame, row, x, column, occupancy, overlap, now):
        key = row, column
        past_occupancy, votes = self._history.get(key, (None, None))
        if occupancy != past_occupancy:
            votes = deque(maxlen=3)
        while votes and now - votes[0][0] > _MEMORY_SECONDS:
            votes.popleft()
        kind, confidence, hp, armed = "unknown_plant", 0.0, 0, False
        if occupancy == "base_only" and row in WATER_ROWS:
            kind, confidence, hp = "lily_pad", 1.0, 300
        elif not overlap and frame is not None and occupancy in _OCCUPIED:
            kind, confidence, hp, armed = _identify(_cell_roi(frame, row, x), row=row, occupancy=occupancy)
        if kind == "unknown_plant":
            if (overlap and len(votes) >= 2 and votes[-1][1] == votes[-2][1]
                    and votes[-1][1] in (*SHOOTER_DPS, "sunflower", "wallnut")):
                # Keep only recently confirmed identity. Occlusion can hide
                # bite damage, so that uses a conservative bound.
                kind, confidence = votes[-1][1], 0.55
                if kind == "wallnut":
                    hp = min(900, votes[-1][2], votes[-2][2])
            elif not overlap:
                votes.clear()
        else:
            if votes and ((kind in SHOOTER_DPS) != (votes[-1][1] in SHOOTER_DPS)):
                votes.clear()
            votes.append((now, kind, hp))
            # A single crest vote never upgrades a recent run of single peas.
            if kind == "repeater" and sum(vote[1] == "peashooter" for vote in votes) >= 2:
                kind, confidence = "peashooter", 0.80
        self._history[key] = occupancy, votes
        return _plant_cell(row, column, occupancy, (kind, confidence, hp, armed))

    def classify_board(self, frame_bgr, fields, zombie_intel=None, timestamp_s=None) -> dict:
        started = perf_counter()
        now = monotonic() if timestamp_s is None else timestamp_s
        if self._last_time is not None and now < self._last_time:
            self.reset()
        self._last_time = now
        frame = frame_bgr if (
            isinstance(frame_bgr, np.ndarray) and frame_bgr.shape == (600, 800, 3)
            and frame_bgr.dtype == np.uint8
        ) else None
        zombie_x = {row: [] for row in range(1, 7)}
        for row, z in lane_entities(zombie_intel):
            x = entity_x(z)
            if row in zombie_x and isinstance(x, (int, float)) and isfinite(x):
                point = z.get("head_xy") or z.get("center_xy") or z.get("vehicle_center_xy")
                foot = _FEET_Y[row - 1]
                if not point or foot - 165 <= point[1] <= foot + 8:
                    zombie_x[row].append((x, z.get("species_id") == "zomboni"))

        cells_by_row = {row: [] for row in range(1, 7)}
        seen = set()
        for cell in fields.get("cells", ()) or ():
            row, column = cell.get("row"), cell.get("column")
            if row not in cells_by_row or column not in range(1, 10):
                continue
            occupancy = cell.get("occupancy")
            if (row, column) in seen:
                continue
            seen.add((row, column))
            if row not in WATER_ROWS:
                if occupancy == "base_only":
                    self._history.pop((row, column), None)
                    continue
                if occupancy == "base_and_upper":
                    occupancy = "ground_occupied"
            x = 80 * column
            overlap = any((-130 if vehicle else -60) <= zx - x <= (53 if vehicle else 35)
                          for zx, vehicle in zombie_x[row])
            if occupancy in _OCCUPIED or (occupancy != "empty" and cell.get("occupied") is True):
                cells_by_row[row].append(self._classify_cell(frame, row, x, column, occupancy, overlap, now))
            else:
                self._history.pop((row, column), None)
                # The frozen reader also emits unknown for unobscured kelp and mines.
                reference = self._reference_rois.get((row, column))
                if occupancy in ("empty", "unknown") and not overlap and frame is not None and reference is not None:
                    trap = _rescue_trap(_cell_roi(frame, row, x), reference, row in WATER_ROWS)
                    if trap and is_plant_legal_for_row(trap[0], row):
                        rescued = _plant_cell(row, column, "ground_occupied", trap)
                        rescued["source"] = "low_profile_scan"
                        cells_by_row[row].append(rescued)
        for key in self._history.keys() - seen:
            del self._history[key]

        row_summary = {}
        for row, plants in cells_by_row.items():
            counts = Counter(p["plant_id"] for p in plants)
            row_summary[row] = {
                "counts": dict(counts),
                "total_dps": sum(p["dps"] for p in plants),
                "blocker_hp": sum(p["blocker_hp"] for p in plants),
                "spikeweed_count": counts["spikeweed"],
                "instant_kills": counts["tangle_kelp"] + counts["squash"] + sum(
                    p["plant_id"] == "potato_mine" and p["armed"] for p in plants
                ),
                "empty_lily_pads": counts["lily_pad"],
                "unknown_occupied": counts["unknown_plant"],
                "compact_zh": compact_plants(counts),
            }
        return {
            "cells_by_row": cells_by_row, "row_summary": row_summary,
            "elapsed_ms": (perf_counter() - started) * 1000,
        }
