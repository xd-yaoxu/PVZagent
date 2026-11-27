"""Classify localized PVZ threats and plan counters from read-only frame evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ZombieProfile:
    species_id: str
    name_cn: str
    hp_body: int
    hp_armor: int
    hp_total: int
    hp_tier: str
    threat_weight: float
    traits: tuple[str, ...]
    land_counters: tuple[str, ...]
    water_counters: tuple[str, ...]
    counter_hint: str
    cv_supported: bool


ZOMBIE_SPECIES_DB: dict[str, ZombieProfile] = {
    profile.species_id: profile
    for profile in (
        ZombieProfile(
            "zomboni", "冰车僵尸", 1350, 0, 1350, "高甲载具", 4.5,
            ("碾压植物", "留冰道禁种", "免疫减速"), ("spikeweed", "squash", "cherry_bomb"), (),
            "优先地刺爆胎秒杀，或窝瓜/樱桃炸弹紧急拦截", True,
        ),
        ZombieProfile(
            "buckethead_zombie", "铁桶僵尸", 270, 1100, 1370, "超高护甲", 3.4,
            ("铁桶高甲", "单发难破"), ("squash", "potato_mine", "cherry_bomb", "wallnut", "repeater"),
            ("tangle_kelp", "squash", "cherry_bomb", "repeater"),
            "水路用缠绕海草秒杀，陆路用窝瓜/土豆雷/樱桃或坚果+双发集火", True,
        ),
        ZombieProfile(
            "snorkel_zombie", "潜水僵尸", 270, 0, 270, "中低血量", 2.4,
            ("水路潜伏", "潜伏免直伤", "啃咬才露头"), (),
            ("tangle_kelp", "lily_pad", "wallnut", "cherry_bomb", "squash", "repeater", "peashooter"),
            "水路优先缠绕海草秒杀，或放睡莲/坚果逼迫露头后集火", True,
        ),
        ZombieProfile(
            "conehead_zombie", "路障僵尸", 270, 370, 640, "中等护甲", 1.9,
            ("路障护甲", "单豌豆吃紧"), ("repeater", "potato_mine", "squash", "peashooter", "wallnut"),
            ("tangle_kelp", "repeater", "peashooter", "wallnut"),
            "补双发/双豌豆火力，或用土豆雷/窝瓜低费处理", True,
        ),
        ZombieProfile(
            "flag_zombie", "旗帜僵尸", 270, 0, 270, "普通血量", 1.5,
            ("大波旗手", "移速略快"), ("peashooter", "repeater", "cherry_bomb", "wallnut"),
            ("peashooter", "repeater", "tangle_kelp"),
            "大波进攻信号，提前稳固多路火力并预留灰烬/窝瓜", True,
        ),
        ZombieProfile(
            "basic_zombie", "普通僵尸", 270, 0, 270, "普通血量", 1.0,
            ("常规慢速", "无护甲"), ("peashooter", "potato_mine", "repeater", "wallnut"),
            ("peashooter", "tangle_kelp", "repeater"),
            "常规单发火力即可处理，前期单走可用土豆雷省阳", True,
        ),
        ZombieProfile(
            "pole_vaulter", "撑杆跳僵尸", 500, 0, 500, "中等血量", 2.6,
            ("跳过首个植物",), ("wallnut", "squash", "potato_mine"), (),
            "用炮灰或坚果诱跳，再用窝瓜拦截", False,
        ),
        ZombieProfile(
            "football_zombie", "橄榄球僵尸", 270, 1400, 1670, "超高护甲", 4.3,
            ("极高护甲", "高速突进"), ("squash", "cherry_bomb", "wallnut", "repeater"), (),
            "优先窝瓜或樱桃炸弹，或坚果拖住后集中火力", False,
        ),
        ZombieProfile(
            "screen_door_zombie", "铁栅门僵尸", 270, 1100, 1370, "超高护甲", 3.2,
            ("前置盾牌挡直伤",), ("cherry_bomb", "squash", "spikeweed"), (),
            "用樱桃/窝瓜/地刺绕盾，或用穿透射击及投手", False,
        ),
        ZombieProfile(
            "dolphin_rider", "海豚骑士僵尸", 500, 0, 500, "中等血量", 3.0,
            ("水路高速", "跃过首个植物"), (), ("tangle_kelp", "lily_pad"),
            "用缠绕海草秒杀，或用睡莲诱跳、高坚果挡跳", False,
        ),
        ZombieProfile(
            "newspaper_zombie", "读报僵尸", 270, 150, 420, "轻甲", 2.1,
            ("破报后狂暴加速",), ("squash", "wallnut", "repeater"), (),
            "用坚果拖住后双发集火，或用窝瓜单解", False,
        ),
        ZombieProfile(
            "gargantuar", "巨人僵尸", 3000, 0, 3000, "巨量血量", 5.0,
            ("砸扁植物", "投掷小鬼"), ("cherry_bomb", "squash", "spikeweed"), (),
            "用樱桃炸弹加窝瓜双灰烬处理，必要时用地刺拖延", False,
        ),
        ZombieProfile(
            "bobsled_team", "冰车小队", 1080, 0, 1080, "小队合计血量", 3.1,
            ("4名队员各270血", "沿冰道高速滑行"), ("cherry_bomb", "repeater", "spikeweed"), (),
            "先拦截造冰的冰车，再用灰烬或双发清理小队", False,
        ),
    )
}

PLANT_CATALOG: dict[str, dict[str, Any]] = {
    "sunflower": {"name_cn": "向日葵", "cost": 50, "role": "产阳", "terrain": "land_or_lily_pad"},
    "lily_pad": {"name_cn": "睡莲", "cost": 25, "role": "水路底座", "terrain": "water_only"},
    "potato_mine": {"name_cn": "土豆雷", "cost": 25, "role": "延时单体爆发", "terrain": "land_only"},
    "peashooter": {"name_cn": "豌豆射手", "cost": 100, "role": "持续火力", "terrain": "land_or_lily_pad"},
    "cherry_bomb": {"name_cn": "樱桃炸弹", "cost": 150, "role": "范围爆发", "terrain": "land_or_lily_pad"},
    "repeater": {"name_cn": "双发射手", "cost": 200, "role": "双发火力", "terrain": "land_or_lily_pad"},
    "tangle_kelp": {"name_cn": "缠绕海草", "cost": 25, "role": "水路单解", "terrain": "water_only"},
    "squash": {"name_cn": "窝瓜", "cost": 50, "role": "近距爆发", "terrain": "land_or_lily_pad"},
    "wallnut": {"name_cn": "坚果", "cost": 50, "role": "阻挡", "terrain": "land_or_lily_pad"},
    "spikeweed": {"name_cn": "地刺", "cost": 100, "role": "爆胎与持续地面伤害", "terrain": "land_only"},
}

_REFERENCE_PNG = "evidence/day06/new-material-a-20261107-v1/canonical/frame-0000.png"
_FRAME_SHAPE = (600, 800, 3)
DEFAULT_LEFT_WARNING_X = 200
WATER_ROWS = (3, 4)
LAND_ROWS = (1, 2, 5, 6)
_DIVING_POSES = ("water_entry", "water_jump", "water_dive")
_SNORKEL_STANCES = {
    "diving": ("跃入下潜", ("水路突进", "即将潜伏免直伤")),
    "eating": ("露头啃咬", ("已露头可直伤", "正在啃咬植物")),
    "submerged": ("潜伏水中", ("水路潜伏", "潜伏免直伤")),
}
_DUCKY_TUBE_VETO = 180
_TRACK_MAX_AGE = 1.8
_OCCUPIED = ("ground_occupied", "base_only", "base_and_upper")


def is_plant_legal_for_row(plant_id: str, row: int) -> bool:
    """Check terrain compatibility. Board space and card availability are separate."""
    terrain = PLANT_CATALOG.get(plant_id, {}).get("terrain")
    if row in WATER_ROWS:
        return terrain != "land_only"
    if row in LAND_ROWS:
        return terrain != "water_only"
    return False


def get_zombie_profile(species_id: str) -> ZombieProfile:
    return ZOMBIE_SPECIES_DB.get(species_id, ZOMBIE_SPECIES_DB["basic_zombie"])


def extended_species_prompt_hint() -> str:
    return "；".join(
        f"{profile.name_cn}（{'270×4' if profile.species_id == 'bobsled_team' else profile.hp_total}血，"
        f"{'、'.join(profile.traits)}）：{profile.counter_hint}"
        for profile in ZOMBIE_SPECIES_DB.values() if not profile.cv_supported
    )


def _localized_point(candidate: Any, key: str) -> tuple[float, float] | None:
    if not isinstance(candidate, dict) or type(candidate.get("row")) is not int or not 1 <= candidate["row"] <= 6:
        return None
    point = candidate.get(key)
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    if not all(type(value) in (int, float) and isfinite(value) for value in point):
        return None
    return float(point[0]), float(point[1])


def is_localized_vehicle(candidate: Any) -> bool:
    return _localized_point(candidate, "front_xy") is not None


def _region(hx: int, hy: int, dx0: int, dx1: int, dy0: int, dy1: int, *, min_y: int = 0) -> tuple[slice, slice]:
    height, width, _ = _FRAME_SHAPE
    # The calibrated offsets include both endpoints.
    return (
        slice(max(min_y, hy + dy0), min(height, max(min_y, hy + dy1 + 1))),
        slice(max(0, hx + dx0), min(width, max(0, hx + dx1 + 1))),
    )


def _largest_component(mask: np.ndarray) -> tuple[int, int, int]:
    if not mask.any():
        return 0, 0, 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    foreground = stats[1:]
    biggest = foreground[np.argmax(foreground[:, cv2.CC_STAT_AREA])]
    return int(biggest[cv2.CC_STAT_AREA]), int(biggest[cv2.CC_STAT_WIDTH]), int(biggest[cv2.CC_STAT_HEIGHT])


def _bucket_support(rgb: np.ndarray, fg: np.ndarray, hx: int, hy: int) -> tuple[int, int, int, int]:
    roi = _region(hx, hy, -20, 28, -48, -10)
    pixels = rgb[roi]
    low = pixels.min(axis=2)
    mask = (low >= 105) & (pixels.max(axis=2) - low <= 32) & fg[roi]
    area, width, height = _largest_component(mask)
    return area, width, height, int(np.count_nonzero(mask))


def _hsv_roi(rgb: np.ndarray, roi: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop = rgb[roi]
    if not crop.size:
        return crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    return cv2.split(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV))


def _cone_support(rgb: np.ndarray, fg: np.ndarray, hx: int, hy: int) -> int:
    roi = _region(hx, hy, -20, 28, -45, -2)
    h, s, v = _hsv_roi(rgb, roi)
    return int(np.count_nonzero((10 <= h) & (h <= 34) & (s >= 120) & (v >= 135) & fg[roi]))


def _flag_supported(rgb: np.ndarray, fg: np.ndarray, hx: int, hy: int) -> bool:
    roi = _region(hx, hy, -52, -4, -32, 24, min_y=80)
    h, s, v = _hsv_roi(rgb, roi)
    area, width, _ = _largest_component(((h <= 8) | (h >= 168)) & (s >= 120) & (v >= 100) & fg[roi])
    return area >= 130 and width >= 14


def _ducky_tube_area(rgb: np.ndarray, fg: np.ndarray, hx: int, hy: int) -> int:
    roi = _region(hx, hy, -22, 30, 16, 62)
    h, s, v = _hsv_roi(rgb, roi)
    return _largest_component((16 <= h) & (h <= 36) & (s >= 110) & (v >= 130) & fg[roi])[0]


def _maroon_mask(rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
    if not rgb.size:
        return fg
    r, g, b = cv2.split(rgb)
    return ((35 <= r) & (r <= 135) & (g <= 55) & (b <= 65) & fg
            & (cv2.subtract(r, g) >= 16) & (cv2.subtract(r, b) >= 10))


def _goggle_rim(h: np.ndarray, s: np.ndarray, v: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return (17 <= h) & (h <= 39) & (s >= 100) & (v >= 95) & fg


def _snorkel_gear_score(rgb: np.ndarray, fg: np.ndarray, hx: int, hy: int, row: int | None = None) -> int:
    upper = _region(hx, hy, -25, 27, -30, 14)
    area, _, _ = _largest_component(_maroon_mask(rgb[upper], fg[upper]))
    if area < 24:
        return 0
    eye = _region(hx, hy, -19, 16, -12, 10)
    rim_area, rim_width, _ = _largest_component(_goggle_rim(*_hsv_roi(rgb, eye), fg[eye]))
    deep = row in WATER_ROWS and hy >= (304 if row == 3 else 384)
    return area if (rim_area >= 24 and rim_width >= 12) or deep else 0


def _cell_x(cell: dict[str, Any]) -> float | None:
    column = cell.get("column")
    # The canonical grid starts at x=40 and has 80-pixel cells.
    return 80 * column if type(column) is int and 1 <= column <= 9 else None


def _occupied_left_cells(cells: list[dict[str, Any]] | None, row: int, x: float) -> list[tuple[float, str]]:
    return [
        (cx, cell["occupancy"]) for cell in (cells or ()) if isinstance(cell, dict) and cell.get("row") == row
        and cell.get("occupancy") in _OCCUPIED and (cx := _cell_x(cell)) is not None and cx < x
    ]


def _position_stance(row: int, x: float, y: float, cells: list[dict[str, Any]] | None) -> str:
    if y <= (245 if row == 3 else 345):
        return "diving"
    if y >= (305 if row == 3 else 388):
        return "submerged"
    adjacent = any(x - cx <= 80 for cx, _ in _occupied_left_cells(cells, row, x))
    return "eating" if adjacent else "diving"


def _scan_snorkels(
    rgb: np.ndarray, fg: np.ndarray, water_heads: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    # The upper arc can be above the head band, especially when only the brow
    # and rim remain visible behind a splash.
    x0, x1 = 120, 786
    found = []
    for row, lo, hi in ((3, 255, 335), (4, 340, 425)):
        y0 = lo - 24
        area_mask = _maroon_mask(rgb[y0:hi + 1, x0:x1], fg[y0:hi + 1, x0:x1])
        _, _, stats, _ = cv2.connectedComponentsWithStats(area_mask.astype(np.uint8), connectivity=8)
        for xx, yy, width, height, area in stats[1:].tolist():
            if not (12 <= width <= 35 and 8 <= height <= 17 and 30 <= area <= 145 and area <= .68 * width * height):
                continue
            hx, hy = x0 + xx + max(width // 2, 14) + 1, y0 + yy + height
            if not (x0 <= hx < x1 and lo <= hy <= hi):
                continue
            if any((hx - x) ** 2 + (hy - y) ** 2 <= 34 ** 2 for x, y in water_heads):
                continue
            eye = _region(hx, hy, -19, 16, -12, 10)
            h, s, v = _hsv_roi(rgb, eye)
            rim, rim_width, rim_height = _largest_component(_goggle_rim(h, s, v, fg[eye]))
            if rim < 28 or rim_width < 16:
                continue
            dark = np.count_nonzero((s <= 100) & (30 <= v) & (v <= 115) & fg[eye])
            if hy >= (305 if row == 3 else 388):
                # A submerged sprite shows a thin cap arc and only a sliver of
                # upright rim. Ducky tubes form much larger yellow regions.
                if area > .33 * width * height or rim > 115 or rim_height > 12 or not 20 <= dark <= 160:
                    continue
                r, g, b = np.moveaxis(rgb[eye].astype(np.int16), -1, 0)
                skin = (60 <= g) & (g <= 155) & (g >= r) & (g >= b + 8) & (r >= b + 2)
                ripple = (b >= 128) & (g >= r + 30) & (b >= r + 38) & (b >= g - 16)
                supported = np.count_nonzero((skin | ripple) & fg[eye]) >= 65
                stance = "submerged"
            else:
                tube = _region(hx, hy, -34, -11, 2, 18)
                tube_area, _, _ = _largest_component(_maroon_mask(rgb[tube], fg[tube]))
                supported = dark >= 110 and rim_width >= 23 and tube_area >= 28
                stance = "eating"
            if supported and _ducky_tube_area(rgb, fg, hx, hy) < _DUCKY_TUBE_VETO:
                found.append(_snorkel_entity(row, float(hx), stance, "snorkel_scanner"))
                water_heads.append((hx, hy))
    return found


def _entity(
    species_id: str,
    row: int,
    x: float,
    hp_est: int,
    armor_stage: str,
    source: str,
    traits: tuple[str, ...] | None = None,
    stance: str = "normal",
) -> dict[str, Any]:
    profile = get_zombie_profile(species_id)
    return {
        "species_id": species_id, "name_cn": profile.name_cn, "row": row, "x": x,
        "is_water": row in WATER_ROWS, "hp_est": hp_est, "armor_stage": armor_stage,
        "traits": profile.traits if traits is None else traits,
        "threat_weight": profile.threat_weight, "source": source, "stance": stance,
    }


def _snorkel_entity(row: int, x: float, stance: str, source: str) -> dict[str, Any]:
    stage, traits = _SNORKEL_STANCES[stance]
    return _entity("snorkel_zombie", row, x, 270, stage, source, traits, stance)


def _classify_head(
    candidate: dict[str, Any],
    point: tuple[float, float],
    rgb: np.ndarray | None,
    fg: np.ndarray | None,
    cells: list[dict[str, Any]] | None = None,
    known_snorkel: bool = False,
) -> dict[str, Any]:
    row = candidate["row"]
    x, y = point
    hx, hy = round(x), round(y)
    hint = candidate.get("species_hint")
    if not isinstance(hint, str) or hint not in ZOMBIE_SPECIES_DB:
        hint = None
    source = "species_hint" if hint else "zombie_detector"
    hat = candidate.get("hat_hypothesis")
    water_pose = candidate.get("water_pose")
    has_pixels = rgb is not None and fg is not None

    if hint == "snorkel_zombie" or (hint in (None, "basic_zombie") and (
        candidate.get("pose") == "submerged" or water_pose in _DIVING_POSES
        or water_pose in ("water_submerged", "water_eating")
        or candidate.get("row_evidence") == "visible_water_pose_parts"
    )):
        stance = ("diving" if water_pose in _DIVING_POSES else
                  "eating" if water_pose == "water_eating" else "submerged")
        pose_source = source if hint == "snorkel_zombie" else "zombie_detector"
        return _snorkel_entity(row, x, stance, pose_source)

    if hint in (None, "buckethead_zombie") or (row in WATER_ROWS and hint == "basic_zombie"):
        area, bw, bh, pixel_count = _bucket_support(rgb, fg, hx, hy) if has_pixels else (0, 0, 0, 0)
        if hint == "buckethead_zombie" or hat == "bucket" or (area >= 340 and bw >= 16 and bh >= 16):
            helmet = candidate.get("helmet_support") or {}
            helmet_area = max(area, int(helmet.get("area") or 0), pixel_count if hat == "bucket" else 0)
            full = helmet_area >= 560
            return _entity("buckethead_zombie", row, x, 1370 if full else 820, "满甲" if full else "破损", source)

    if hint in (None, "conehead_zombie") or (row in WATER_ROWS and hint == "basic_zombie"):
        cone_area = _cone_support(rgb, fg, hx, hy) if has_pixels else 0
        if hint == "conehead_zombie" or hat == "cone" or cone_area >= 160:
            full = cone_area >= 420 or not has_pixels
            return _entity("conehead_zombie", row, x, 640 if full else 440, "满甲" if full else "受损", source)

    if row in WATER_ROWS and hint in (None, "basic_zombie") and has_pixels:
        gear = _snorkel_gear_score(rgb, fg, hx, hy, row)
        if (gear or known_snorkel) and _ducky_tube_area(rgb, fg, hx, hy) < _DUCKY_TUBE_VETO:
            return _snorkel_entity(row, x, _position_stance(row, x, y, cells),
                                   "snorkel_gear" if gear else "snorkel_track")

    if hint == "flag_zombie" or (
        hint is None and row not in WATER_ROWS and has_pixels and _flag_supported(rgb, fg, hx, hy)
    ):
        return _entity("flag_zombie", row, x, 270, "旗手", source)
    profile = get_zombie_profile(hint or "basic_zombie")
    stage = "载具满甲" if profile.species_id == "zomboni" else "满甲" if profile.hp_armor else "无护甲"
    return _entity(profile.species_id, row, x, profile.hp_total, stage, source)


def threat_detail(entity: dict[str, Any]) -> str:
    """Compact armor or behavior descriptor shared by summaries and local advice."""
    species_id = entity["species_id"]
    if species_id == "zomboni":
        return "碾压"
    if species_id == "snorkel_zombie":
        return entity["traits"][-1]
    if species_id == "basic_zombie":
        return ""
    return entity["armor_stage"] if get_zombie_profile(species_id).cv_supported else entity["traits"][0]


def _lane_summary(lane: dict[str, Any]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entity in lane["entities"]:
        groups.setdefault(entity["species_id"], []).append(entity)
    descriptions = []
    for species_id, members in groups.items():
        name = get_zombie_profile(species_id).name_cn.removesuffix("僵尸")
        hp = sum(entity["hp_est"] for entity in members)
        details = {threat_detail(entity) for entity in members}
        detail = f"·{next(iter(details))}" if len(details) == 1 and "" not in details else ""
        descriptions.append(f"{name}×{len(members)}({hp}血{detail})")
    return "、".join(descriptions)


def _classify_frame(
    rgb: np.ndarray | None,
    ref_rgb: np.ndarray | None,
    state: dict[str, Any],
    left_warning_x: int,
    snorkel_tracks: dict[tuple[int, str], float] | None = None,
) -> dict[str, Any]:
    entities = []
    for candidate in state.get("vehicles", {}).get("candidates", ()):
        point = _localized_point(candidate, "front_xy")
        if point is not None:
            entities.append(_entity("zomboni", candidate["row"], point[0], 1350, "载具满甲", "vehicle_detector"))

    heads = state.get("zombies", {}).get("candidates", ())
    playing = state.get("scene", {}).get("value") == "playing"
    fg = None
    if (heads or playing) and rgb is not None and rgb.shape == _FRAME_SHAPE and rgb.dtype == np.uint8:
        if ref_rgb is None:
            fg = np.ones(_FRAME_SHAPE[:2], dtype=bool)
        else:
            # inRange tests each channel, so its inverse is max(abs(RGB - reference)) >= 24.
            diff = cv2.absdiff(rgb, ref_rgb)
            fg = cv2.inRange(diff, (0, 0, 0), (23, 23, 23)) == 0
    cells = state.get("fields", {}).get("cells")
    water_heads = []
    for candidate in heads:
        point = _localized_point(candidate, "head_xy")
        if point is not None:
            row, track_id = candidate["row"], candidate.get("track_id")
            key = (row, track_id) if row in WATER_ROWS and isinstance(track_id, str) and track_id else None
            known = key is not None and snorkel_tracks is not None and key in snorkel_tracks
            entity = _classify_head(candidate, point, rgb, fg, cells, known)
            entities.append(entity)
            if row in WATER_ROWS:
                water_heads.append(point)
            if (key is not None and snorkel_tracks is not None and entity["species_id"] == "snorkel_zombie"
                    and entity["source"] != "snorkel_track"):
                snorkel_tracks[key] = state["frame_time_seconds"]
    if playing and fg is not None:
        entities.extend(_scan_snorkels(rgb, fg, water_heads))
    return _aggregate(entities, left_warning_x)


def _aggregate(entities: list[dict[str, Any]], left_warning_x: int) -> dict[str, Any]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for entity in entities:
        x = entity["x"]
        proximity = max(0.0, (800.0 - x) / 800.0)
        left_boost = 2.5 if x <= left_warning_x else 0.0
        entity["entity_urgency"] = round(entity["threat_weight"] * (1.0 + 1.8 * proximity) + left_boost, 3)
        rows.setdefault(entity["row"], []).append(entity)

    by_lane = {}
    for row, members in sorted(rows.items()):
        primary = max(members, key=lambda entity: (entity["entity_urgency"], -entity["x"]))
        has_vehicle = any(entity["species_id"] == "zomboni" for entity in members)
        count = len(members)
        by_lane[row] = {
            "entities": members, "count": count, "has_vehicle": has_vehicle,
            "is_water": row in WATER_ROWS, "total_hp_est": sum(entity["hp_est"] for entity in members),
            "max_threat_weight": max(entity["threat_weight"] for entity in members),
            "leftmost_x": min(entity["x"] for entity in members),
            "species_counts": dict(Counter(entity["species_id"] for entity in members)),
            "primary_entity": primary,
            "lane_urgency": primary["entity_urgency"] + 0.35 * (count - 1) + (1.5 if has_vehicle else 0.0),
        }
    priority_row = max(
        by_lane,
        key=lambda row: (by_lane[row]["lane_urgency"], -by_lane[row]["leftmost_x"], -row),
        default=None,
    )
    return {
        "by_lane": by_lane, "priority_row": priority_row,
        "species_counts": dict(Counter(entity["species_id"] for entity in entities)),
        "summary_short": "; ".join(f"{row}路:{_lane_summary(lane)}" for row, lane in by_lane.items()) or "无确认威胁",
        "left_warning_x": left_warning_x,
    }


class ZombieSpeciesClassifier:
    def __init__(self, root: Path):
        self._ref_rgb: np.ndarray | None = None
        self._track_run: tuple[str, int | None] | None = None
        self._last_frame_time: float | None = None
        self._snorkel_tracks: dict[tuple[int, str], float] = {}
        try:
            data = (Path(root) / _REFERENCE_PNG).read_bytes()
        except FileNotFoundError:
            return
        if data:
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None and bgr.shape == _FRAME_SHAPE:
                self._ref_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def classify_frame(
        self,
        rgb: np.ndarray | None,
        state: dict[str, Any],
        *,
        left_warning_x: int = DEFAULT_LEFT_WARNING_X,
    ) -> dict[str, Any]:
        run_id, stamp = state.get("run_id"), state.get("frame_time_seconds")
        tracks = None
        if isinstance(run_id, str) and run_id and type(stamp) in (float, int) and isfinite(stamp):
            run = (run_id, (state.get("tracking_context") or {}).get("epoch"))
            if (run != self._track_run or self._last_frame_time is None or stamp < self._last_frame_time
                    or state.get("scene", {}).get("value") not in (None, "playing")):
                self._snorkel_tracks.clear()
            self._track_run, self._last_frame_time = run, stamp
            self._snorkel_tracks = {
                key: seen for key, seen in self._snorkel_tracks.items() if stamp - seen <= _TRACK_MAX_AGE
            }
            if state.get("scene", {}).get("value") in (None, "playing"):
                tracks = self._snorkel_tracks
        return _classify_frame(rgb, self._ref_rgb, state, left_warning_x, tracks)


def classify_state_metadata(
    state: dict[str, Any], *, left_warning_x: int = DEFAULT_LEFT_WARNING_X,
) -> dict[str, Any]:
    return _classify_frame(None, None, state, left_warning_x)


def prioritize_zombie_intel(intel: dict[str, Any], left_warning_x: int) -> dict[str, Any]:
    """Apply a caller's warning threshold to existing classifications without re-reading pixels."""
    if intel.get("left_warning_x", DEFAULT_LEFT_WARNING_X) == left_warning_x:
        return intel
    return _aggregate(
        [dict(entity) for lane in intel["by_lane"].values() for entity in lane["entities"]], left_warning_x,
    )


def supplementary_snorkels(intel: dict[str, Any]) -> list[tuple[int, float]]:
    """Pool threats localized by the supplemental scanner, outside the frozen detector."""
    return [
        (row, entity["x"])
        for row, lane in intel.get("by_lane", {}).items()
        for entity in lane.get("entities", ())
        if entity.get("source") == "snorkel_scanner"
    ]


_COUNTER_ACTIONS = {
    ("zomboni", "spikeweed"): "放地刺({cost})爆胎秒杀",
    ("zomboni", "squash"): "放窝瓜({cost})紧急单解冰车",
    ("zomboni", "cherry_bomb"): "放樱桃炸弹({cost})紧急拦截冰车",
    ("buckethead_zombie", "squash"): "放窝瓜({cost})单杀高甲",
    ("buckethead_zombie", "potato_mine"): "放土豆雷({cost})提前埋雷秒杀",
    ("buckethead_zombie", "cherry_bomb"): "放樱桃炸弹({cost})清高甲",
    ("buckethead_zombie", "wallnut"): "放坚果({cost})拖住铁桶，集中火力",
    ("buckethead_zombie", "repeater"): "补双发射手({cost})集火破甲",
    ("conehead_zombie", "repeater"): "补双发射手({cost})压制路障",
    ("conehead_zombie", "potato_mine"): "埋土豆雷({cost})省阳解路障",
    ("conehead_zombie", "peashooter"): "补豌豆({cost})集火压制路障",
    ("conehead_zombie", "wallnut"): "放坚果({cost})拖延路障",
    ("snorkel_zombie", "lily_pad"): "放睡莲({cost})垫潜逼迫露头",
    ("snorkel_zombie", "wallnut"): "放坚果({cost})逼迫露头，配合集火",
    ("snorkel_zombie", "cherry_bomb"): "放樱桃炸弹({cost})范围清潜",
    ("snorkel_zombie", "squash"): "放窝瓜({cost})压杀露头潜水",
    ("snorkel_zombie", "repeater"): "补双发射手({cost})趁啃咬集火",
    ("snorkel_zombie", "peashooter"): "补豌豆({cost})趁啃咬集火",
}


def _ready_action(species_id: str, plant_id: str, cost: int) -> str:
    if plant_id == "tangle_kelp":
        return f"水路放缠绕海草({cost})秒杀"
    template = _COUNTER_ACTIONS.get((species_id, plant_id))
    if template:
        return template.format(cost=cost)
    return f"放{PLANT_CATALOG[plant_id]['name_cn']}({cost})应对{get_zombie_profile(species_id).name_cn}"


def _firepower(species_id: str, total_hp: int) -> str:
    if species_id == "zomboni":
        return "anti_vehicle"
    if species_id == "buckethead_zombie" or total_hp >= 1200:
        return "heavy_burst"
    if species_id == "snorkel_zombie":
        return "water_counter"
    if species_id == "conehead_zombie" or total_hp >= 550:
        return "medium_dps"
    return "standard_dps"


_WATER_COUNTER_HINTS = {
    "basic_zombie": "保留同路火力应对普僵，观察水路空位",
    "conehead_zombie": "集中同路火力破路障护甲，预留阳光应急",
    "buckethead_zombie": "集中同路火力破铁桶护甲，预留阳光应急",
    "zomboni": "确认水路目标后再决定对策，先留阳并观察",
    "snorkel_zombie": "盯住水路，潜水啃咬露头时集中后排火力",
}

_LAND_COUNTER_HINTS = {
    "buckethead_zombie": "陆路用窝瓜/土豆雷/樱桃或坚果+双发集火",
    "snorkel_zombie": "确认陆路目标，先观察并保留阳光",
    "dolphin_rider": "确认陆路目标，先观察并保留阳光",
}


def _placement_modes(row, target_x, counters, cells, ready, sun):
    """Resolve a confirmed placement or a currently affordable two-card placement."""
    if row in LAND_ROWS:
        return {plant: "direct" for plant in counters}
    known_cells = [cell for cell in (cells or ()) if isinstance(cell, dict)]
    row_occupancy = {cell.get("occupancy") for cell in known_cells if cell.get("row") == row}
    empty_water = cells is None or "empty" in row_occupancy
    splash_row = 2 if row == 3 else 5
    can_splash = any(
        cell.get("row") == splash_row and cell.get("occupancy") == "empty"
        and (cx := _cell_x(cell)) is not None and abs(cx - target_x) <= 1.5 * 80
        for cell in known_cells
    )
    modes = {}
    for plant in counters:
        if PLANT_CATALOG[plant]["terrain"] == "water_only":
            if empty_water:
                modes[plant] = "direct"
        elif "base_only" in row_occupancy:
            modes[plant] = "existing_pad"
        elif plant == "cherry_bomb" and can_splash:
            modes[plant] = "splash"
        elif empty_water and "lily_pad" in ready and plant in ready and ready["lily_pad"] + ready[plant] <= sun:
            modes[plant] = "with_pad"
    return modes


def _placement_action(action, mode, row, cost, *, wait=""):
    if mode == "splash":
        land_row = 2 if row == 3 else 5
        return f"{wait}{land_row}路陆地放樱桃炸弹({cost})溅射{row}路"
    prefix = {"with_pad": "先垫睡莲再", "existing_pad": "在空睡莲上"}.get(mode, "")
    return wait + prefix + action


def _snorkel_tactics(primary, lane, row, cells, options, later, modes, ready, sun, fallback):
    stance = primary.get("stance", "normal")
    lane_has_plants = bool(_occupied_left_cells(cells, row, primary["x"]))
    urgent = lane["count"] >= 2 or primary["x"] <= 280
    eating = stance == "eating"
    kelp, pad, nut, cherry, squash, repeater, pea = (
        "tangle_kelp", "lily_pad", "wallnut", "cherry_bomb", "squash", "repeater", "peashooter",
    )
    order = (kelp, squash, repeater, pea, nut, pad, cherry) if urgent and eating else (
        (squash, repeater, pea, kelp, nut, pad, cherry) if eating else (kelp, pad, cherry, nut, squash)
    )
    available = [item for item in order if item in options]
    recommended = []
    action = fallback
    if urgent and cherry in options:
        recommended, action = [cherry], options[cherry]
    elif (urgent or (not eating and kelp not in options)) and modes.get(nut) == "with_pad":
        recommended, action = [pad, nut], options[nut]
    elif (
        urgent and kelp in options and pad in options and ready[kelp] + ready[pad] <= sun
        and (cells is None or len({
            cell.get("column") for cell in cells if isinstance(cell, dict)
            and cell.get("row") == row and cell.get("occupancy") == "empty"
        }) >= 2)
    ):
        recommended = [kelp, pad]
        action = f"海草({ready[kelp]})先抓最前，睡莲({ready[pad]})垫潜拖住其余"
    elif available:
        best = available[0]
        recommended, action = [best], options[best]
    elif later:
        best, action = next(iter(later.items()))
        recommended = [best]

    alternatives = [options[item] for item in available if item not in recommended]
    alternatives.extend(message for item, message in later.items() if item not in recommended)
    return {
        "action_text": action, "recommended_plants": recommended,
        "alternatives": alternatives[:3] or [fallback], "target_stance": stance,
        "lane_count": lane["count"], "snorkel_count": lane.get("species_counts", {}).get(primary["species_id"], 1),
        "lane_has_plants": lane_has_plants,
    }


def recommend_counter_plan(
    intel: dict[str, Any],
    card_slots: list[dict[str, Any]],
    sun: int | None,
    *,
    raw_cards: list[dict[str, Any]] | None = None,
    cells: list[dict[str, Any]] | None = None,
    lane_defense: dict | None = None,
) -> dict[str, Any]:
    focus_row = intel.get("priority_row")
    if lane_defense is None:
        plan = _counter_plan(intel, focus_row, card_slots, sun, raw_cards, cells)
        return {**plan, "priority_row": focus_row, "needs_reinforcement": focus_row is not None,
                "status": "reinforce" if focus_row is not None else "clear", "action_hint": plan["action_text"]}

    lanes = lane_defense.get("by_row", {})
    needs = [row for row in lane_defense.get("needs_defense_rows", ())
             if row in intel["by_lane"] and lanes.get(row, {}).get("verdict") in ("deficit", "tight")]
    sufficient = [
        f"{row}路[{lanes[row]['plants_compact_zh'] or '现有部署'}]"
        for row in lane_defense.get("sufficient_threat_rows", ()) if row in lanes
    ]
    covered = f"{'、'.join(sufficient)}已稳守，无需重复增防" if sufficient else ""
    focus_row = needs[0] if needs else None
    plan = _counter_plan(intel, focus_row, card_slots, sun, raw_cards, cells)
    if focus_row is not None:
        lane = lanes[focus_row]
        existing = lane["plants_compact_zh"] or "无已确认部署"
        basis = f"{focus_row}路[{existing}]判定{lane['verdict_zh']}，{lane['reason_zh']}"
        action = f"{basis}，建议{plan['action_text']}"
        line = f"{basis}。{plan['llm_counter_line']}"
        if covered:
            action += f"；{covered}"
            line += f"。{covered}"
        return {**plan, "priority_row": focus_row, "status": "reinforce",
                "needs_reinforcement": True, "action_hint": action, "llm_counter_line": line}

    status, action = "clear", ""
    if lane_defense.get("all_threatened_sufficient"):
        status = "all_sufficient"
        action = f"{covered or '受威胁路已稳守'}，可收阳、补经济或留阳"
    elif lane_defense.get("unknown_threat_rows"):
        status = "observe_unknown"
        rows = "、".join(f"{row}路" for row in lane_defense["unknown_threat_rows"])
        action = f"{rows}部署有遮挡，先观察并留备用阳光"
        if covered:
            action += f"；{covered}"
    return {**plan, "priority_row": None, "status": status, "needs_reinforcement": False,
            "action_hint": action, "action_text": action, "llm_counter_line": action}


def _counter_plan(intel, focus_row, card_slots, sun, raw_cards, cells):
    if focus_row is None:
        return {
            "focus_row": None, "target_species_id": None, "target_species_cn": None,
            "target_hp_est": None, "target_armor_stage": None, "target_traits": (),
            "firepower_level": "none", "recommended_plants": [], "ready_counters": [],
            "action_text": "", "llm_counter_line": "",
        }

    lane = intel["by_lane"][focus_row]
    primary = lane["primary_entity"]
    profile = get_zombie_profile(primary["species_id"])
    species_id = profile.species_id
    is_water = focus_row in WATER_ROWS
    candidates = (*profile.water_counters, "cherry_bomb") if is_water else profile.land_counters
    counters = [plant for plant in dict.fromkeys(candidates) if is_plant_legal_for_row(plant, focus_row)]
    cards_to_check = {*counters, "lily_pad"} if is_water else set(counters)
    visuals = {
        card["slot"]: card.get("visual_state")
        for card in (raw_cards or ()) if isinstance(card, dict) and "slot" in card
    }
    ready: dict[str, int] = {}
    need_sun: dict[str, int] = {}
    cooling: dict[str, int] = {}
    for slot in card_slots:
        plant = slot.get("plant_id")
        if slot.get("status") != "observed" or plant not in cards_to_check:
            continue
        visual = "bright" if raw_cards is None else visuals.get(slot.get("slot"))
        cost = slot.get("printed_cost", slot.get("display_cost"))
        if visual == "dim":
            cooling[plant] = cost if type(cost) is int else PLANT_CATALOG[plant]["cost"]
        elif visual == "bright" and sun is not None and type(cost) is int:
            budget = ready if sun >= cost else need_sun
            budget[plant] = min(cost, budget.get(plant, cost))

    modes = _placement_modes(focus_row, primary["x"], counters, cells, ready, sun)
    ready_counters = [plant for plant in counters if plant in ready and plant in modes]
    fallback = (
        _WATER_COUNTER_HINTS.get(species_id, "观察水路威胁，保留阳光并利用现有火力")
        if is_water else _LAND_COUNTER_HINTS.get(species_id, profile.counter_hint)
    )
    options = {}
    later = {}
    for plant, mode in modes.items():
        name = PLANT_CATALOG[plant]["name_cn"]
        if plant in ready:
            cost = ready[plant]
            options[plant] = _placement_action(
                _ready_action(species_id, plant, cost), mode, focus_row, cost,
            )
        elif plant in need_sun or plant in cooling:
            cost = need_sun[plant] if plant in need_sun else cooling[plant]
            wait = f"攒阳到{cost}再" if plant in need_sun else f"等{name}冷却后"
            if plant == "lily_pad" and species_id == "snorkel_zombie":
                action = "垫前逼露头" if plant in cooling else "放睡莲垫前逼露头"
            else:
                effect = "爆胎" if species_id == "zomboni" and plant == "spikeweed" else f"应对{profile.name_cn}"
                action = f"放{name}{effect}"
            if mode == "direct" and plant in need_sun:
                wait = f"攒阳到{cost}"
            later[plant] = _placement_action(action, mode, focus_row, cost, wait=wait)

    tactics = {}
    recommended = []
    action = fallback
    if species_id == "snorkel_zombie" and is_water:
        tactics = _snorkel_tactics(primary, lane, focus_row, cells, options, later, modes, ready, sun, fallback)
        recommended, action = tactics["recommended_plants"], tactics["action_text"]
    elif ready_counters:
        plant = ready_counters[0]
        if (
            species_id == "buckethead_zombie" and not is_water and plant == "wallnut"
            and "repeater" in ready and sun >= ready["wallnut"] + ready["repeater"]
        ):
            recommended = ["wallnut", "repeater"]
            action = f"坚果({ready['wallnut']})挡住+补双发射手({ready['repeater']})集火"
        else:
            recommended = [plant]
            action = options[plant]
    elif later:
        plant = next((plant for plant in later if plant in need_sun), next(iter(later)))
        recommended, action = [plant], later[plant]

    action = action[:46]
    line = f"第{focus_row}路{profile.name_cn}（{primary['armor_stage']}，约{primary['hp_est']}血）：{action}"
    if tactics:
        line += f"。备选对策：{'；'.join(tactics['alternatives'])}"
    if len(recommended) == 1 and modes.get(recommended[0]) == "with_pad":
        recommended = ["lily_pad", *recommended]
    return {
        **tactics,
        "focus_row": focus_row, "target_species_id": species_id, "target_species_cn": profile.name_cn,
        "target_hp_est": primary["hp_est"], "target_armor_stage": primary["armor_stage"],
        "target_traits": primary["traits"], "firepower_level": _firepower(species_id, lane["total_hp_est"]),
        "recommended_plants": recommended, "ready_counters": ready_counters, "action_text": action,
        "llm_counter_line": line,
    }
