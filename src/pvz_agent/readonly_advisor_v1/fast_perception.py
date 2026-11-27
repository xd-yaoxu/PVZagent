"""Fast single-pass perception, cached-affine geometry normalizer, 3-channel evaluator, and tactical phase timeline.

Touches zero of the 59 pinned startup files; composes the frozen v10 resource stream
and v1 in-memory card identity in a single pass (~720 ms/frame vs ~1620 ms/frame in v1r7).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..card_identity import CardIdentityPerception, gray as card_gray, source_run
from ..goal_review import ReviewBlocked
from ..product_card_strategy import card_context
from ..product_startup import verify_product_startup
from ..resource_review import sun_cue_ids
from ..resource_stream import create_product_resource_stream
from ..resource_threat_v2 import validate_threat_context
from .lane_defense import evaluate_all_lanes
from .plant_intel import PlantBoardClassifier, WATER_ROWS
from .zombie_intel import (
    DEFAULT_LEFT_WARNING_X,
    ZombieSpeciesClassifier,
    classify_state_metadata,
    is_localized_vehicle,
    prioritize_zombie_intel,
    recommend_counter_plan,
    supplementary_snorkels,
    threat_detail,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PNG = REPO_ROOT / "evidence/day06/new-material-a-20261107-v1/canonical/frame-0000.png"
REFERENCE_SHA256 = "afb3daf6d2f6e5d1db3f02d44e72343cb7fa41cf2b08865adb2ebf8edf0a6c86"
TACTICAL_PRIORITIES_JSON = REPO_ROOT / "config/tactical-priorities.json"

CANONICAL_SIZE = (800, 600)
RGB_BYTES = 800 * 600 * 3
ENVELOPE_LIMIT_PX = {(800, 600): 2.5, (1200, 900): 3.75}
BOARD_RECT = (40, 80, 760, 590)
SUN_RECT = (24, 55, 68, 72)
NO_EVIDENCE_TEXT = "当前无有效对局帧证据（未连接、已暂停或已过期）"

REGIONS = {
    "fit_pool_rim_top": (25, 252, 790, 292),
    "fit_pool_rim_bottom": (25, 430, 790, 460),
    "fit_right_board_upper": (675, 92, 790, 247),
    "fit_right_board_lower": (690, 465, 790, 570),
    "hold_upper_bank_structure": (4, 4, 85, 53),
    "hold_left_board_edge": (4, 100, 41, 248),
    "hold_lower_board_left": (8, 476, 218, 590),
    "hold_lower_board_middle": (265, 508, 611, 594),
    "hold_right_hud": (688, 5, 792, 80),
}
PATCHES = {
    "bank_top": (10, 10, 76, 47),
    "pool_left_upper": (7, 253, 89, 279),
    "pool_floor_rim_lower": (252, 433, 367, 454),
    "right_board": (726, 469, 785, 526),
    "left_board": (8, 130, 32, 213),
    "bottom_left": (14, 487, 95, 570),
    "bottom_middle": (280, 550, 379, 584),
}


def _covered(matrix: np.ndarray, size: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    points = np.array([(x1, y1), (x1, y2 - 1), (x2 - 1, y1), (x2 - 1, y2 - 1)], dtype=np.float64)
    xy = points @ matrix[:, :2].T + matrix[:, 2]
    return bool((xy >= 0).all() and (xy < np.array(size) - 1).all())


class FastGeometryNormalizer:
    """Frame-1 full occlusion-tolerant SIFT + fast cached-affine HUD/board anchor verification."""

    def __init__(self, root: Path = REPO_ROOT):
        self.root = Path(root).resolve()
        ref_path = self.root / "evidence/day06/new-material-a-20261107-v1/canonical/frame-0000.png"
        blob = ref_path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != REFERENCE_SHA256:
            raise ValueError("canonical_reference_hash_mismatch")
        with Image.open(ref_path) as im:
            self.ref_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        assert self.ref_rgb.shape == (600, 800, 3)
        self.ref_gray = cv2.cvtColor(self.ref_rgb, cv2.COLOR_RGB2GRAY)
        self._sift = cv2.SIFT_create(nfeatures=6000)
        self._rk, self._rd = self._sift.detectAndCompute(self.ref_gray, None)
        self._region_indices = {
            name: [i for i, k in enumerate(self._rk) if x1 <= k.pt[0] < x2 and y1 <= k.pt[1] < y2]
            for name, (x1, y1, x2, y2) in REGIONS.items()
        }
        self._cached_rect_sig: tuple[int, ...] | None = None
        self._cached_size: tuple[int, int] | None = None
        self._cached_matrix: np.ndarray | None = None
        self._cached_corner_err: float = 0.0
        self.fast_hits = 0
        self.sift_runs = 0

    def invalidate_cache(self) -> None:
        self._cached_rect_sig = None
        self._cached_size = None
        self._cached_matrix = None

    def _anchor_ok(self, raw_gray: np.ndarray, matrix: np.ndarray) -> bool:
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        board_ok = 0
        bank_ok = False
        for name, (x1, y1, x2, y2) in PATCHES.items():
            yy, xx = np.indices((y2 - y1 + 8, x2 - x1 + 8), dtype=np.float32)
            xx += x1 - 4
            yy += y1 - 4
            mx = (matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]).astype(np.float32)
            my = (matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]).astype(np.float32)
            if mx.min() < 0 or my.min() < 0 or mx.max() >= raw_gray.shape[1] - 1 or my.max() >= raw_gray.shape[0] - 1:
                continue
            band = cv2.remap(raw_gray, mx, my, cv2.INTER_LINEAR)
            score = cv2.matchTemplate(band, self.ref_gray[y1:y2, x1:x2], cv2.TM_CCOEFF_NORMED)
            _, peak, _, (px, py) = cv2.minMaxLoc(score)
            corroborated = peak >= 0.62 and (scale * math.hypot(px - 4, py - 4)) <= 2.0
            if name == "bank_top":
                bank_ok = corroborated
            elif corroborated:
                board_ok += 1
        return bank_ok and board_ok >= 3

    def _fit_full_sift(self, rgb_src: np.ndarray, raw_gray: np.ndarray) -> np.ndarray:
        self.sift_runs += 1
        corners = np.array([[0.0, 0.0], [799.0, 0.0], [0.0, 599.0], [799.0, 599.0]], dtype=np.float64)
        dk, dd = self._sift.detectAndCompute(raw_gray, None)
        if self._rd is None or dd is None:
            raise ValueError("geometry_features_missing")
        groups = {}
        matcher = cv2.BFMatcher()
        for key, use in self._region_indices.items():
            matches = matcher.knnMatch(self._rd[use], dd, k=2)
            good = [p[0] for p in matches if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
            groups[key] = (
                np.float32([self._rk[use[m.queryIdx]].pt for m in good]).reshape(-1, 2),
                np.float32([dk[m.trainIdx].pt for m in good]).reshape(-1, 2),
            )
        fit = {k: v for k, v in groups.items() if k.startswith("fit_")}
        hold = {k: v for k, v in groups.items() if k.startswith("hold_")}
        src = np.vstack([v[0] for v in fit.values()])
        dst = np.vstack([v[1] for v in fit.values()])
        if len(src) < 12:
            raise ValueError("insufficient_fit_matches")
        matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if matrix is None or mask is None:
            raise ValueError("affine_estimation_failed")
        h_src = np.vstack([v[0] for v in hold.values()])
        h_dst = np.vstack([v[1] for v in hold.values()])
        if len(h_src) < 6:
            raise ValueError("insufficient_hold_matches")

        def residuals(m, a, b):
            return np.linalg.norm((a @ m[:, :2].T + m[:, 2]) - b, axis=1)

        fit_in = residuals(matrix, src, dst) <= 2.5
        hold_in = residuals(matrix, h_src, h_dst) <= 2.5
        fit_counts = {k: int((residuals(matrix, a, b) <= 2.5).sum()) for k, (a, b) in fit.items()}
        hold_counts = {k: int((residuals(matrix, a, b) <= 2.5).sum()) for k, (a, b) in hold.items()}

        affine, _ = cv2.estimateAffine2D(
            src[fit_in], dst[fit_in], method=cv2.RANSAC, ransacReprojThreshold=2.5
        ) if int(fit_in.sum()) >= 6 else (None, None)

        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        fit_p95 = float(np.percentile(residuals(matrix, src[fit_in], dst[fit_in]), 95)) if int(fit_in.sum()) else 999.0
        hold_p95 = float(np.percentile(residuals(matrix, h_src[hold_in], h_dst[hold_in]), 95)) if int(hold_in.sum()) else 999.0
        mapped = corners @ matrix[:, :2].T + matrix[:, 2]
        h_img, w_img = rgb_src.shape[:2]
        cov = float(
            max(0.0, min(mapped[:, 0].max(), w_img - 1) - max(mapped[:, 0].min(), 0))
            * max(0.0, min(mapped[:, 1].max(), h_img - 1) - max(mapped[:, 1].min(), 0))
            / (w_img * h_img)
        )
        aff_err = (
            float(np.linalg.norm(mapped - (corners @ affine[:, :2].T + affine[:, 2]), axis=1).max())
            if affine is not None else 999.0
        )
        fit_good = (
            int(fit_in.sum()) >= 40
            and sum(v >= 10 for v in fit_counts.values()) >= 2
            and fit_p95 <= 2.0
        )
        hold_good = (
            int(hold_in.sum()) >= 25
            and sum(v >= 5 for v in hold_counts.values()) >= 2
            and hold_p95 <= 2.0
        )
        gates = {
            "fit_good": fit_good,
            "hold_good": hold_good,
            "scale_rotation": 0.45 <= scale <= 3.0 and abs(angle) <= 1.0,
            "coverage": cov >= 0.985,
            "affine_agreement": aff_err <= 3.0,
            "direct_non_digit_hud_and_board": self._anchor_ok(raw_gray, matrix),
        }
        if not all(gates.values()):
            failed = ",".join(k for k, v in gates.items() if not v)
            raise ValueError(f"geometry_gates_failed:{failed}")
        return matrix

    def normalize(
        self,
        bgra: np.ndarray,
        size: tuple[int, int],
        rect_signature: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
        """Normalize raw BGRA client buffer (800x600 or 1200x900) to canonical 800x600 RGB."""
        t0 = time.perf_counter()
        rgb_src = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        if not np.any(rgb_src) or np.all(rgb_src == rgb_src[0, 0]):
            return None, "uniform_or_black", {"mode": "rejected", "elapsed_ms": (time.perf_counter() - t0) * 1000.0}
        raw_gray = cv2.cvtColor(rgb_src, cv2.COLOR_RGB2GRAY)

        matrix = None
        mode = "sift"
        if (
            self._cached_matrix is not None
            and self._cached_size == size
            and (rect_signature is None or self._cached_rect_sig == rect_signature)
            and self._anchor_ok(raw_gray, self._cached_matrix)
        ):
            matrix = self._cached_matrix
            mode = "cached_affine"
            self.fast_hits += 1
        else:
            try:
                matrix = self._fit_full_sift(rgb_src, raw_gray)
            except ValueError as exc:
                self.invalidate_cache()
                return None, str(exc), {"mode": "sift_rejected", "elapsed_ms": (time.perf_counter() - t0) * 1000.0}

        corners = np.array([[0.0, 0.0], [799.0, 0.0], [0.0, 599.0], [799.0, 599.0]], dtype=np.float64)
        expected = corners * (np.array(size, dtype=np.float64) / np.array([800.0, 600.0]))
        actual = corners @ matrix[:, :2].T + matrix[:, 2]
        corner_err = float(np.max(np.abs(actual - expected)))
        if corner_err > ENVELOPE_LIMIT_PX.get(size, 3.75):
            self.invalidate_cache()
            return None, "geometry_envelope", {"mode": mode, "corner_err_px": corner_err}
        if not (_covered(matrix, size, BOARD_RECT) and _covered(matrix, size, SUN_RECT)):
            self.invalidate_cache()
            return None, "required_region_clipped", {"mode": mode, "corner_err_px": corner_err}

        self._cached_matrix = matrix
        self._cached_size = size
        self._cached_rect_sig = rect_signature
        self._cached_corner_err = corner_err

        warped = cv2.warpAffine(
            rgb_src, matrix, CANONICAL_SIZE, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )
        return warped, None, {
            "mode": mode,
            "corner_err_px": round(corner_err, 2),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }


class FastSinglePassPerception:
    """Single-pass composition of the 59-file frozen resource stream + in-memory v1 card identity."""

    def __init__(self, root: Path = REPO_ROOT):
        self.root = Path(root).resolve()
        startup = verify_product_startup(self.root)
        if startup.get("artifacts_checked") != 59:
            raise ValueError("product_startup_59_artifacts_mismatch")
        freeze = json.loads(
            (self.root / "evidence/day06/card-identity-v1/freeze-v1.json").read_text(encoding="utf-8")
        )
        for rel, expected in freeze["artifacts"].items():
            if hashlib.sha256((self.root / rel).read_bytes()).hexdigest() != expected:
                raise ValueError(f"frozen_card_artifact_changed:{rel}")
        self.config = json.loads((self.root / "config/card-identity-v1.json").read_text(encoding="utf-8"))
        self.base = create_product_resource_stream(self.root)
        self.cards = CardIdentityPerception(self.root, self.config)
        self.catalog = copy.deepcopy(self.cards.catalog)
        self.zombie_classifier = ZombieSpeciesClassifier(self.root)
        self._plant_classifier = PlantBoardClassifier(self.root)
        self._plant_epoch = None

    def reset(self) -> None:
        self._plant_classifier.reset()
        self._plant_epoch = None

    def _observe_cards_in_memory(self, image: Image.Image, state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        result = {
            "schema": "pvz-card-identity/1",
            "state_id": state["state_id"],
            "capture_time": state["capture_time"],
            "source_run_id": source_run(state),
            "evidence": copy.deepcopy(state["evidence"]),
            "catalog_sha256": self.config["catalog_sha256"],
            "config_sha256": self.cards.config_sha256,
            "slots": [],
            "status": "scene_unverified",
            "input_sent": False,
            "action_eligible": False,
            "scope": "Current-frame icon and printed-price matches; read-only, no debit/cooldown/action legality.",
        }
        if state.get("scene", {}).get("value") != "playing":
            return result
        arr = card_gray(image)
        for slot in range(1, 11):
            identity = self.cards._identity(arr, slot)
            best = identity["best"]
            price = self.cards._price(arr, slot, best["dx"], best["dy"])
            ref = next((c for c in self.cards.cards if c["plant_id"] == identity["value"]), None)
            consistent = ref is not None and price["value"] is not None and price["value"] == ref["display_cost"]
            status = (
                "observed" if consistent else
                "unknown" if price["value"] is None or ref is None else
                "price_conflict"
            )
            result["slots"].append({
                "slot": slot,
                "identity": identity,
                "printed_price": price,
                "status": status,
                "plant_id": identity["value"],
                "display_cost": price["value"],
                "actual_debit_verified": False,
                "cooldown_verified": False,
                "action_eligible": False,
            })
        result["status"] = "observed"
        result["processing_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    def observe(self, image: Image.Image, **kwargs) -> dict[str, Any]:
        t0 = time.perf_counter()
        state = self.base.observe(image, **kwargs)
        state["loadout"] = self._observe_cards_in_memory(image, state)
        fields = state.get("fields", {})
        cards = fields.get("cards")
        if isinstance(cards, list) and len(cards) == 10:
            known_bd = sum(1 for c in cards if isinstance(c, dict) and c.get("visual_state") in ("bright", "dim"))
            for c in cards:
                if isinstance(c, dict):
                    raw_vs = c.get("raw_visual_state", c.get("visual_state"))
                    c["raw_visual_state"] = raw_vs
                    if known_bd >= 5 and raw_vs not in ("bright", "dim"):
                        c["visual_state"] = "dim"
        playing = state.get("scene", {}).get("value") == "playing"
        rgb_arr = np.asarray(image if image.mode == "RGB" else image.convert("RGB"), dtype=np.uint8) if playing else None
        intel = self.zombie_classifier.classify_frame(rgb_arr, state)
        if playing:
            state["zombie_intel"] = intel
            epoch = (state["run_id"], kwargs.get("tracking_epoch"))
            if epoch != self._plant_epoch:
                self.reset()
                self._plant_epoch = epoch
            plants = self._plant_classifier.classify_board(
                cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR), fields, zombie_intel=intel,
                timestamp_s=state["frame_time_seconds"],
            )
            state["plant_intel"] = plants
            defense = evaluate_all_lanes(plants, intel, fields)
            state["lane_defense"] = defense
            sun_field = fields.get("sun", {})
            sun = sun_field.get("raw_value", sun_field.get("value"))
            state["counter_plan"] = recommend_counter_plan(
                intel, state.get("loadout", {}).get("slots", []),
                sun if type(sun) is int and 0 <= sun <= 999 else None,
                raw_cards=cards, cells=fields.get("cells"), lane_defense=defense,
            )
        else:
            self.reset()
        # Preserve raw sun OCR value for truthful 3-channel display; if sun digits are transiently
        # occluded while a localized threat or sun cue is present, allow defend_lane/collect_resources
        # evaluation without blanking the live sidebar (defense/resource reviews do not depend on sun OCR).
        sun_obj = fields.get("sun")
        if isinstance(sun_obj, dict):
            raw_sun = sun_obj.get("raw_value", sun_obj.get("value"))
            sun_obj["raw_value"] = raw_sun
            has_cues = bool(state.get("resources", {}).get("sun_appearance", {}).get("candidates"))
            if type(raw_sun) is not int and (intel["priority_row"] is not None or has_cues):
                sun_obj["value"] = 0
        state["processing_ms"] = (time.perf_counter() - t0) * 1000.0
        return state


def evaluate_three_channels(
    root: Path,
    state: dict[str, Any],
    *,
    frame_index: int = 1,
) -> dict[str, Any]:
    """Separately evaluate Cards, Threats, and Positions from real single-frame evidence."""
    fields = state.get("fields", {})
    raw_cards = fields.get("cards")
    visual_cards = raw_cards or []
    sun_obj = fields.get("sun", {})
    sun_val = sun_obj.get("raw_value", sun_obj.get("value")) if isinstance(sun_obj, dict) else None
    sun_known = type(sun_val) is int and 0 <= sun_val <= 999

    cards_bright = sum(1 for c in visual_cards if isinstance(c, dict) and c.get("raw_visual_state", c.get("visual_state")) == "bright")
    cards_dim = sum(1 for c in visual_cards if isinstance(c, dict) and c.get("raw_visual_state", c.get("visual_state")) == "dim")
    cards_unknown = max(0, 10 - cards_bright - cards_dim)

    if "loadout" in state and state.get("loadout", {}).get("status") == "observed" and "continuity" in state:
        try:
            cc = card_context(root, state)
            cc_slots = cc["slots"]
        except Exception:
            cc_slots = [
                {"slot": s["slot"], "status": s["status"], "plant_id": s.get("plant_id"),
                 "printed_cost": s.get("display_cost"),
                 "observed_budget_sufficient": (sun_val >= s["display_cost"] if sun_known and type(s.get("display_cost")) is int else None)}
                for s in state["loadout"].get("slots", [])
            ]
    else:
        cc_slots = []
        for s in state.get("loadout", {}).get("slots", []):
            cost = s.get("display_cost")
            cc_slots.append({
                "slot": s.get("slot"),
                "status": s.get("status", "unknown"),
                "plant_id": s.get("plant_id"),
                "printed_cost": cost,
                "observed_budget_sufficient": (sun_val >= cost if sun_known and type(cost) is int else None),
            })

    id_obs = sum(1 for s in cc_slots if s.get("status") == "observed")
    budget_ok = sum(1 for s in cc_slots if s.get("status") == "observed" and s.get("observed_budget_sufficient") is True)
    bright_slots = {
        c.get("slot") for c in visual_cards
        if isinstance(c, dict) and c.get("raw_visual_state", c.get("visual_state")) == "bright"
    }
    ready_plants = [
        f"{s['plant_id']}({s['printed_cost']})"
        for s in cc_slots
        if s.get("status") == "observed" and s.get("observed_budget_sufficient") is True
        and s.get("slot") in bright_slots
    ][:4]

    validate_threat_context(state)
    zombies = state["zombies"]
    vehicles = state["vehicles"]
    z_cands = zombies["candidates"]
    zombie_heads = len(z_cands)
    per_lane: dict[int, list[float]] = {}
    for item in z_cands:
        per_lane.setdefault(int(item["row"]), []).append(float(item["head_xy"][0]))
    leftmost_x = int(round(min(float(c["head_xy"][0]) for c in z_cands))) if zombie_heads > 0 else None
    uw = len(zombies["uncertain_water_candidates"])
    ue = len(zombies["uncertain_edge_presence"])
    vc = len(vehicles["candidates"])
    uv = len(vehicles["uncertain_candidates"])
    confirmed_vehicles = sum(is_localized_vehicle(item) for item in vehicles["candidates"])
    uncertain_total = uw + ue + uv + (vc - confirmed_vehicles)
    intel = state.get("zombie_intel")
    if intel is None:
        intel = classify_state_metadata(state)
    for row, x in supplementary_snorkels(intel):
        per_lane.setdefault(row, []).append(x)
    plant_intel = state.get("plant_intel")
    lane_defense = state.get("lane_defense")
    counter_plan = recommend_counter_plan(
        intel, cc_slots, sun_val if sun_known else None, raw_cards=raw_cards, cells=fields.get("cells"),
        lane_defense=lane_defense,
    )
    priority_row = counter_plan["priority_row"]
    leftmost_threat_x = min((lane["leftmost_x"] for lane in intel["by_lane"].values()), default=None)

    cells = fields.get("cells") or []
    occ = sum(1 for c in cells if isinstance(c, dict) and c.get("occupancy") in ("ground_occupied", "base_only", "base_and_upper"))
    emp = sum(1 for c in cells if isinstance(c, dict) and c.get("occupancy") == "empty")
    unk = max(0, 54 - occ - emp)
    rescued = {
        (p["row"], p["column"])
        for row in (plant_intel or {}).get("cells_by_row", {}).values() for p in row
        if p.get("source") == "low_profile_scan"
    }
    rescued_empty = sum(c["occupancy"] == "empty" for c in cells
                        if isinstance(c, dict) and (c.get("row"), c.get("column")) in rescued)
    occ += len(rescued)
    emp -= rescued_empty
    unk = max(0, unk - (len(rescued) - rescued_empty))

    try:
        sun_cues = len(sun_cue_ids(state))
    except ReviewBlocked:
        sun_cues = 0

    if not intel["by_lane"]:
        threat_text = f"确认威胁=0（无确认僵尸/载具）| 水路/边缘/车辆不确定={uncertain_total}"
    else:
        threat_text = intel["summary_short"]
        if lane_defense:
            threat_text += f"｜{lane_defense['summary_zh']}"
        if priority_row is not None:
            primary = intel["by_lane"][priority_row]["primary_entity"]
            threat_text += f"｜{priority_row}路x={round(primary['x'])}"
        if counter_plan["action_text"]:
            threat_text += f"→建议：{counter_plan['action_text']}"
        if uncertain_total:
            threat_text += f" | 水路/边缘/车辆不确定={uncertain_total}"
    sun_str = str(sun_val) if sun_known else "未知"
    ready_str = f" | 可买参考：{', '.join(ready_plants)}" if ready_plants else ""
    summary_text = (
        f"【卡片】阳光={sun_str} | 亮={cards_bright}, 暗={cards_dim}, 冷却/未知={cards_unknown} | 身份一致={id_obs}/10, 够买={budget_ok}{ready_str}\n"
        f"【威胁】{threat_text}\n"
        f"【位置】54格：已占={occ}, 空={emp}, 未知={unk} | 地面阳光候选={sun_cues}"
    )
    return {
        "frame_index": frame_index,
        "sun_estimate": sun_val if sun_known else None,
        "cards_bright": cards_bright,
        "cards_dim": cards_dim,
        "cards_unknown": cards_unknown,
        "cards_identity_observed": id_obs,
        "cards_budget_sufficient": budget_ok,
        "ready_plants": ready_plants,
        "zombie_heads": zombie_heads,
        "leftmost_head_x": leftmost_x,
        "per_lane": {r: sorted(xs) for r, xs in per_lane.items()},
        "uncertain_threats": uncertain_total,
        "confirmed_vehicles": confirmed_vehicles,
        "zombie_intel": intel,
        "plant_intel": plant_intel,
        "lane_defense": lane_defense,
        "counter_plan": counter_plan,
        "species_counts": intel["species_counts"],
        "priority_row": priority_row,
        "leftmost_threat_x": round(leftmost_threat_x) if leftmost_threat_x is not None else None,
        "cells_occupied": occ,
        "cells_empty": emp,
        "cells_unknown": unk,
        "sun_cues": sun_cues,
        "summary_text": summary_text,
    }


def _economy_option(state, intel, sun):
    fields = state.get("fields", {})
    bright = {c["slot"] for c in fields.get("cards") or () if c.get("visual_state") == "bright"}
    occupied = {(p["row"], p["column"]) for row in state["plant_intel"]["cells_by_row"].values() for p in row}
    opening = set()
    for cell in fields.get("cells") or ():
        row, col = cell["row"], cell["column"]
        if cell.get("occupancy") != "empty" or col > 3 or (row, col) in occupied:
            continue
        lane = intel["by_lane"].get(row)
        if lane is None or lane["leftmost_x"] > col * 80 + 160:
            opening.add("lily_pad" if row in WATER_ROWS else "sunflower")
    if type(sun) is not int or not 0 <= sun <= 999:
        return None
    slots = state.get("loadout", {}).get("slots", [])
    for plant, name in (("sunflower", "向日葵"), ("lily_pad", "睡莲")):
        if plant in opening and any(
            s.get("status") == "observed" and s.get("plant_id") == plant and s.get("slot") in bright
            and type(s.get("display_cost")) is int and s["display_cost"] <= sun for s in slots
        ):
            return name
    return None


def evaluate_local_strategy(state: dict[str, Any], policy: dict[str, int] | None = None) -> dict[str, Any]:
    """Deterministic real-evidence strategy backend (zero external network latency)."""
    if state.get("scene", {}).get("value") != "playing":
        raise ReviewBlocked("nonplaying_scene")
    try:
        validate_threat_context(state)
    except ValueError as exc:
        raise ReviewBlocked("threat_context_invalid") from exc
    zombies, vehicles = state["zombies"], state["vehicles"]
    if zombies["uncertain_edge_presence"] or zombies["uncertain_water_candidates"] or vehicles["uncertain_candidates"]:
        raise ReviewBlocked("unlocalized_threat_context")
    if any(not is_localized_vehicle(item) for item in vehicles["candidates"]):
        raise ReviewBlocked("vehicle_position_unknown")

    fields = state.get("fields", {})
    sun_field = fields.get("sun", {})
    sun = sun_field.get("raw_value", sun_field.get("value"))
    left_warning_x = (policy or {}).get("left_warning_x", DEFAULT_LEFT_WARNING_X)
    intel = state.get("zombie_intel")
    if intel is None:
        intel = classify_state_metadata(state, left_warning_x=left_warning_x)
    else:
        intel = prioritize_zombie_intel(intel, left_warning_x)
    defense = state.get("lane_defense")
    plan = recommend_counter_plan(
        intel, state.get("loadout", {}).get("slots", []),
        sun if type(sun) is int and 0 <= sun <= 999 else None, raw_cards=fields.get("cards"),
        cells=fields.get("cells"), lane_defense=defense,
    )
    focus_row = plan["priority_row"]
    if focus_row is not None:
        primary = intel["by_lane"][focus_row]["primary_entity"]
        detail = threat_detail(primary) or primary["armor_stage"]
        if detail in ("满甲", "破损", "受损"):
            hp = f"{detail}约{primary['hp_est']}血"
        else:
            hp = f"约{primary['hp_est']}血·{detail}"
        reason = f"第{focus_row}路{primary['name_cn']}({hp},x={round(primary['x'])})：{plan['action_text']}"
        if defense is not None:
            lane = defense["by_row"][focus_row]
            reason += f"；{lane['verdict_zh']}，{lane['reason_zh']}"
            covered = defense.get("sufficient_threat_rows", ())
            if covered:
                reason += f"；{'、'.join(map(str, covered))}路已稳守"
        return {
            "strategy": {
                "source_state_id": state["state_id"],
                "goal": "defend_lane",
                "focus_row": focus_row,
                "reason": reason[:96],
                "uncertainties": ["血量按外观护甲阶段估算，未读取游戏内存"],
            }
        }
    try:
        cues = sun_cue_ids(state)
    except ReviewBlocked:
        cues = []
    if plan["status"] == "observe_unknown":
        return {"strategy": {
            "source_state_id": state["state_id"], "goal": "hold_formation", "focus_row": None,
            "reason": plan["action_hint"][:96], "uncertainties": ["部分占格待识别"],
        }}
    if plan["status"] == "all_sufficient":
        safe_cues = cues and all(lane["leftmost_x"] > left_warning_x for lane in intel["by_lane"].values())
        option = _economy_option(state, intel, sun) if not safe_cues and state.get("plant_intel") else None
        summary = defense["summary_zh"]
        if safe_cues:
            goal, reason = "collect_resources", f"受威胁路均已稳守，可优先收取场上{len(cues)}处可见阳光。"
        elif option:
            goal, reason = "develop_economy", f"受威胁路均已稳守，后排有合适空位，可补{option}发育。"
        else:
            goal, reason = "hold_formation", "受威胁路均已稳守，维持阵型并留阳应对后续波次。"
        return {"strategy": {
            "source_state_id": state["state_id"], "goal": goal, "focus_row": None,
            "reason": f"{reason}{summary}"[:96], "uncertainties": ["部署和阳光基于当前画面估算"],
        }}
    if cues:
        return {
            "strategy": {
                "source_state_id": state["state_id"],
                "goal": "collect_resources",
                "focus_row": None,
                "reason": f"场上观测到 {len(cues)} 处阳光外观候选，当前无靠左威胁，优先及时收取阳光。",
                "uncertainties": ["未核实阳光归属与可点击状态"],
            }
        }
    if type(sun) is int and sun < 150:
        return {
            "strategy": {
                "source_state_id": state["state_id"],
                "goal": "develop_economy",
                "focus_row": None,
                "reason": f"当前阳光 {sun} 且未见确认僵尸头，继续补足经济植物与前期发育。",
                "uncertainties": ["未核实即将刷出的下一波威胁"],
            }
        }
    return {
        "strategy": {
            "source_state_id": state["state_id"],
            "goal": "hold",
            "focus_row": None,
            "reason": f"当前阳光 {sun if type(sun) is int else '未知'}，暂未见明确威胁，维持阵型并留意资源。",
            "uncertainties": ["局部未知格或边缘区域仍需持续观测"],
        }
    }


@dataclass
class TimelineSegment:
    segment_id: str
    phase: str
    focus: str
    start_rel_s: float
    end_rel_s: float | None = None
    status: str = "暂定"
    transition_basis: str = ""
    events: list[str] = field(default_factory=list)
    advice_count: int = 0


def format_mmss(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


class PhaseTracker:
    """Thread-safe online tactical phase estimator & vertical timeline builder per config/tactical-priorities.json."""

    def __init__(self, config_path: Path = TACTICAL_PRIORITIES_JSON):
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        self.profiles = {p["phase"]: p for p in cfg.get("profiles", [])}
        self.uncertain_label = cfg.get("uncertain_phase", {}).get("label", "阶段待确认")
        self._lock = threading.RLock()
        self.run_index = 0
        self.run_id = ""
        self.run_started_monotonic = 0.0
        self.segments: list[TimelineSegment] = []
        self._candidate_phase: str | None = None
        self._candidate_streak = 0
        self._last_occ: int | None = None
        self.start_new_run(time.monotonic(), reason="初始对局")

    def elapsed_text(self, now: float) -> str:
        with self._lock:
            return format_mmss(max(0.0, now - self.run_started_monotonic))

    def start_new_run(self, now: float, *, reason: str = "新对局") -> str:
        with self._lock:
            self.run_index += 1
            self.run_id = f"match-{time.strftime('%H%M%S')}-{self.run_index:02d}"
            self.run_started_monotonic = now
            self.segments = []
            self._candidate_phase = None
            self._candidate_streak = 0
            self._last_occ = None
            self._open_segment(self.uncertain_label, 0.0, reason, status="暂定")
            return self.run_id

    def _focus_for(self, phase: str) -> str:
        if phase in self.profiles:
            return self.profiles[phase]["focus"]
        return "持续刷新观测，先处理明确风险"

    def _open_segment(self, phase: str, rel_s: float, basis: str, *, status: str = "暂定") -> None:
        if self.segments and self.segments[-1].end_rel_s is None:
            self.segments[-1].end_rel_s = rel_s
        seg = TimelineSegment(
            segment_id=f"{self.run_id}-seg-{len(self.segments) + 1:02d}",
            phase=phase,
            focus=self._focus_for(phase),
            start_rel_s=rel_s,
            end_rel_s=None,
            status=status,
            transition_basis=basis,
        )
        self.segments.append(seg)

    def _propose_phase(self, ch: dict[str, Any]) -> tuple[str, str]:
        occ = ch["cells_occupied"]
        unk = ch["cells_unknown"]
        heads = ch["zombie_heads"]
        if unk > 24:
            return self.uncertain_label, f"未知格过多({unk}/54)"
        if occ <= 6 and heads <= 1:
            return "开局经济期", f"已占格={occ}、确认僵尸头={heads}"
        if occ <= 15 and heads <= 3:
            return "前期发育期", f"已占格={occ}、确认僵尸头={heads}"
        if occ <= 26 and heads <= 5:
            return "中期转型期", f"已占格={occ}、确认僵尸头={heads}"
        return "后期收尾期", f"高密度/大阵型：已占格={occ}、确认僵尸头={heads}"

    def update(self, ch: dict[str, Any], now: float, *, advice_added: bool = False) -> TimelineSegment:
        with self._lock:
            rel_s = max(0.0, now - self.run_started_monotonic)
            # Detect likely match restart (occupied cells drop sharply from >=12 down to <=3)
            if self._last_occ is not None and self._last_occ >= 12 and ch["cells_occupied"] <= 3 and ch["cells_unknown"] <= 10:
                self.start_new_run(now, reason=f"检测到场地重置(占用 {self._last_occ}→{ch['cells_occupied']})")
                rel_s = 0.0
            self._last_occ = ch["cells_occupied"]

            proposed, basis = self._propose_phase(ch)
            current = self.segments[-1]
            if proposed == current.phase:
                self._candidate_phase = None
                self._candidate_streak = 0
                if current.status == "暂定" and proposed != self.uncertain_label:
                    current.status = "已确认"
            else:
                if proposed == self._candidate_phase:
                    self._candidate_streak += 1
                else:
                    self._candidate_phase = proposed
                    self._candidate_streak = 1
                # Require 2 consecutive consistent observations to switch phase (avoid flicker)
                if self._candidate_streak >= 2 or current.phase == self.uncertain_label:
                    self._open_segment(proposed, rel_s, basis, status="暂定" if proposed == self.uncertain_label else "已确认")
                    self._candidate_phase = None
                    self._candidate_streak = 0
                    current = self.segments[-1]

            if advice_added:
                current.advice_count += 1
            if ch["leftmost_head_x"] is not None and ch["leftmost_head_x"] <= 200:
                evt = f"{format_mmss(rel_s)} 靠左告警(x={ch['leftmost_head_x']})"
                if not current.events or current.events[-1] != evt:
                    current.events.append(evt)
                    current.events = current.events[-6:]
            return current

    def record_gap_event(self, now: float, label: str) -> None:
        with self._lock:
            if not self.segments:
                return
            rel_s = max(0.0, now - self.run_started_monotonic)
            evt = f"{format_mmss(rel_s)} {label}"
            seg = self.segments[-1]
            if not seg.events or seg.events[-1] != evt:
                seg.events.append(evt)
                seg.events = seg.events[-6:]

    def snapshot(self, now: float) -> dict[str, Any]:
        with self._lock:
            rel_s = max(0.0, now - self.run_started_monotonic)
            cur = self.segments[-1] if self.segments else None
            return {
                "run_id": self.run_id,
                "elapsed_text": format_mmss(rel_s),
                "current_phase": cur.phase if cur else self.uncertain_label,
                "current_focus": cur.focus if cur else "",
                "current_status": cur.status if cur else "暂定",
                "segments": [
                    {
                        "segment_id": s.segment_id,
                        "phase": s.phase,
                        "focus": s.focus,
                        "start": format_mmss(s.start_rel_s),
                        "end": "进行中" if s.end_rel_s is None else format_mmss(s.end_rel_s),
                        "status": s.status,
                        "basis": s.transition_basis,
                        "events": list(s.events),
                        "advice_count": s.advice_count,
                    }
                    for s in self.segments
                ],
            }
