"""Estimate each lane's defense from confirmed plants and localized threats."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite

from .plant_intel import SHOOTER_DPS, compact_plants, entity_x, lane_entities
from .zombie_intel import PLANT_CATALOG, WATER_ROWS, get_zombie_profile, is_plant_legal_for_row

_SAFE_RATIO = 1.25
_TIGHT_RATIO = 0.85
_BITE_HP_S = 100.0
_SHOT_MARGIN_S = 1.0
_SPEED = {"flag_zombie": 21.0, "zomboni": 20.0, "snorkel_zombie": 19.0}
_CORE = (*SHOOTER_DPS, "sunflower")
_BLOCKERS = ("wallnut", "lily_pad")
_VERDICT_ZH = {"clear": "无威胁", "sufficient": "稳守", "tight": "吃紧", "deficit": "缺口", "unknown": "未知"}


@dataclass(frozen=True)
class _Plant:
    kind: str
    x: float
    hp: float
    armed: bool


@dataclass(frozen=True)
class _Threat:
    kind: str
    x: float
    hp: float
    speed: float
    submerged: bool
    biting: bool
    urgency: float

    def arrival(self, x, reach=0):
        return max(0.0, (self.x - x - reach) / self.speed)


def _number(value, default):
    return float(value) if isinstance(value, (int, float)) and isfinite(value) else default


def _plant(record):
    kind = record.get("plant_id", "unknown_plant")
    column = _number(record.get("column"), None)
    x = _number(record.get("x_px", record.get("x")), column * 80 if column is not None else None)
    if x is None:
        return None
    hp = {"wallnut": 2400, "lily_pad": 300}.get(kind, 0)
    hp = max(0.0, min(3600 if kind == "wallnut" else hp, _number(record.get("blocker_hp"), hp)))
    return _Plant(kind, x, hp, record.get("armed") is True)


def _legal_plant_record(row, record):
    kind = record.get("plant_id", "unknown_plant")
    occupancy = record.get("occupancy")
    carrier = record.get("carrier")
    if not is_plant_legal_for_row(kind, row):
        return False
    if row not in WATER_ROWS:
        return carrier != "lily_pad" and occupancy not in ("base_only", "base_and_upper")
    if kind == "lily_pad":
        return occupancy in (None, "base_only") and carrier is None
    if occupancy == "base_only":
        return False
    terrain = PLANT_CATALOG.get(kind, {}).get("terrain")
    if terrain == "land_or_lily_pad" and occupancy == "ground_occupied":
        return False
    if terrain == "water_only" and (carrier == "lily_pad" or occupancy == "base_and_upper"):
        return False
    return True


def _threat(record):
    x = _number(entity_x(record), None)
    if x is None:
        return None
    kind = record.get("species_id") or record.get("species_hint") or "basic_zombie"
    profile = get_zombie_profile(kind)
    hp = _number(record.get("hp_est"), profile.hp_total)
    if hp <= 0:
        return None
    stance = record.get("stance") or record.get("water_pose") or record.get("pose")
    submerged = kind == "snorkel_zombie" and stance not in ("eating", "water_eating", "surfaced", "emerged")
    urgency = _number(record.get("entity_urgency", record.get("threat_weight")), profile.threat_weight)
    return _Threat(kind, x, hp, _SPEED.get(kind, 18.0), submerged,
                   stance in ("eating", "water_eating", "biting"), urgency)


def _trap_reach(plant, zombie, water):
    vehicle = zombie.kind == "zomboni"
    if plant.kind == "spikeweed" and not water:
        return 25
    if plant.kind == "tangle_kelp" and water:
        return 60
    if plant.kind == "squash" and not zombie.submerged:
        return 55
    if plant.kind == "potato_mine" and not water and (plant.armed or vehicle):
        return 25 if vehicle else 30
    return None


def _apply_traps(plants, threats, water):
    """Walk through projected contacts in time order, consuming each trap once."""
    hp = [z.hp for z in threats]
    events = []
    for pi, plant in enumerate(plants):
        for zi, z in enumerate(threats):
            reach = _trap_reach(plant, z, water)
            if reach is not None and plant.x <= z.x + reach:
                events.append((z.arrival(plant.x, reach), zi, -plant.x, pi))
    consumed = set()
    killed_at = {}
    removed_hp = 0.0
    for at, zi, _, pi in sorted(events):
        if pi in consumed or hp[zi] <= 0:
            continue
        plant, z = plants[pi], threats[zi]
        vehicle = z.kind == "zomboni"
        if plant.kind == "spikeweed" and not vehicle:
            damage = 85
        else:
            consumed.add(pi)
            if plant.kind == "potato_mine":
                if vehicle:
                    continue
                # A vehicle destroys the mine on contact. The explosion hits
                # nearby walkers that have also reached the trigger vicinity.
                for other, neighbor in enumerate(threats):
                    if (other != zi and neighbor.kind != "zomboni" and hp[other] > 0
                            and abs(neighbor.x - z.x) <= 40
                            and abs(neighbor.x - at * neighbor.speed - (z.x - at * z.speed)) <= 40):
                        dealt = min(hp[other], 1800)
                        hp[other] -= dealt
                        removed_hp += dealt
                damage = 1800
            else:
                damage = 1800 if plant.kind == "squash" else hp[zi]
        dealt = min(hp[zi], damage)
        hp[zi] -= dealt
        removed_hp += dealt
        if hp[zi] <= 0:
            killed_at[zi] = plant.x
    return hp, removed_hp, consumed, killed_at


def _stall_time(plant, zombie, walkers):
    # Integrate 100 HP/s per biter as successive walkers reach this blocker.
    # Include leading walkers conservatively even if they may die to fire.
    joins = sorted(other.arrival(plant.x) for other in walkers if plant.x <= other.x + 20)
    time_s, remaining, bite_rate = joins[0], plant.hp, _BITE_HP_S
    for join in joins[1:]:
        used = (join - time_s) * bite_rate
        if used >= remaining:
            break
        remaining -= used
        time_s = join
        bite_rate += _BITE_HP_S
    return max(0.0, time_s + remaining / bite_rate - zombie.arrival(plant.x))


def _fire_window(zombie, defense_x, plants, ahead, walkers):
    if zombie.kind == "zomboni":
        # The first solid plant is already at risk when the vehicle reaches it.
        front = max((p.x for p in plants if p.kind in (*_CORE, "wallnut", "potato_mine")
                     and p.x <= zombie.x + 20), default=defense_x)
        return 0.0, max(0.0, zombie.arrival(max(front, defense_x)) - _SHOT_MARGIN_S), 0.0

    if zombie.submerged:
        # Only exposure at a confirmed, expendable obstacle counts as safe fire.
        if not ahead:
            return 0.0, 0.0, 0.0
        first = ahead[0]
        start = zombie.arrival(first.x)
        stall = _stall_time(first, zombie, walkers)
        return start, start + max(0.0, stall - _SHOT_MARGIN_S), stall
    stall = sum(_stall_time(p, zombie, walkers) for p in ahead)
    return 0.0, max(0.0, zombie.arrival(defense_x) + stall - _SHOT_MARGIN_S), stall


def evaluate_lane_defense(row, plants, threats, *, defense_line_x=140.0,
                          unknown_occupied=0, plants_compact_zh=None) -> dict:
    """Evaluate one lane. Inputs are plant cells and zombie-intel entity records."""
    plants = list(plants)
    records = [record for record in plants if _legal_plant_record(row, record)]
    counts = Counter(record.get("plant_id", "unknown_plant") for record in records)
    unknown_occupied = max(unknown_occupied, counts["unknown_plant"])
    if plants_compact_zh is None or len(records) != len(plants):
        plants_compact_zh = compact_plants(counts)
    known = sorted((plant for r in records if (plant := _plant(r)) is not None
                    and plant.kind != "unknown_plant"), key=lambda p: -p.x)
    zombies = sorted((z for r in threats if (z := _threat(r)) is not None), key=lambda z: z.x)
    initial_hp = sum(z.hp for z in zombies)
    blocker_hp = sum(p.hp for p in known)
    defense_x = max(defense_line_x, max((p.x for p in known if p.kind in _CORE), default=0))
    verdict, reason = "clear", "当前无确认威胁"
    ratio = damage = effective_dps = stall_seconds = 0.0
    if zombies:
        hp, removed_hp, consumed, killed_at = _apply_traps(known, zombies, row in WATER_ROWS)
        active = [(z, hp[i]) for i, z in enumerate(zombies) if hp[i] > 0]
        if not active:
            verdict, reason = "sufficient", "现有陷阱可处理本路全部威胁"
            ratio, damage = 1.0, removed_hp
        else:
            # A surviving vehicle, or one neutralized behind a blocker, can
            # crush that blocker before it contributes a reliable bite window.
            vehicles = [(z, killed_at.get(i, -float("inf"))) for i, z in enumerate(zombies)
                        if z.kind == "zomboni"]
            remaining = [p for i, p in enumerate(known) if i not in consumed]
            blockers = [p for p in remaining if p.kind in _BLOCKERS and p.hp > 0
                        and not any(stop < p.x <= z.x + 20 for z, stop in vehicles)]
            walkers = [z for z, _ in active if z.kind != "zomboni"]
            fire_used_until = 0.0
            capacity = 0.0
            weakest = float("inf")
            uncovered = None
            plant_at_risk = False
            for z, remaining_hp in active:
                ahead = [p for p in blockers if defense_x <= p.x <= z.x + 20]
                start, end, stall = _fire_window(z, defense_x, remaining, ahead, walkers)
                stall_seconds = max(stall_seconds, stall)
                # Advance the shot position to the exposure point for a diving
                # target. A shooter it has already passed cannot hit it there.
                target_x = z.x - start * z.speed
                dps = sum(SHOOTER_DPS[p.kind] for p in remaining
                          if p.kind in SHOOTER_DPS and p.x <= target_x + 20)
                if end > start:
                    effective_dps = max(effective_dps, dps)
                start = max(start, fire_used_until)
                available = dps * max(0.0, end - start)
                target_ratio = available / remaining_hp
                weakest = min(weakest, target_ratio)
                # Spend each firing second only once and reserve the same
                # safety margin per target when assessing the queue behind it.
                if dps > 0 and end > start:
                    fire_used_until = start + min(end - start, _SAFE_RATIO * remaining_hp / dps)
                capacity += available if len(active) == 1 else min(available, _SAFE_RATIO * remaining_hp)
                if z.kind == "zomboni" and available < 1350 * _SAFE_RATIO:
                    uncovered = "剩余冰车缺少地刺、窝瓜或足够的碾压前火力"
                elif z.submerged and end == 0:
                    uncovered = "潜伏潜水缺少海草或可迫其露头的阻挡"
                elif available == 0 and uncovered is None:
                    uncovered = "剩余威胁缺少有效直射火力"
                front = ahead[0].x if ahead else defense_x
                plant_at_risk |= (z.x <= front + 70 and (stall == 0 or target_ratio < _SAFE_RATIO))
                plant_at_risk |= any(p.kind in _CORE and (
                    z.x < p.x - 20 or (z.biting and not ahead and z.x <= p.x + 35)
                ) for p in known)

            ratio = capacity / max(1.0, sum(hp))
            damage = removed_hp + capacity
            if uncovered or weakest < _TIGHT_RATIO:
                verdict, reason = "deficit", uncovered or "有效伤害不足以覆盖剩余血量"
            elif weakest < _SAFE_RATIO or plant_at_risk:
                verdict, reason = "tight", "威胁已逼近防线" if plant_at_risk else "剩余威胁的火力余量不足"
            else:
                verdict, reason = "sufficient", "已确认火力与阻挡足以处理剩余威胁"
            if verdict in ("tight", "deficit") and unknown_occupied:
                verdict, reason = "unknown", f"{reason}，另有{unknown_occupied}个占格待识别"

    return {
        "verdict": verdict, "verdict_zh": _VERDICT_ZH[verdict],
        "coverage_ratio": round(ratio, 3), "threat_hp": initial_hp,
        "estimated_damage": round(damage, 2), "effective_dps": effective_dps,
        "blocker_hp": blocker_hp, "stall_seconds": round(stall_seconds, 3),
        "reason_zh": reason, "plants_compact_zh": plants_compact_zh,
        "needs_reinforcement": verdict in ("tight", "deficit"),
    }


def evaluate_all_lanes(plant_intel, zombie_intel, fields=None) -> dict:
    """Combine lane assessments using the recognizers' row-indexed dictionaries."""
    cells_by_row = plant_intel.get("cells_by_row", {})
    row_summary = plant_intel.get("row_summary", {})
    threats_by_row = {row: [] for row in range(1, 7)}
    for row, z in lane_entities(zombie_intel):
        if row in threats_by_row:
            threats_by_row[row].append(z)
    # Occupancy can corroborate an unclassified plant. A bare "unknown"
    # occupancy does not establish that a plant exists.
    occupied = {row: set() for row in threats_by_row}
    for cell in (fields or {}).get("cells", ()) or ():
        row = cell.get("row")
        if row not in WATER_ROWS and cell.get("occupancy") == "base_only":
            continue
        if row in occupied and (
            cell.get("occupied") is True or cell.get("occupancy") in ("ground_occupied", "base_only", "base_and_upper")
        ):
            occupied[row].add(cell.get("column"))
    by_row = {}
    for row, threats in threats_by_row.items():
        plants = cells_by_row.get(row, cells_by_row.get(str(row), ()))
        summary = row_summary.get(row, row_summary.get(str(row), {}))
        unseen = occupied[row] - {p.get("column") for p in plants}
        unknown = max(summary.get("unknown_occupied", 0), sum(
            p.get("plant_id") == "unknown_plant" for p in plants
        )) + len(unseen)
        by_row[row] = evaluate_lane_defense(
            row, plants, threats, unknown_occupied=unknown, plants_compact_zh=summary.get("compact_zh"),
        )

    needs = [row for row, lane in by_row.items() if lane["needs_reinforcement"]]
    zombie_lanes = zombie_intel.get("by_lane", {}) if isinstance(zombie_intel, dict) else {}

    def priority(row):
        lane = zombie_lanes.get(row, zombie_lanes.get(str(row), {}))
        zombies = [z for rec in threats_by_row[row] if (z := _threat(rec)) is not None]
        urgency = _number(lane.get("lane_urgency"), max(z.urgency for z in zombies))
        return (by_row[row]["verdict"] != "deficit", -urgency,
                by_row[row]["coverage_ratio"], min(z.x for z in zombies), row)

    needs.sort(key=priority)
    sufficient = [row for row, lane in by_row.items() if lane["verdict"] == "sufficient"]
    unknown = [row for row, lane in by_row.items() if lane["verdict"] == "unknown"]
    threatened = [row for row, lane in by_row.items() if lane["verdict"] != "clear"]
    summary = "；".join(
        f"{row}路 {by_row[row]['plants_compact_zh'] or '无已确认部署'}，{by_row[row]['verdict_zh']}"
        for row in threatened
    )
    return {
        "by_row": by_row, "needs_defense_rows": needs, "sufficient_threat_rows": sufficient,
        "unknown_threat_rows": unknown, "all_threatened_sufficient": bool(threatened) and sufficient == threatened,
        "summary_zh": summary or "各路当前无确认威胁",
    }
