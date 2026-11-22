"""Frame-local icon and printed-price evidence, independent of slot semantics.

Closed experimental catalog. No cooldown, actual debit or planting legality is
inferred. Price templates are matched separately; a mismatch cannot silently
inherit the catalog price. No session cache survives a changed frame.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_run(state):
    continuity = state.get('continuity', {})
    return continuity.get('source_run_id', state['run_id'])


def gray(image):
    return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _crop(array, slot, roi):
    x = 90 + (slot - 1) * 51
    a, b, c, d = roi
    return array[b:d, x+a:x+c].copy()


class CardIdentityPerception:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = copy.deepcopy(config)
        raw = (self.root / config['catalog']).read_bytes()
        if digest(raw) != config['catalog_sha256']:
            raise ValueError('Catalog changed')
        self.catalog = json.loads(raw)
        source = self.catalog['reference']
        data = (self.root / source['image']).read_bytes()
        if digest(data) != source['image_sha256']:
            raise ValueError('Reference image changed')
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (800, 600): raise ValueError('Canonical reference required')
            array = gray(image)
        self.cards = self.catalog['cards']
        if len(self.cards) != 10 or len({c['plant_id'] for c in self.cards}) != 10:
            raise ValueError('Ten distinct reviewed references required')
        self.icons = [_crop(array, c['reference_slot'], config['icon_roi']) for c in self.cards]
        self.prices = [_crop(array, c['reference_slot'], config['price_roi']) for c in self.cards]
        self.config_sha256 = digest(json.dumps(config, sort_keys=True, allow_nan=False).encode())

    def _identity(self, array, slot):
        cfg = self.config
        x = 90 + (slot - 1) * 51
        left, top, right, bottom = cfg['icon_roi']
        radius = cfg['search_radius']
        patch = array[top-radius:bottom+radius, x+left-radius:x+right+radius]
        ranked = []
        for card, template in zip(self.cards, self.icons):
            matches = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (dx, dy) = cv2.minMaxLoc(matches)
            ranked.append(dict(plant_id=card['plant_id'], display_name=card['display_name'],
                               score=float(score), dx=dx-radius, dy=dy-radius,
                               reference_slot=card['reference_slot']))
        ranked.sort(key=lambda row: row['score'], reverse=True)
        best, runner = ranked[:2]
        accepted = (best['score'] >= cfg['icon_similarity_min'] and
                    best['score'] - runner['score'] >= cfg['identity_margin_min'] and
                    abs(best['dx']) < radius and abs(best['dy']) < radius)
        return dict(value=best['plant_id'] if accepted else None,
                    display_name=best['display_name'] if accepted else None,
                    best=best, runner_up_score=runner['score'],
                    reason=None if accepted else 'icon_low_similarity_ambiguous_or_boundary')

    def _price(self, array, slot, dx, dy):
        left, top, right, bottom = self.config['price_roi']
        x = 90 + (slot - 1) * 51
        patch = array[top+dy:bottom+dy, x+left+dx:x+right+dx]
        values = {}
        for card, template in zip(self.cards, self.prices):
            score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            value = card['display_cost']
            values[value] = max(values.get(value, -1.), score)
        ranked = sorted(values.items(), key=lambda row: row[1], reverse=True)
        (price, best), (_, runner) = ranked[:2]
        accepted = best >= self.config['price_similarity_min'] and best-runner >= self.config['price_margin_min']
        return dict(value=price if accepted else None, similarity=best, runner_up_similarity=runner,
                    reason=None if accepted else 'printed_price_unknown_or_ambiguous')

    def observe(self, state):
        start = time.perf_counter()
        evidence = state['evidence']
        result = dict(schema='pvz-card-identity/1', state_id=state['state_id'], capture_time=state['capture_time'],
                      source_run_id=source_run(state), evidence=copy.deepcopy(evidence),
                      catalog_sha256=self.config['catalog_sha256'], config_sha256=self.config_sha256,
                      slots=[], status='scene_unverified', input_sent=False, action_eligible=False,
                      scope='Current-frame icon and printed-price matches in a ten-type experimental catalog; no actual debit, cooldown or action legality.')
        if state.get('scene', {}).get('value') != 'playing': return result
        path = (self.root / evidence['image']).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or path.stat().st_size > 8_000_000:
            raise ValueError('Image must be a bounded project file')
        data = path.read_bytes()
        if digest(data) != evidence['image_sha256']: raise ValueError('Image evidence changed')
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or image.size != (800, 600): raise ValueError('Canonical PNG required')
            array = gray(image)
        for slot in range(1, 11):
            identity = self._identity(array, slot)
            best = identity['best']
            price = self._price(array, slot, best['dx'], best['dy'])
            reference = next((c for c in self.cards if c['plant_id'] == identity['value']), None)
            consistent = reference is not None and price['value'] is not None and price['value'] == reference['display_cost']
            result['slots'].append(dict(slot=slot, identity=identity, printed_price=price,
                                        status='observed' if consistent else 'unknown' if price['value'] is None or reference is None else 'price_conflict',
                                        plant_id=identity['value'], display_cost=price['value'],
                                        actual_debit_verified=False, cooldown_verified=False, action_eligible=False))
        result.update(status='observed', processing_ms=(time.perf_counter()-start)*1000)
        return result


class CardIdentityStream:
    """Add a channel without changing any existing field/object/strategy result."""
    def __init__(self, base, cards):
        self.base, self.cards = base, cards

    def observe(self, image, **kwargs):
        state = self.base.observe(image, **kwargs)
        started = time.perf_counter()
        state['loadout'] = self.cards.observe(state)
        state['processing_ms'] += (time.perf_counter()-started)*1000
        return state
