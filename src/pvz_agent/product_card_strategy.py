"""Opt-in current-frame card evidence for the existing strategic request.

This carries observations, never actions or readiness assertions. Invalid channel
bindings fail closed; unconfirmed individual slots remain unknown.
"""
import hashlib
import json
from pathlib import Path
import re

from .card_identity import source_run
from .strategy_profile import recommend_profiled


def card_context(root, state):
    root = Path(root).resolve()
    config = json.loads((root / 'config/card-identity-v1.json').read_text(encoding='utf-8'))
    catalog_path = (root / config['catalog']).resolve()
    if not catalog_path.is_relative_to(root):
        raise ValueError('card_catalog_outside_project')
    catalog_bytes = catalog_path.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != config['catalog_sha256']:
        raise ValueError('card_catalog_changed')
    catalog = json.loads(catalog_bytes)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    try:
        if type(state) is not dict:
            raise ValueError('card_state_invalid')
        channel = state['loadout']; fields = state['fields']; evidence = state['evidence']
        if (type(channel) is not dict or type(fields) is not dict or type(evidence) is not dict
                or type(state.get('continuity')) is not dict
                or type(state.get('scene')) is not dict or type(fields.get('scene')) is not dict
                or type(fields.get('sun')) is not dict or type(channel.get('evidence')) is not dict):
            raise ValueError('card_channel_invalid')
        def bounded_text(value):
            return type(value) is str and 0 < len(value) <= 256
        if (not all(bounded_text(state.get(key)) for key in ('state_id','run_id','capture_time'))
                or not bounded_text(source_run(state)) or not bounded_text(evidence.get('image'))
                or type(evidence.get('image_sha256')) is not str
                or re.fullmatch(r'[0-9a-f]{64}', evidence['image_sha256']) is None):
            raise ValueError('card_source_invalid')
        if (type(channel) is not dict or channel['schema'] != 'pvz-card-identity/1'
                or channel['status'] != 'observed' or channel['action_eligible'] is not False
                or channel['input_sent'] is not False or state['scene']['value'] != 'playing'
                or channel['state_id'] != state['state_id'] or channel['capture_time'] != state['capture_time']
                or channel['source_run_id'] != source_run(state)
                or channel['catalog_sha256'] != config['catalog_sha256'] or channel['config_sha256'] != config_hash
                or any(channel['evidence'][k] != evidence[k] for k in ('image', 'image_sha256'))
                or fields['state_id'] != state['state_id'] or fields['capture_time'] != state['capture_time']
                or fields['scene']['value'] != 'playing'):
            raise ValueError('card_channel_unbound')
        slots = channel['slots']
        if (type(slots) is not list or len(slots) != 10
                or any(type(row) is not dict or type(row.get('slot')) is not int for row in slots)
                or {row['slot'] for row in slots} != set(range(1, 11))):
            raise ValueError('card_slots_invalid')
        cards = catalog['cards']
        if type(cards) is not list or len(cards) != 10:
            raise ValueError('card_catalog_invalid')
        refs = {c['plant_id']: c for c in cards}
        if len(refs) != 10:
            raise ValueError('card_catalog_invalid')
        sun = fields['sun']['value']
        if type(sun) is not int or not 0 <= sun <= 999999:
            sun = None
        result = []
        for row in sorted(slots, key=lambda item: item['slot']):
            item = dict(slot=row['slot'], status='unknown', plant_id=None, printed_cost=None,
                        observed_budget_sufficient=None)
            plant = row.get('plant_id'); cost = row.get('display_cost')
            ref = refs.get(plant) if type(plant) is str else None
            identity = row.get('identity'); price = row.get('printed_price')
            if (row.get('status') == 'observed' and ref is not None and type(cost) is int
                    and type(ref.get('display_cost')) is int and cost == ref['display_cost']
                    and type(identity) is dict and identity.get('value') == plant
                    and type(price) is dict and type(price.get('value')) is int and price['value'] == cost
                    and row.get('action_eligible') is False):
                item.update(status='observed', plant_id=plant, printed_cost=cost,
                            observed_budget_sufficient=(sun >= cost if sun is not None else None))
            result.append(item)
        return dict(source_state_id=state['state_id'], source_run_id=source_run(state),
                    capture_time=state['capture_time'], image_sha256=evidence['image_sha256'],
                    sun_estimate=sun, slots=result)
    except (KeyError, TypeError, AttributeError):
        raise ValueError('card_channel_invalid') from None


class _EvidenceProvider:
    """Append context while retaining the actual profiled provider and schema."""
    def __init__(self, provider, context):
        self.provider, self.context = provider, context
        self.clock, self.profiles = provider.clock, provider.profiles

    def complete(self, prompt, **kwargs):
        suffix = ('\n本帧本地卡面和印刷价格估计如下，仅可把 observed 的身份和价格作为观察；'
                  'unknown 一律未知，不得根据图片或槽号补猜。预算比较仅为估计，'
                  '卡片亮度不代表就绪，冷却、选择状态及种植合法性未知；没有动作授权。\n'
                  + json.dumps(self.context, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
        return self.provider.complete(prompt + suffix, **kwargs)


def recommend_product_card_strategy(provider, root, state):
    context = card_context(root, state)
    result = recommend_profiled(_EvidenceProvider(provider, context), root, state)
    result['card_evidence'] = context
    result['card_evidence_scope'] = ('Request source frame only; later goal review does not recheck card identity, '
                                     'price, budget or readiness. New actions require current-frame preflight.')
    return result
