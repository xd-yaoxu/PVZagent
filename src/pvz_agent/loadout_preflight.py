"""Opt-in printed-card identity/cost diagnostics over the frozen preflight.

Only two previously unknown resource checks can be resolved. Selection,
cooldown, plant legality, actual debit, STOP and latest-frame checks survive.
"""
import copy
import hashlib
import json

from .action_preflight import ActionPreflightMonitor, _finish, _bound
from .card_identity import source_run
from .strategy_review import fresh


def enrich_loadout(report, state, intent, config, catalog):
    result = copy.deepcopy(report)
    if (result.get('kind') != 'plant' or 'loadout_cost_unverified' not in result.get('unknown_codes', [])):
        return result
    try:
        if _bound(state) != result['binding'] or any(intent[k] != v for k,v in result['binding'].items()):
            raise ValueError('loadout_current_frame_changed')
        channel = state['loadout']
        expected_config = hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if (channel['schema'] != 'pvz-card-identity/1' or channel['status'] != 'observed' or
            channel['action_eligible'] is not False or channel['state_id'] != state['state_id'] or
            channel['capture_time'] != state['capture_time'] or channel['source_run_id'] != source_run(state) or
            channel['catalog_sha256'] != config['catalog_sha256'] or channel['config_sha256'] != expected_config or
            any(channel['evidence'][k] != state['evidence'][k] for k in ['image','image_sha256'])):
            raise ValueError('loadout_evidence_unbound')
        slots = channel['slots']
        if (type(slots) is not list or len(slots) != 10 or
            any(type(row['slot']) is not int for row in slots) or {row['slot'] for row in slots} != set(range(1,11))):
            raise ValueError('loadout_slots_invalid')
        item = next(row for row in slots if row['slot'] == intent['slot'])
        if item['status'] != 'observed':
            raise ValueError('loadout_identity_price_unconfirmed')
        ref = next(row for row in catalog['cards'] if row['plant_id'] == item['plant_id'])
        if (item['identity']['value'] != ref['plant_id'] or type(item['display_cost']) is not int or
            item['display_cost'] != ref['display_cost'] or item['printed_price']['value'] != item['display_cost'] or
            item['action_eligible'] is not False):
            raise ValueError('loadout_identity_price_conflict')
    except (KeyError,TypeError,ValueError,StopIteration) as error:
        code = str(error) if type(error) is ValueError and str(error).startswith('loadout_') else 'loadout_evidence_unbound'
        result['checks'].append(dict(layer='resources',code=code,status='unknown',
                                     message='本帧卡片身份与价格证据未完整确认；继续保留资源未知项。',observed=None))
        return _finish(result)
    result['checks'] = [check for check in result['checks'] if check['code'] not in ['loadout_cost_unverified','budget_unverified']]
    result['checks'].append(dict(layer='resources',code='loadout_identity_price_observed',status='passed',
        message='本帧图标和卡面价格分别匹配，绑定当前局来源；未验证实际扣费。',
        observed=dict(slot=item['slot'],plant_id=item['plant_id'],display_cost=item['display_cost'],
                      catalog_sha256=channel['catalog_sha256'])))
    sun = state['fields']['sun']['value']
    known = type(sun) is int and 0 <= sun <= 999999
    sufficient = sun >= item['display_cost'] if known else None
    result['checks'].append(dict(layer='resources',code='observed_budget_sufficient' if sufficient is True else
        'observed_budget_insufficient' if sufficient is False else 'budget_unverified',
        status='passed' if sufficient is True else 'blocked' if sufficient is False else 'unknown',
        message='仅比较当前阳光读数与已读卡面价格；冷却、选中状态和种植合法性仍需单独证据。',
        observed=dict(sun=sun if known else None,display_cost=item['display_cost'])))
    result['loadout_scope'] = 'Current printed icon/price observations only; no actual debit or execution authority.'
    return _finish(result)


class LoadoutPreflightMonitor(ActionPreflightMonitor):
    def __init__(self, *, config, catalog, **kwargs):
        super().__init__(**kwargs)
        self.config, self.catalog = copy.deepcopy(config), copy.deepcopy(catalog)
        raw = (self.root / self.config['catalog']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.config['catalog_sha256'] or json.loads(raw) != self.catalog:
            raise ValueError('Preflight catalog mismatch')

    def review(self, intent):
        report = super().review(intent)
        try:
            current = self.current_state(self.clock())
            report = enrich_loadout(report, current, intent, self.config, self.catalog)
            later = self.current_state(self.clock())
            same = _bound(current) == _bound(later) and fresh(later,self.clock())
            stop = self.stop_requested()
        except Exception:
            same,stop = False,None
        if report['reviewed_state_id'] is not None and not same:
            report['checks'].append(dict(layer='freshness',code='state_changed_during_loadout_review',status='blocked',
                                         message='卡片复核期间当前状态失效。',observed=None))
        if stop is not False:
            report['checks'].append(dict(layer='control',code='loadout_review_stop',status='blocked',
                                         message='停止仍锁定或状态不确定。',observed=None))
        return _finish(report)
