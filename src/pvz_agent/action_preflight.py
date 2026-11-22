"""Read-only action preconditions over current screenshot estimates.

This is a diagnostic contract, NOT an executor gate or input permission. Missing
selection, plant identity, cooldown, legality and interactability detectors stay
unknown. Model intents cannot supply assertions that make those checks pass.
"""
import copy
import hashlib
import io
from pathlib import Path
import re
import time

from PIL import Image

from .resource_review import sun_cue_ids, ReviewBlocked
from .strategy_review import fresh


LAYERS = {
    'control': '停止与运行状态', 'proposal': '动作意图',
    'freshness': '时效与连续性', 'evidence': '截图证据',
    'scene': '对局与选择状态', 'resources': '阳光与卡片',
    'target': '目标与格子', 'legality': '动作合法性',
}
COMMON = {'intent_id', 'kind', 'source_state_id', 'source_run_id',
          'source_image_sha256', 'source_generation', 'source_epoch'}
TARGETS = {'plant': {'slot', 'row', 'column'},
           'collect': {'cue_id'}, 'shovel': {'row', 'column', 'layer'}}


def _text(value):
    return type(value) is str and 0 < len(value) <= 256


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _mapping(value):
    return value if type(value) is dict else {}


def validate_intent(value):
    if type(value) is not dict or type(value.get('kind')) is not str:
        raise ValueError('intent_schema_invalid')
    kind = value['kind']
    if kind not in TARGETS or set(value) != COMMON | TARGETS[kind]:
        raise ValueError('intent_schema_invalid')
    if not all(_text(value[k]) for k in ('intent_id', 'source_state_id', 'source_run_id')):
        raise ValueError('intent_schema_invalid')
    if (type(value['source_image_sha256']) is not str or
            re.fullmatch('[0-9a-f]{64}', value['source_image_sha256']) is None):
        raise ValueError('intent_schema_invalid')
    if not all(_integer(value[k], 0, 2**63 - 1) for k in ('source_generation', 'source_epoch')):
        raise ValueError('intent_schema_invalid')
    if kind in ('plant', 'shovel') and not (
            _integer(value['row'], 1, 6) and _integer(value['column'], 1, 9)):
        raise ValueError('intent_target_invalid')
    if kind == 'plant' and not _integer(value['slot'], 1, 10):
        raise ValueError('intent_target_invalid')
    if kind == 'shovel' and value['layer'] not in ('upper', 'carrier'):
        raise ValueError('intent_target_invalid')
    if kind == 'collect' and not _text(value['cue_id']):
        raise ValueError('intent_target_invalid')
    return copy.deepcopy(value)


def frame_binding(state):
    """Use frame, image, generation AND session epoch; never just coordinates."""
    return dict(source_state_id=state['state_id'], source_run_id=state['run_id'],
                source_image_sha256=state['evidence']['image_sha256'],
                source_generation=state['readonly_runtime']['generation'],
                source_epoch=state['continuity']['epoch'])


def _bound(state):
    binding = frame_binding(state)
    # Validate the binding with the same strict types as an intent.
    validate_intent(dict(binding, intent_id='binding-check', kind='collect', cue_id='check'))
    if (type(state['continuity']['capture_epoch']) is not int or
            state['continuity']['capture_epoch'] != binding['source_epoch']):
        raise ValueError('continuity_binding_invalid')
    if not _text(state['capture_time']):
        raise ValueError('capture_time_invalid')
    return binding


def _image_matches(root, state):
    root = Path(root).resolve()
    raw = state['evidence']['image']
    if not _text(raw):
        return False
    path = (root / raw).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 8_000_000:
        return False
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != state['evidence']['image_sha256']:
        return False
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG' or image.size != (800, 600):
            return False
        image.verify()
    return True


def _unique(rows, **keys):
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        return None
    selected = [row for row in rows if all(type(row.get(k)) is type(v) and row.get(k) == v
                                           for k, v in keys.items())]
    return selected[0] if len(selected) == 1 else None


def _finish(report):
    report['blocking_codes'] = [c['code'] for c in report['checks'] if c['status'] == 'blocked']
    report['unknown_codes'] = [c['code'] for c in report['checks'] if c['status'] == 'unknown']
    report['disposition'] = ('blocked' if report['blocking_codes'] else
                             'incomplete' if report['unknown_codes'] else 'observed_checks_match')
    return report


def inspect_intent(intent, state, *, root, now, stopped):
    """Pure diagnostic entry, also suitable for explicitly labelled archive replay."""
    report = dict(schema='pvz-action-preflight/1', mode='diagnostic_only', checks=[],
                  intent_id=None, kind=None, reviewed_state_id=None, evidence=None,
                  input_sent=False, action_eligible=False, execution_connected=False)

    def check(layer, code, status, message, observed=None):
        report['checks'].append(dict(layer=layer, code=code, status=status,
                                     message=message, observed=observed))

    check('control', 'stop_latched' if stopped is True else 'stop_state_unknown' if stopped is not False else 'stop_clear',
          'passed' if stopped is False else 'blocked',
          'STOP/F8只停止Agent输入；不会发送游戏暂停键。', stopped if type(stopped) is bool else None)
    try:
        intent = validate_intent(intent)
    except ValueError as exc:
        check('proposal', str(exc), 'blocked', '意图只接受绑定到来源帧的槽位/格子/候选ID；不接受自报通过标志或坐标。')
        return _finish(report)
    report.update(intent_id=intent['intent_id'], kind=intent['kind'])
    check('proposal', 'intent_valid', 'passed', '意图字段和目标范围有效。')
    if type(state) is not dict or not fresh(state, now):
        check('freshness', 'current_snapshot_unavailable', 'blocked', '没有当前可发布且不超过2秒的快照。')
        return _finish(report)
    try:
        binding = _bound(state)
    except (KeyError, TypeError, ValueError):
        check('freshness', 'continuity_binding_invalid', 'blocked', '当前快照缺少有效连续性或证据绑定。')
        return _finish(report)
    report.update(reviewed_state_id=state['state_id'], capture_time=state['capture_time'],
                  evidence=copy.deepcopy(state['evidence']), binding=binding)
    if any(intent[k] != value for k, value in binding.items()):
        check('freshness', 'intent_source_changed', 'blocked', '当前帧、图像、代次或会话已变化，必须重新提出意图。')
        return _finish(report)
    check('freshness', 'current_frame_bound', 'passed', '意图与当前快照一致；历史战略有效期不能延长动作时效。')
    try:
        valid_image = _image_matches(root, state)
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        valid_image = False
    if not valid_image:
        check('evidence', 'image_evidence_invalid', 'blocked', '项目内原始800×600 PNG不存在、损坏或哈希不匹配。')
        return _finish(report)
    check('evidence', 'image_evidence_verified', 'passed', '原图路径、尺寸及SHA256校验通过。')
    if _mapping(state.get('scene')).get('value') != 'playing':
        check('scene', 'scene_not_playing', 'blocked', '当前画面未确认处于对局。')
        return _finish(report)
    check('scene', 'playing_observed', 'passed', '截图分类为对局；这不单独证明不存在遮挡或选中光标。')
    check('scene', 'selection_state_unverified', 'unknown', '尚未识别卡片/铲子选中状态，不能把一次点击解释成预期动作。')
    check('scene', 'target_occlusion_unverified', 'unknown', '尚未完成动作目标局部遮挡与交互区域验证。')
    if intent['kind'] == 'collect':
        try:
            ids = sun_cue_ids(state)
        except ReviewBlocked:
            check('target', 'sun_evidence_unknown', 'unknown', '阳光外观通道缺失或证据未绑定到当前帧。')
        else:
            present = intent['cue_id'] in ids
            check('target', 'sun_candidate_present' if present else 'sun_target_unconfirmed',
                  'passed' if present else 'blocked',
                  '检查当前帧是否仍包含指定阳光外观；未确认不等于已拾取或已消失。', intent['cue_id'])
        check('legality', 'sun_interactable_unverified', 'unknown', '外观候选还不是已验证的可拾取目标。')
        return _finish(report)

    fields = state.get('fields')
    if (type(fields) is not dict or fields.get('state_id') != state['state_id'] or
            fields.get('capture_time') != state['capture_time'] or
            _mapping(fields.get('scene')).get('value') != 'playing'):
        check('evidence', 'field_evidence_unbound', 'blocked', '阳光、卡片或格子字段缺失、非对局或属于其他帧。')
        return _finish(report)
    check('evidence', 'fields_bound', 'passed', '字段结果绑定当前帧和采集时间。')
    cell = _unique(fields.get('cells'), row=intent['row'], column=intent['column'])
    occupancy = cell.get('occupancy') if cell and type(cell.get('occupancy')) is str else None
    known = occupancy in ('empty', 'ground_occupied', 'base_only', 'base_and_upper')
    check('target', 'cell_appearance_observed' if known else 'cell_appearance_unknown',
          'passed' if known else 'unknown', '格子外观仅作局部证据，不能推出植物品种或种植合法性。', occupancy)
    if intent['kind'] == 'shovel':
        layer = intent['layer']
        present = (occupancy in ('base_only', 'base_and_upper') if layer == 'carrier'
                   else occupancy in ('ground_occupied', 'base_and_upper')) if known else None
        check('target', 'shovel_layer_observed' if present is True else
              'shovel_layer_absent' if present is False else 'shovel_layer_unknown',
              'passed' if present is True else 'blocked' if present is False else 'unknown',
              '单独核对铲除目标的上层植物或水面底座。', layer)
        if layer == 'carrier' and occupancy == 'base_and_upper':
            check('legality', 'carrier_has_upper_plant', 'blocked', '底座上还有植物；需要先明确处理上层的独立意图。')
        check('legality', 'shovel_identity_unverified', 'unknown', '尚未核实目标植物身份和铲除层级的实际点击行为。')
        return _finish(report)

    card = _unique(fields.get('cards'), slot=intent['slot'])
    visual = card.get('visual_state') if card and type(card.get('visual_state')) is str else None
    check('resources', 'card_bright' if visual == 'bright' else 'card_dim' if visual == 'dim' else 'card_appearance_unknown',
          'passed' if visual == 'bright' else 'blocked' if visual == 'dim' else 'unknown',
          '亮/暗仅是外观；暗卡的具体原因不能据此确定。', visual)
    sun = _mapping(fields.get('sun')).get('value')
    sun_known = _integer(sun, 0, 999999)
    check('resources', 'sun_counter_observed' if sun_known else 'sun_counter_unknown',
          'passed' if sun_known else 'unknown', '阳光存量读数；不是地面可拾取阳光。', sun if sun_known else None)
    check('resources', 'loadout_cost_unverified', 'unknown', '缺少与本局绑定的卡片身份/成本；不按槽号猜测，也不采用意图自报成本。')
    check('resources', 'budget_unverified', 'unknown', '本局成本未核实，尚不能证明阳光足够。')
    check('resources', 'cooldown_unverified', 'unknown', '尚无独立的冷却可用性证据，亮卡不能单独放行动作。')
    check('legality', 'planting_legality_unverified', 'unknown', '还需结合植物品种、水陆、底座及上层占用验证种植规则。')
    return _finish(report)


class ActionPreflightMonitor:
    """In-process current-state reader, with a second read after evidence checks.

    A change during review revokes its diagnostic snapshot. There is no input
    backend, no queue and no transferable permission token. A future executor
    still needs atomic stop/latest-frame checks immediately before each input.
    """
    def __init__(self, *, root, current_state, stop_requested, clock=time.monotonic):
        self.root = Path(root)
        self.current_state = current_state
        self.stop_requested = stop_requested
        self.clock = clock

    def review(self, intent):
        try:
            state = copy.deepcopy(self.current_state(self.clock()))
            stopped = self.stop_requested()
        except Exception:
            state, stopped = None, None
        report = inspect_intent(intent, state, root=self.root, now=self.clock(), stopped=stopped)
        try:
            later = self.current_state(self.clock())
            same = (type(state) is dict and type(later) is dict and
                    _bound(state) == _bound(later) and fresh(later, self.clock()))
            stop = self.stop_requested()
        except Exception:
            same, stop = False, None
        if report['reviewed_state_id'] is not None and not same:
            report['checks'].append(dict(layer='freshness', code='state_changed_during_review', status='blocked',
                                         message='检查期间快照被替换、过期或连续性失效。', observed=None))
        if stopped is False and stop is not False:
            report['checks'].append(dict(layer='control', code='stop_changed_during_review', status='blocked',
                                         message='检查期间收到停止或无法确认停止状态。', observed=None))
        return _finish(report)
