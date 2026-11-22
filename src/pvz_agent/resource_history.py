"""Persist checked strategy history separately from unverified provider output."""

import json
import re


HISTORY_SCHEMA = 'pvz-resource-review-history/1'
SAFE_ID = re.compile(r'^[A-Za-z0-9_.:+-]{1,160}$')
GOAL_POLICIES = {
    'collect_resources': 'resource_attention/v1',
    'defend_lane': 'defense_attention/v1',
    'hold': 'selected_observations/v1',
    'develop_economy': 'selected_observations/v1',
}


def _positive_count(value):
    return type(value) is int and value > 0


def _checked_basis(review):
    strategy = review.get('strategy')
    if (review.get('disposition') != 'selected_observations_match'
            or review.get('validity') != 'historical_at_commit'
            or review.get('input_sent') is not False
            or review.get('action_eligible') is not False
            or type(strategy) is not dict
            or type(strategy.get('goal')) is not str
            or strategy['goal'] not in GOAL_POLICIES
            or review.get('review_policy') != GOAL_POLICIES[strategy['goal']]
            or type(review.get('source_state_id')) is not str
            or not review['source_state_id']
            or strategy.get('source_state_id') != review['source_state_id']
            or type(review.get('reviewed_state_id')) is not str
            or not review['reviewed_state_id']):
        return None

    goal, row = strategy['goal'], strategy.get('focus_row')
    evidence = review.get('evidence')
    if goal == 'defend_lane':
        if (type(row) is not int or not 1 <= row <= 6 or type(evidence) is not dict
                or type(evidence.get('focus_row')) is not int or evidence['focus_row'] != row
                or not _positive_count(evidence.get('current_visible_count'))
                or type(evidence.get('source_visible_count')) is not int
                or evidence['source_visible_count'] != evidence['current_visible_count']
                or type(evidence.get('current_left_warning')) is not bool
                or any(evidence.get(key) is not False for key in
                       ('identity_verified', 'death_verified', 'full_board_verified'))):
            return None
        return dict(kind='visible_heads_in_focus_lane', row=row,
                    count=evidence['current_visible_count'],
                    left_warning=evidence['current_left_warning'])
    if row is not None or 'focus_row' not in strategy:
        return None
    if goal == 'collect_resources':
        if (type(evidence) is not dict or not _positive_count(evidence.get('current_cue_count'))
                or any(evidence.get(key) is not False for key in
                       ('original_sun_identity_verified', 'interactable_verified', 'full_board_verified'))):
            return None
        source_ids, current_ids = evidence.get('source_cue_ids'), evidence.get('current_cue_ids')
        if any(type(ids) is not list or not ids or any(type(v) is not str or not v for v in ids)
               or len(set(ids)) != len(ids) for ids in (source_ids, current_ids)):
            return None
        if len(current_ids) != evidence['current_cue_count']:
            return None
        return dict(kind='visible_sun_appearance', count=len(current_ids))
    if evidence is None:
        return dict(kind='selected_observations_match')
    return None


def project_history_record(record):
    """Called by the product writer with an in-process review, never with old files."""
    if record.get('kind') != 'strategy_review':
        return record
    result = {key: record[key] for key in
              ('kind', 'reviewed_state_id', 'monotonic', 'input_sent', 'action_eligible')
              if key in record}
    review = record.get('review')
    if type(review) is not dict:
        return result
    history = {key: review[key] for key in
               ('disposition', 'review_policy', 'source_state_id', 'reviewed_state_id',
                'expires_monotonic', 'source_age_seconds', 'validity', 'input_sent',
                'action_eligible', 'requires_new_action_proposal',
                'requires_current_target_and_legality_check', 'requires_fresh_pickup_proposal')
               if key in review}
    basis = _checked_basis(review)
    if basis is not None:
        strategy = review['strategy']
        history['strategy'] = {key: strategy[key] for key in
                               ('source_state_id', 'goal', 'focus_row')}
        history['verified_basis'] = basis
        history['history_schema'] = HISTORY_SCHEMA
    result['review'] = history
    return result


def append_history_record(stream, record):
    stream.write(json.dumps(project_history_record(record), ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def write_history_summary(path, result, *, write):
    strategy = result['strategy']
    write(path, {**result, 'strategy': {**strategy, 'records': [
        project_history_record(record) for record in strategy['records']
    ]}})


def write_provider_result(folder, name, record, *, write):
    """Return the diagnostic file path for existing explicit replay readers."""
    diagnostic = folder / 'diagnostics' / 'model-unverified' / name
    write(diagnostic, {**record, 'diagnostic_scope': 'unverified_model_output'})
    metadata = {key: record[key] for key in
                ('success', 'source_state_id', 'started_at', 'completed_at',
                 'duration_ms', 'input_sent', 'action_eligible') if key in record}
    model = record.get('model_returned')
    if type(model) is str and SAFE_ID.fullmatch(model):
        metadata['model_returned'] = model
    strategy = record.get('strategy')
    source = strategy.get('source_state_id') if type(strategy) is dict else None
    if type(source) is str and SAFE_ID.fullmatch(source):
        metadata['strategy'] = {'source_state_id': source}
    usage = record.get('usage')
    if type(usage) is dict:
        metadata['usage'] = {key: usage[key] for key in
                             ('input_tokens', 'output_tokens', 'total_tokens')
                             if type(usage.get(key)) is int and usage[key] >= 0}
    write(folder / name, metadata)
    return diagnostic


def history_basis_text(review):
    """Render only the versioned, typed projection. Legacy prose is not evidence."""
    if (review.get('history_schema') != HISTORY_SCHEMA
            or review.get('disposition') != 'selected_observations_match'
            or review.get('validity') != 'historical_at_commit'
            or review.get('input_sent') is not False or review.get('action_eligible') is not False):
        return '已核实依据未知'
    basis = review.get('verified_basis')
    strategy = review.get('strategy')
    if type(basis) is not dict or type(strategy) is not dict:
        return '已核实依据未知'
    goal = strategy.get('goal')
    if (type(goal) is not str or goal not in GOAL_POLICIES
            or review.get('review_policy') != GOAL_POLICIES[goal]
            or type(review.get('source_state_id')) is not str or not review['source_state_id']
            or strategy.get('source_state_id') != review['source_state_id']
            or type(review.get('reviewed_state_id')) is not str or not review['reviewed_state_id']):
        return '已核实依据未知'
    kind, count, row = basis.get('kind'), basis.get('count'), strategy.get('focus_row')
    if (goal == 'collect_resources' and row is None and kind == 'visible_sun_appearance'
            and _positive_count(count)):
        return f'复核时有 {count} 个阳光外观候选'
    if (goal == 'defend_lane' and kind == 'visible_heads_in_focus_lane'
            and type(row) is int and 1 <= row <= 6 and type(basis.get('row')) is int
            and basis['row'] == row and _positive_count(count)
            and type(basis.get('left_warning')) is bool):
        return f'复核时第 {row} 路有 {count} 个可见头部候选'
    if goal in ('hold', 'develop_economy') and row is None and kind == 'selected_observations_match':
        return '复核时场景、阳光读数、卡面状态和头部候选分区匹配'
    return '已核实依据未知'
