"""Bounded, offline projection of product run records. No runtime imports."""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
from collections import Counter, defaultdict

from .resource_history import history_basis_text

MAX_FILE = 4 * 1024 * 1024
MAX_LINES = 20000
MAX_LINE = 128 * 1024
FILES = ('journal.jsonl', 'summary.json', 'provider-result-00.json',
         'provider-result-01.json', 'provider-result-02.json')
KINDS = {'captured', 'inferred', 'replaced_pending', 'rejected_capture',
         'expired_before_inference', 'strategy_requested', 'strategy_review',
         'strategy_source_unknown'}
PUBLICATIONS = {'published', 'superseded', 'continuity_rejected', 'expired',
                'expired_after_inference', 'rejected', 'error'}
ADVISER = {'hypothesis_retained', 'stale_before_request', 'invalidated_response',
           'request_failed', 'historical_window_expired', 'invalidated',
           'stale_response', 'timeout', 'local_dispatch_committed', 'launch_failed'}
REASONS = {'capture_source_closed', 'continuity_lost'}
DISPOSITIONS = {
    'focus_lane_unconfirmed': '关注行证据未确认，复核阻断。',
    'observed_context_changed': '所比较的观测上下文发生变化，复核阻断。',
    'selected_observations_match': '选定观测匹配；不代表执行、通关或当前可用。',
}
EVENT_FIELDS = {'kind', 'state_id', 'source_state_id', 'reviewed_state_id', 'valid',
                'monotonic_ns', 'recorded_at', 'publication', 'monotonic', 'review'}
SAFE_ID = re.compile(r'^[A-Za-z0-9_.:+-]{1,160}$')
SAFE_TIME = re.compile(r'^[0-9TtZz:+.\-]{1,64}$')


class ReportError(ValueError):
    pass


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ReportError('JSON contains duplicate key')
        out[key] = value
    return out


def _constant(_value):
    raise ReportError('JSON contains non-finite number')


def _number(value):
    # Reject overflow and bound even unused numeric fields.
    if len(value) > 64:
        raise ReportError('JSON number too large')
    result = float(value)
    if not math.isfinite(result) or abs(result) > 1e100:
        raise ReportError('JSON number out of range')
    return result


def _integer(value):
    if len(value.lstrip('-')) > 100:
        raise ReportError('JSON integer too large')
    return int(value)


def _json(data):
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=_constant, parse_float=_number,
                           parse_int=_integer)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ReportError('invalid JSON') from exc
    if not isinstance(value, dict):
        raise ReportError('JSON root must be an object')
    return value


def _reparse(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def _read(root, name, required=False):
    path = root / name
    try:
        if _reparse(path) or path.resolve(strict=True).parent != root:
            raise ReportError('input is a link or escapes the run directory: ' + name)
        if not path.is_file():
            raise ReportError('input is not a regular file: ' + name)
        with path.open('rb') as stream:
            data = stream.read(MAX_FILE + 1)
        if len(data) > MAX_FILE:
            raise ReportError('input exceeds size limit: ' + name)
        # A changed path is rejected, rather than following an embedded path.
        if _reparse(path) or path.resolve(strict=True).parent != root:
            raise ReportError('input changed during read: ' + name)
        return data
    except FileNotFoundError:
        if required:
            raise ReportError('required journal missing')
        return None
    except OSError as exc:
        raise ReportError('cannot read fixed input: ' + name) from exc


def _identifier(value):
    return value if isinstance(value, str) and SAFE_ID.fullmatch(value) else 'unknown'


def _timestamp(value):
    return value if isinstance(value, str) and SAFE_TIME.fullmatch(value) else 'unknown'


def _num(value):
    return str(value) if type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e15 else 'unknown'


def _bool(value):
    return 'true' if value is True else 'false' if value is False else 'unknown'


def _table(headers, rows):
    esc = lambda value: html.escape(str(value), quote=True)
    return ('<table><thead><tr>' + ''.join('<th>'+esc(x)+'</th>' for x in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>'+esc(x)+'</td>'
                for x in row) + '</tr>' for row in rows) + '</tbody></table>')


def build_report(run_directory):
    try:
        supplied = Path(run_directory)
        root = supplied.resolve(strict=True)
        if not root.is_dir():
            raise ReportError('run directory is not a directory')
    except OSError as exc:
        raise ReportError('run directory unavailable') from exc
    raw = {name: _read(root, name, name == 'journal.jsonl') for name in FILES}
    journal = raw['journal.jsonl']
    if not journal or not journal.endswith(b'\n'):
        raise ReportError('journal empty or incomplete final line')
    lines = journal.splitlines()
    if len(lines) > MAX_LINES or any(len(line) > MAX_LINE or not line.strip() for line in lines):
        raise ReportError('journal line count, size, or structure invalid')
    events = [_json(line) for line in lines]
    summary = _json(raw['summary.json']) if raw['summary.json'] is not None else None
    providers = [(name, _json(raw[name])) for name in FILES[2:] if raw[name] is not None]
    files = [(name, hashlib.sha256(data).hexdigest() if data is not None else '缺失 / unknown')
             for name, data in raw.items()]

    counts, pub = Counter(), Counter()
    extra_fields = 0
    event_rows = []
    flag_evidence = defaultdict(lambda: defaultdict(list))

    def flags(obj, ref):
        if isinstance(obj, dict):
            for field in ('input_sent', 'action_eligible'):
                value = obj.get(field)
                if type(value) is bool:
                    flag_evidence[field][value].append(ref + '.' + field)

    links = defaultdict(lambda: {'capture': [], 'inference': [], 'request': [], 'provider': [], 'review': []})
    for line_no, event in enumerate(events, 1):
        flags(event, 'journal:%d' % line_no)
        kind = event.get('kind')
        known = kind if isinstance(kind, str) and kind in KINDS else 'unknown kind'
        counts[known] += 1
        extra_fields += len(set(event) - EVENT_FIELDS)
        sid = _identifier(event.get('state_id'))
        source = _identifier(event.get('source_state_id'))
        reviewed = _identifier(event.get('reviewed_state_id'))
        detail = ''
        if known == 'captured':
            detail = 'valid=' + _bool(event.get('valid'))
            counts['captured valid='+_bool(event.get('valid'))] += 1
            if sid != 'unknown':
                links[sid]['capture'].append('journal:%d' % line_no)
        elif known == 'inferred':
            publication = event.get('publication')
            publication = publication if isinstance(publication, str) and publication in PUBLICATIONS else 'unknown'
            pub[publication] += 1
            detail = 'publication=' + publication
            if sid != 'unknown':
                links[sid]['inference'].append('journal:%d' % line_no)
        elif known == 'strategy_requested' and source != 'unknown':
            links[source]['request'].append('journal:%d' % line_no)
        elif known == 'strategy_source_unknown':
            detail = '原因未记录 / unknown'
        elif known == 'strategy_review':
            review = event.get('review')
            if not isinstance(review, dict):
                detail = 'review 缺失或结构未知'
            else:
                flags(review, 'journal:%d.review' % line_no)
                outer_source = source
                inner_source = _identifier(review.get('source_state_id'))
                source_conflict = outer_source != 'unknown' and inner_source != 'unknown' and outer_source != inner_source
                source = inner_source if inner_source != 'unknown' else outer_source
                inner = _identifier(review.get('reviewed_state_id'))
                conflict = reviewed != 'unknown' and inner != 'unknown' and reviewed != inner
                if source_conflict:
                    detail = 'source_state_id 冲突：outer=' + outer_source + ' inner=' + inner_source + '；'
                if conflict:
                    detail += 'reviewed_state_id 冲突：outer=' + reviewed + ' inner=' + inner
                elif inner == 'unknown' and reviewed != 'unknown':
                    detail += '嵌套 reviewed_state_id 缺失；使用外层'
                else:
                    reviewed = inner if inner != 'unknown' else reviewed
                disposition = review.get('disposition')
                disposition = disposition if isinstance(disposition, str) and disposition in DISPOSITIONS else 'unknown'
                detail += ' disposition=' + disposition + ' ' + DISPOSITIONS.get(disposition, '原因未知 / 未覆盖。')
                if disposition == 'selected_observations_match' and not conflict and not source_conflict:
                    basis = history_basis_text(review) if inner_source != 'unknown' and inner != 'unknown' else '已核实依据未知'
                    detail += ' ' + basis
                if source != 'unknown' and not conflict and not source_conflict:
                    links[source]['review'].append('journal:%d' % line_no)
        event_rows.append(('journal.jsonl:%d' % line_no, known, sid, source, reviewed,
                           _timestamp(event.get('recorded_at')), _num(event.get('monotonic_ns')),
                           _num(event.get('monotonic')), detail))

    provider_rows = []
    for name, obj in providers:
        flags(obj, name+':$')
        sid = _identifier(obj.get('source_state_id'))
        nested = obj.get('strategy')
        nested_sid = _identifier(nested.get('source_state_id')) if isinstance(nested, dict) else 'unknown'
        conflict = sid != 'unknown' and nested_sid != 'unknown' and sid != nested_sid
        if sid != 'unknown' and not conflict:
            links[sid]['provider'].append(name+':$.source_state_id')
        success = '客户端记录成功' if obj.get('success') is True else ('客户端记录失败' if obj.get('success') is False else 'unknown')
        usage = obj.get('usage') if isinstance(obj.get('usage'), dict) else {}
        provider_rows.append((name+':$.success', sid, success, _num(obj.get('duration_ms')),
                              _timestamp(obj.get('started_at')), _timestamp(obj.get('completed_at')),
                              _identifier(obj.get('model_returned')),
                              ('source_state_id 冲突：outer=' + sid + ' inner=' + nested_sid) if conflict else '',
                              '/'.join(_num(usage.get(x)) for x in ('input_tokens', 'output_tokens', 'total_tokens'))))

    strategy = summary.get('strategy') if isinstance(summary, dict) and isinstance(summary.get('strategy'), dict) else {}
    flags(summary, 'summary.json:$')
    flags(strategy, 'summary.json:$.strategy')
    summary_rows = [('summary.json:$.strategy.requests', _num(strategy.get('requests'))),
                    ('summary.json:$.strategy.workers_finished', _bool(strategy.get('workers_finished'))),
                    ('summary.json:$.input_sent（日志声明）', _bool(summary.get('input_sent')) if summary else 'unknown')]
    adviser_rows = []
    adviser = strategy.get('adviser_events')
    if isinstance(adviser, list):
        for index, item in enumerate(adviser):
            if not isinstance(item, dict):
                adviser_rows.append((f'summary.json:$.strategy.adviser_events[{index}]', 'unknown', 'unknown', 'unknown'))
                continue
            k = item.get('kind'); k = k if isinstance(k, str) and k in ADVISER else 'unknown'
            reason = item.get('reason'); reason = reason if isinstance(reason, str) and reason in REASONS else 'unknown'
            adviser_rows.append((f'summary.json:$.strategy.adviser_events[{index}]', k,
                                 _identifier(item.get('source_state_id')), reason))
    link_rows = []
    for sid, parts in sorted(links.items()):
        ambiguous = any(len(parts[k]) > 1 for k in ('capture', 'inference', 'request', 'provider'))
        note = '多条记录，关联多义；未合并' if ambiguous else '仅按明确来源 ID 列示；缺项为未知'
        if len(parts['review']) > 1:
            note += '；多条复核可能来自不同当前帧或重复记录，独立性未验证，不自动合并或去重，不证明成功'
        link_rows.append((sid, ', '.join(parts['capture']) or 'unknown', ', '.join(parts['inference']) or 'unknown', ', '.join(parts['request']) or 'unknown',
                          ', '.join(parts['provider']) or 'unknown', ', '.join(parts['review']) or 'unknown',
                          note))

    count_rows = sorted(counts.items()) + [('publication '+k, v) for k, v in sorted(pub.items())]
    count_rows += [('journal 未覆盖字段数（顶层字段出现次数）', extra_fields)]
    count_rows += [('summary 未投影顶层字段数', len(set(summary) - {'strategy', 'input_sent'}) if summary else 'unknown'),
                   ('provider 未投影顶层字段数', sum(len(set(obj) - {'success', 'source_state_id', 'duration_ms', 'started_at', 'completed_at', 'model_returned', 'usage'}) for _, obj in providers)),
                   ('建议器事件类别未知条数', sum(1 for row in adviser_rows if row[1] == 'unknown')),
                   ('建议器事件原因未知条数', sum(1 for row in adviser_rows if row[3] == 'unknown'))]
    declared = strategy.get('requests')
    actual = counts['strategy_requested']
    checks = []
    if type(declared) is int and 0 <= declared <= 1e15:
        checks.append(('请求事件与摘要声明', '一致' if declared == actual else '冲突：journal=%s summary=%s' % (actual, declared)))
    else:
        checks.append(('请求事件与摘要声明', 'unknown：缺少有效摘要声明'))
    for field in ('input_sent', 'action_eligible'):
        values = flag_evidence[field]
        if values[True] and values[False]:
            checks.append((field, '冲突：true=' + ', '.join(values[True]) + '；false=' + ', '.join(values[False])))
        elif values[True]:
            checks.append((field, '存在 true 声明：' + ', '.join(values[True])))
        elif values[False]:
            checks.append((field, '仅发现 false 声明（非历史完整性证明）'))
        else:
            checks.append((field, 'unknown'))
    sections = [('<h2>固定输入与实际 SHA256</h2>', _table(('相对文件', 'SHA256 / 状态'), files)),
                ('<h2>记录计数</h2>', _table(('类别', '条数'), count_rows)),
                ('<h2>摘要声明</h2>', _table(('JSON 路径', '值'), summary_rows)),
                ('<h2>证据一致性</h2>', _table(('检查', '结果'), checks)),
                ('<h2>客户端结果</h2>', _table(('JSON 路径', 'source_state_id', '结果', 'duration_ms', 'started_at', 'completed_at', 'model_returned', '状态冲突', 'usage input/output/total'), provider_rows)),
                ('<h2>明确来源 ID 索引</h2>', _table(('source_state_id', '采集记录', '感知记录', '请求记录', '客户端文件', '复核记录', '关联限制'), link_rows)),
                ('<h2>摘要中的建议器事件</h2>', _table(('JSON 路径', '类别', 'source_state_id', '原因'), adviser_rows)),
                ('<h2>逐条日志</h2>', _table(('行号', '类别', 'state_id', 'source_state_id', 'reviewed_state_id', 'recorded_at', 'monotonic_ns', 'monotonic', '说明'), event_rows))]
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>只读运行诊断</title>'
            '<style>body{font:15px sans-serif;margin:2em;color:#20252b}table{border-collapse:collapse;width:100%;margin-bottom:2em}th,td{border:1px solid #ccc;padding:.4em;overflow-wrap:anywhere;text-align:left}th{background:#edf2f5}</style>'
            '<h1>只读运行诊断</h1><p>本报告为离线只读投影，不提供游戏输入或当前行动资格。缺少摘要或客户端文件不证明未调用。请求事件数、摘要声明和实际 HTTP / 计费是不同事实；实际 HTTP 与费用未知。本地 workers_finished 不证明远端取消。local_dispatch_committed 只记录本地派发点，不证明 HTTP 接收。hypothesis_retained 不代表当前可用。时钟原样列示，未跨时钟相减或使用文件修改时间。未知字段和原因不解释为成功。</p>'
            + ''.join(a+b for a,b in sections) + '</html>')


def write_report(run_directory, output):
    text = build_report(run_directory)
    target = Path(output)
    root = Path(run_directory).resolve(strict=True)
    try:
        dest = target.resolve(strict=False)
        if dest == root or root in dest.parents:
            raise ReportError('output must be outside the input directory')
        try:
            with target.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
        except OSError as exc:
            # A failed write/flush/close may leave a partial file. Never unlink a
            # path which another process could have replaced in the meantime.
            raise ReportError('output creation/write failed; a partial output may remain') from exc
    except OSError as exc:
        raise ReportError('cannot exclusively create output') from exc
