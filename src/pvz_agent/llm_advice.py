"""Historical screenshot recommendations with strict local validation; no actions."""
import base64
import hashlib
import json
from pathlib import Path
from PIL import Image
from .llm_provider import ProviderError


ADVICE_SCHEMA=dict(type='object',additionalProperties=False,required=['source_state_id','scene','public_summary','uncertainties','recommendations'],
    properties=dict(source_state_id=dict(type='string'),scene=dict(type='string',enum=['playing','nonplaying','unknown']),
        public_summary=dict(type='string'),uncertainties=dict(type='array',items=dict(type='string')),
        recommendations=dict(type='array',items=dict(type='object',additionalProperties=False,
            required=['kind','slot','row','column','x','y','reason'],properties=dict(
                kind=dict(type='string',enum=['wait','collect_sun','plant','shovel']),
                slot=dict(type=['integer','null']),row=dict(type=['integer','null']),column=dict(type=['integer','null']),
                x=dict(type=['integer','null']),y=dict(type=['integer','null']),reason=dict(type='string'))))))


def strict_json(text):
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(text,object_pairs_hook=object_pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def validate_advice(advice,source_state_id):
    if not isinstance(advice,dict) or set(advice)!=set(ADVICE_SCHEMA['required']):raise ValueError('Advice fields mismatch')
    if advice['source_state_id']!=source_state_id:raise ValueError('Source state mismatch')
    if advice['scene'] not in ['playing','nonplaying','unknown']:raise ValueError('Unknown scene enum')
    if not isinstance(advice['public_summary'],str) or not 1<=len(advice['public_summary'])<=240:raise ValueError('Public summary length')
    if not isinstance(advice['uncertainties'],list) or len(advice['uncertainties'])>12 or any(not isinstance(s,str) or len(s)>160 for s in advice['uncertainties']):
        raise ValueError('Uncertainty list invalid')
    if not isinstance(advice['recommendations'],list) or len(advice['recommendations'])>3:raise ValueError('At most3 recommendations')
    for action in advice['recommendations']:
        if not isinstance(action,dict) or set(action)!=set(['kind','slot','row','column','x','y','reason']):raise ValueError('Recommendation fields mismatch')
        if action['kind'] not in ['wait','collect_sun','plant','shovel']:raise ValueError('Unknown proposal')
        if not isinstance(action['reason'],str) or not 1<=len(action['reason'])<=160:raise ValueError('Brief public reason required')
        for key,maximum in [('slot',10),('row',6),('column',9),('x',799),('y',599)]:
            value=action[key]
            if value is not None and (type(value) is not int or not (0 if key in ['x','y'] else 1)<=value<=maximum):
                raise ValueError('Canonical target range')
        values={k:action[k] for k in ['slot','row','column','x','y']}
        required={'wait':set(),'collect_sun':{'x','y'},'plant':{'slot','row','column'},'shovel':{'row','column'}}[action['kind']]
        if {k for k,v in values.items() if v is not None}!=required:raise ValueError('Proposal target semantics mismatch')
        if advice['scene']!='playing' and action['kind']!='wait':raise ValueError('Nonplaying proposal must wait')
    return advice


def screenshot_request(root,state):
    root=Path(root).resolve();path=(root/state['evidence']['image']).resolve()
    if not path.is_relative_to(root):raise ValueError('Screenshot outside project')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=state['evidence']['image_sha256']:raise ValueError('Screenshot hash mismatch')
    with Image.open(path) as image:
        if image.size!=(800,600) or image.format!='PNG':raise ValueError('Canonical PNG screenshot required')
    brief=dict(source_state_id=state['state_id'],scene_estimate=state['scene']['value'],
        sun_estimate=state['fields']['sun']['value'],
        card_appearance_estimates=[dict(slot=c['slot'],visual_state=c['visual_state']) for c in state['fields']['cards']],
        headed_candidate_estimates=[dict(row=p['row'],visible_reference_xy=p['head_xy']) for p in state['zombies']['candidates']],
        limits='Auxiliary screenshot estimates can miss edge targets; missing detections never prove an empty lane. Alive/health/legality unverified.')
    prompt=('你是植物大战僵尸3-7截图战术顾问。这是历史静态截图，只针对当时画面给出最多3条建议。'
        '主要根据原图，附带估计可能漏检。不得声称已经操作、通关或确认死亡。'
        '输出简短中文公开理由，不需要思维过程。未知资源、卡片或目标前提请列入uncertainties；不能确认时建议wait。'
        'kind为wait/collect_sun/plant/shovel，slot为1到10，row为1到6，column为1到9，x/y为800x600原图坐标。'
        'wait所有目标null；collect_sun只填x/y；plant只填slot/row/column；shovel只填row/column。'
        '任何建议都需要未来新截图重新验证，当前不执行。请原样回填source_state_id。\n'+json.dumps(brief,ensure_ascii=False))
    return prompt,'data:image/png;base64,'+base64.b64encode(raw).decode('ascii')


def recommend_static(provider,root,state):
    prompt,image=screenshot_request(root,state)
    result=provider.complete(prompt,image_data_url=image,json_schema=ADVICE_SCHEMA)
    try:advice=validate_advice(strict_json(result['text']),state['state_id'])
    except (ValueError,TypeError,KeyError):raise ProviderError('invalid_structured_advice') from None
    return dict(advice=advice,protocol=result['protocol'],response_id=result['response_id'],model_returned=result['model_returned'],usage=result['usage'],
        mode='historical_static_screenshot',input_sent=False,action_eligible=False,current_game_state=False)
