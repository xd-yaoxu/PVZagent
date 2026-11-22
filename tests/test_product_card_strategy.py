"""Known-source card pixels; sun/cards/cell fields are synthetic, not visual truth."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from pvz_agent.action_preflight import frame_binding
from pvz_agent.card_identity import CardIdentityPerception
from pvz_agent.loadout_preflight import LoadoutPreflightMonitor
from pvz_agent.product_card_strategy import card_context, recommend_product_card_strategy
from pvz_agent.strategy_profile import ProfiledStrategyProvider
from pvz_agent.strategy_review import STRATEGY_SCHEMA


class ProductCardStrategyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT / 'config/card-identity-v1.json').read_text(encoding='utf-8'))
        self.catalog = json.loads((ROOT / self.config['catalog']).read_text(encoding='utf-8'))
        for rel in ['config/card-identity-v1.json', self.config['catalog'], self.catalog['reference']['image'],
                    'data/day06/card-identity-v1/alternate-day01.png']:
            target = self.root / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
        self.engine = CardIdentityPerception(self.root, self.config)
        self.bodies = []

    def state(self, alternate=False):
        rel = ('data/day06/card-identity-v1/alternate-day01.png' if alternate
               else self.catalog['reference']['image'])
        raw = (self.root / rel).read_bytes()
        state = dict(state_id='f1', run_id='run1/continuity-0', capture_time='historical:f1',
                     frame_time_seconds=10., scene=dict(value='playing'),
                     evidence=dict(image=rel, image_sha256=hashlib.sha256(raw).hexdigest()),
                     continuity=dict(epoch=0, capture_epoch=0, source_run_id='run1'),
                     readonly_runtime=dict(generation=1, expires_monotonic=12., publication_at_completion='published'),
                     fields=dict(state_id='f1', capture_time='historical:f1', scene=dict(value='playing'),
                                 sun=dict(value=200), cards=[dict(slot=i, visual_state='bright') for i in range(1,11)],
                                 cells=[dict(row=1,column=1,occupancy='empty')]),
                     zombies=dict(candidates=[]))
        state['loadout'] = self.engine.observe(state)
        return state

    def provider(self):
        def transport(url, body, key, timeout):
            self.bodies.append(copy.deepcopy(body))
            response = dict(source_state_id='f1', goal='hold', focus_row=None, reason='wait', uncertainties=[])
            return dict(status='completed', id='fake', model='fake', output=[dict(type='message',content=[
                dict(type='output_text',text=json.dumps(response))])])
        cfg = dict(base_url='https://example.invalid/v1', model='fixture', protocol='responses',
                   image_transfer_authorized=True, max_output_tokens=128, timeout_seconds=1,
                   api_key_environment_variable='unused')
        return ProfiledStrategyProvider(cfg, transport=transport, key_reader=lambda _: 'fixture')

    def test_real_pixels_request_and_existing_preflight(self):
        state = self.state(); result = recommend_product_card_strategy(self.provider(), self.root, state)
        body = self.bodies[-1]; prompt = body['input'][0]['content'][0]['text']
        self.assertEqual(body['text']['format']['schema'], STRATEGY_SCHEMA)
        self.assertIn('"plant_id":"sunflower"', prompt)
        self.assertIn('"printed_cost":50', prompt)
        self.assertIn(state['evidence']['image_sha256'], prompt)
        self.assertFalse(result['action_eligible']); self.assertEqual(result['strategy']['goal'], 'hold')
        monitor = LoadoutPreflightMonitor(root=self.root, current_state=lambda _: state,
            stop_requested=lambda: True, clock=lambda: 10.5, config=self.config, catalog=self.catalog)
        intent = dict(frame_binding(state), intent_id='diagnostic', kind='plant', slot=1, row=1, column=1)
        report = monitor.review(intent)
        self.assertIn('observed_budget_sufficient', [c['code'] for c in report['checks']])
        self.assertIn('stop_latched', report['blocking_codes'])
        for code in ('selection_state_unverified','cooldown_unverified','planting_legality_unverified'):
            self.assertIn(code, report['unknown_codes'])
        self.assertFalse(report['action_eligible'])

    def test_alternate_image_unknown_and_no_cross_frame_cache(self):
        first = card_context(self.root, self.state())
        second_state = self.state(True); second = card_context(self.root, second_state)
        self.assertNotEqual(first['image_sha256'], second['image_sha256'])
        self.assertEqual(second['slots'][3]['status'], 'unknown')
        self.assertIsNone(second['slots'][3]['plant_id'])
        recommend_product_card_strategy(self.provider(), self.root, second_state)
        self.assertIn('"status":"unknown","plant_id":null', self.bodies[-1]['input'][0]['content'][0]['text'])

    def test_invalid_bindings_fail_and_conflicts_remain_unknown(self):
        original = self.state()
        for key, value in [('state_id','foreign'),('capture_time','foreign'),('source_run_id','foreign'),
                           ('catalog_sha256','0'*64),('config_sha256','0'*64)]:
            state = copy.deepcopy(original); state['loadout'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): card_context(self.root,state)
        for key, value in [('image','foreign'),('image_sha256','0'*64)]:
            state = copy.deepcopy(original); state['loadout']['evidence'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): card_context(self.root,state)
        state=copy.deepcopy(original); state['loadout']['slots'][1]['slot']=1
        with self.assertRaises(ValueError): card_context(self.root,state)
        for slots in ([], [None]*10, copy.deepcopy(original['loadout']['slots'])[:-1]):
            state=copy.deepcopy(original); state['loadout']['slots']=slots
            with self.assertRaises(ValueError): card_context(self.root,state)
        state=copy.deepcopy(original); state['fields']['state_id']='foreign'
        with self.assertRaises(ValueError): card_context(self.root,state)
        for mutate in (lambda s: s['loadout']['slots'][0].update(display_cost=0),
                       lambda s: s['loadout']['slots'][0].update(display_cost=True),
                       lambda s: s['loadout']['slots'][0].update(status='price_conflict'),
                       lambda s: s['loadout']['slots'][0].update(identity=[])):
            state=copy.deepcopy(original); mutate(state)
            item=card_context(self.root,state)['slots'][0]
            self.assertEqual(item['status'],'unknown'); self.assertIsNone(item['printed_cost'])
        state=copy.deepcopy(original); del state['loadout']
        with self.assertRaises(ValueError): card_context(self.root,state)
        state=copy.deepcopy(original); state['fields']['sun']['value']=True
        self.assertIsNone(card_context(self.root,state)['sun_estimate'])
        catalog_path=self.root/self.config['catalog']; old=catalog_path.read_bytes()
        catalog_path.write_bytes(old+b' ')
        with self.assertRaises(ValueError): card_context(self.root,original)
        catalog_path.write_bytes(old)

    def test_common_source_type_pollution_rejected_before_provider(self):
        original=self.state(); calls=[]
        config=dict(base_url='https://example.invalid/v1',model='fixture',protocol='responses',
                    image_transfer_authorized=True,max_output_tokens=128,timeout_seconds=1,
                    api_key_environment_variable='unused')
        provider=ProfiledStrategyProvider(config, transport=lambda *args: calls.append('transport'),
                                         key_reader=lambda _: calls.append('key'))
        for key in ('state_id','capture_time'):
            for invalid in (True, 1, [], '', 'x'*257):
                state=copy.deepcopy(original)
                state[key]=invalid; state['loadout'][key]=invalid; state['fields'][key]=invalid
                with self.subTest(key=key,invalid=invalid), self.assertRaises(ValueError):
                    recommend_product_card_strategy(provider,self.root,state)
        for invalid in (True, 1, [], '', 'x'*257):
            for key in ('run_id','source_run_id'):
                state=copy.deepcopy(original)
                if key=='run_id': state[key]=invalid
                else:
                    state['continuity'][key]=invalid; state['loadout'][key]=invalid
                with self.subTest(key=key,invalid=invalid), self.assertRaises(ValueError):
                    recommend_product_card_strategy(provider,self.root,state)
        for key, values in (('image',(True,1,[],'','x'*257)),
                            ('image_sha256',(True,1,[],'','A'*64,'0'*63))):
            for invalid in values:
                state=copy.deepcopy(original)
                state['evidence'][key]=invalid; state['loadout']['evidence'][key]=invalid
                with self.subTest(key=key,invalid=invalid), self.assertRaises(ValueError):
                    recommend_product_card_strategy(provider,self.root,state)
        for key in ('scene','continuity','fields','evidence','loadout'):
            for invalid in (None, [], True):
                state=copy.deepcopy(original); state[key]=invalid
                with self.subTest(container=key,invalid=invalid), self.assertRaises(ValueError):
                    recommend_product_card_strategy(provider,self.root,state)
        self.assertEqual(calls,[])

    def test_actual_cli_flag_uses_card_stream_and_real_provider_body(self):
        module=importlib.import_module('day06_product_resource_live'); state=self.state(); provider=self.provider()
        auth=dict(user_reply='\u5141\u8bb8',base_url='https://example.invalid/v1')
        cfg=dict(image_authorization_record='auth.json',image_transfer_authorized=True,
                 base_url=auth['base_url'],model='fixture')
        def load(path):
            return auth if path.name == 'auth.json' else {'policy': {}} if path.name == 'resource-attention-v1.json' else {}
        def runner(**kwargs):
            result=kwargs['backend'](state)
            self.assertIn('card_evidence',result)
            return dict(records=[],errors=[],strategy=dict(requests=1,workers_finished=True))
        with patch.multiple(module,ROOT=self.root,verify_product_startup=lambda *_: None,
                resolve_product_strategy=lambda: cfg,load=load,create_product_strategy=lambda _: provider,
                run_goal_readonly=runner), patch('pvz_agent.loadout_runtime.create_card_stream',return_value=object()) as stream, \
             patch.object(sys,'argv',['test','--out','new-output','--with-card-evidence']):
            module.main()
        stream.assert_called_once_with(self.root)
        self.assertIn('"plant_id":"sunflower"',self.bodies[0]['input'][0]['content'][0]['text'])
        self.assertEqual(self.bodies[0]['text']['format']['schema'],STRATEGY_SCHEMA)

    def test_default_cli_and_preflight_keep_original_paths(self):
        module=importlib.import_module('day06_product_resource_live'); state=self.state(); provider=self.provider()
        auth=dict(user_reply='\u5141\u8bb8',base_url='https://example.invalid/v1')
        cfg=dict(image_authorization_record='auth.json',image_transfer_authorized=True,
                 base_url=auth['base_url'],model='fixture')
        def load(path):
            return auth if path.name == 'auth.json' else {'policy': {}} if path.name == 'resource-attention-v1.json' else {}
        def runner(**kwargs):
            result=kwargs['backend'](state)
            self.assertNotIn('card_evidence',result)
            return dict(records=[],errors=[],strategy=dict(requests=1,workers_finished=True))
        with patch.multiple(module,ROOT=self.root,verify_product_startup=lambda *_: None,
                resolve_product_strategy=lambda: cfg,load=load,create_product_strategy=lambda _: provider,
                create_bucket_stream=lambda _: object(),run_goal_readonly=runner), \
             patch('pvz_agent.loadout_runtime.create_card_stream',side_effect=AssertionError('card stream used')), \
             patch.object(sys,'argv',['test','--out','default-output']):
            module.main()
        self.assertNotIn('"printed_cost"',self.bodies[-1]['input'][0]['content'][0]['text'])
        self.assertEqual(self.bodies[-1]['text']['format']['schema'],STRATEGY_SCHEMA)
        def forbidden(*args,**kwargs): raise AssertionError('unexpected access')
        with patch.multiple(module,ROOT=self.root,verify_product_startup=lambda *_: {'scope':'fixture'},
                resolve_product_strategy=forbidden,load=forbidden,create_product_strategy=forbidden,
                create_bucket_stream=forbidden,run_goal_readonly=forbidden,PinnedWindowsCapture=forbidden), \
             patch('pvz_agent.loadout_runtime.create_card_stream',side_effect=forbidden), \
             patch.object(sys,'argv',['test','--preflight-only','--with-card-evidence']):
            module.main()


if __name__ == '__main__': unittest.main()
