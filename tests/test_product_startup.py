"""Offline product startup version-check tests; no game, provider or credentials."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pvz_agent.product_startup import MANIFEST,OLD_FREEZE,OLD_FREEZE_SHA256,REQUIRED,verify_product_startup

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for rel in (*REQUIRED,OLD_FREEZE,MANIFEST):
            path=self.root/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,path)
        (self.root/'evidence/day04').mkdir(parents=True,exist_ok=True);(self.root/'evidence/day04/STOP').touch()
        self.manifest=self.root/MANIFEST
    def read_manifest(self):return json.loads(self.manifest.read_text(encoding='utf-8'))
    def write_manifest(self,value):self.manifest.write_text(json.dumps(value),encoding='utf-8')
    def test_actual_cli_preflight_only_and_no_output_required(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_product_agent.py'),'--preflight-only'],
            cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['scope'],'file version check only')
    def test_each_protected_artifact_mutation_and_deletion_fail(self):
        self.assertEqual(verify_product_startup(self.root)['artifacts_checked'],len(REQUIRED))
        for rel in sorted(REQUIRED):
            path=self.root/rel;original=path.read_bytes()
            with self.subTest(path=rel,case='modified'):
                path.write_bytes(original+b'changed')
                with self.assertRaises(ValueError):verify_product_startup(self.root)
            with self.subTest(path=rel,case='missing'):
                path.unlink()
                with self.assertRaises(FileNotFoundError):verify_product_startup(self.root)
            path.write_bytes(original)
    def test_manifest_missing_empty_required_digest_and_traversal_rejected(self):
        original=self.read_manifest()
        cases=[]
        for artifacts in ({},None,dict(list(original['artifacts'].items())[1:]),
                          dict(original['artifacts'],**{'../outside':'a'*64})):
            m=dict(original);m['artifacts']=artifacts;cases.append(m)
        for digest in ('','A'*64,'a'*63,True,None,'z'*64):
            m=json.loads(json.dumps(original));m['artifacts'][next(iter(REQUIRED))]=digest;cases.append(m)
        for case in cases:
            with self.subTest(case=str(case)[:100]):
                self.write_manifest(case)
                with self.assertRaises(ValueError):verify_product_startup(self.root)
        self.manifest.unlink()
        with self.assertRaises(FileNotFoundError):verify_product_startup(self.root)
    def test_historical_freeze_binding_and_stop_fail_closed(self):
        old=self.root/OLD_FREEZE;original=old.read_bytes();old.write_bytes(original+b'changed')
        with self.assertRaises(ValueError):verify_product_startup(self.root)
        old.write_bytes(original);manifest=self.read_manifest();manifest['historical_freeze']['sha256']='0'*64
        self.write_manifest(manifest)
        with self.assertRaises(ValueError):verify_product_startup(self.root)
        manifest['historical_freeze']['sha256']=OLD_FREEZE_SHA256;self.write_manifest(manifest)
        (self.root/'evidence/day04/STOP').unlink()
        with self.assertRaises(ValueError):verify_product_startup(self.root)
    def test_duplicate_json_keys_rejected(self):
        self.manifest.write_text('{"schema":"pvz-product-startup/1","schema":"pvz-product-startup/1"}',encoding='utf-8')
        with self.assertRaises(ValueError):verify_product_startup(self.root)
    def test_old_freeze_unchanged_and_only_historical_test_differs(self):
        old=json.loads((ROOT/OLD_FREEZE).read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((ROOT/OLD_FREEZE).read_bytes()).hexdigest(),OLD_FREEZE_SHA256)
        changed=[p for p,h in old['artifacts'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        self.assertEqual(changed,['tests/test_goal_review.py'])
        backup=ROOT/'backups/continuity-source-binding-20261104/test_goal_review.py'
        self.assertEqual(hashlib.sha256(backup.read_bytes()).hexdigest(),old['artifacts']['tests/test_goal_review.py'])
    def test_actual_main_preflight_stops_before_configuration_and_runtime(self):
        module=importlib.import_module('day06_product_resource_live')
        def forbidden(*a,**k):raise AssertionError('unexpected access')
        with patch.multiple(module,ROOT=self.root,resolve_product_strategy=forbidden,load=forbidden,
                            create_product_strategy=forbidden,create_bucket_stream=forbidden,
                            run_goal_readonly=forbidden,PinnedWindowsCapture=forbidden), \
             patch.object(sys,'argv',['test','--preflight-only']):module.main()
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),['config','evidence','scripts','src','tests'])
    def test_real_main_checks_versions_before_config_on_tamper(self):
        module=importlib.import_module('day06_product_resource_live')
        path=self.root/'src/pvz_agent/tracking.py';path.write_bytes(path.read_bytes()+b'bad')
        with patch.multiple(module,ROOT=self.root,resolve_product_strategy=lambda: self.fail('config read'),
                            create_product_strategy=lambda *_:self.fail('provider'),
                            create_bucket_stream=lambda *_:self.fail('stream'),
                            run_goal_readonly=lambda **_:self.fail('runner'),
                            PinnedWindowsCapture=lambda *_:self.fail('capture')), \
             patch.object(sys,'argv',['test','--out','new-output']):
            with self.assertRaises(ValueError):module.main()
        self.assertFalse((self.root/'new-output').exists())

if __name__=='__main__':unittest.main()
