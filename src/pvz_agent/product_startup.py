"""Version check for the current product entry, not a readiness or security audit."""
import hashlib
import json
from pathlib import Path
import re

OLD_FREEZE = 'evidence/day06/goal-review-v1/freeze.json'
OLD_FREEZE_SHA256 = '78df4a0b4a13ef11694008019fe2917695b9774e659e01971b92d7e06b8749c7'
MANIFEST = 'evidence/product-card-strategy-v1/startup-manifest-v10.json'
PREVIOUS_MANIFEST = 'evidence/product-card-strategy-v1/startup-manifest-v9.json'
PREVIOUS_MANIFEST_SHA256 = '46666ad9d09cd75d440244a241b49b37788dac6b9f2da0d562d6260462ca83f1'
SELECTOR = 'src/pvz_agent/product_startup.py'
RESOURCE_ARTIFACTS = {
    'src/pvz_agent/resource_stream.py':
        '56862d98dcfad12c7e0a227bf7ba12daa521087ac4b23abbf19b7ffced67eecf',
    'evidence/product-review-race-v1/resource-freeze-v5.json':
        '15d67160f4d1c699563a6cbc4a98698acc7722bb03bfd13fabf970840bad303a',
}
REQUIRED = frozenset({
    'src/pvz_agent/goal_review.py', 'src/pvz_agent/goal_runtime.py',
    'tests/test_goal_review.py', 'scripts/day06_goal_review.py',
    'config/strategy-goal-review-v1.json',
    'src/pvz_agent/readonly_longrun.py', 'src/pvz_agent/observation_stream.py',
    'src/pvz_agent/vehicle_perception.py', 'scripts/day06_product_resource_live.py',
    'tests/test_tracking_continuity_binding.py', 'src/pvz_agent/tracking.py',
    'src/pvz_agent/continuity_frames.py', 'scripts/run_product_agent.py',
    'src/pvz_agent/product_startup.py', 'tests/test_product_startup.py',
    'src/pvz_agent/bound_resource_review.py', 'src/pvz_agent/resource_runtime.py',
    'src/pvz_agent/resource_review.py', 'src/pvz_agent/resource_stream.py',
    'src/pvz_agent/resource_threat_v2.py', 'src/pvz_agent/strategy_review.py',
    'tests/test_product_threat_gate.py', 'tests/test_bound_resource_review.py',
    'tests/test_resource_review.py',
    'evidence/day06/resource-review-v1/resource-freeze.json',
    'evidence/day06/bucket-angle-v1/freeze-v40.json',
    'evidence/product-review-race-v1/resource-freeze-v2.json',
    'evidence/product-startup-v1/startup-manifest.json',
    'evidence/product-review-race-v1/startup-manifest-v2.json',
    'evidence/product-review-race-v1/resource-freeze-v3.json',
    'tests/test_product_poll_ownership.py',
    'evidence/product-review-race-v1/startup-manifest-v3.json',
    'evidence/product-review-race-v1/resource-freeze-v4.json',
    'evidence/product-review-race-v1/revision-v3/unsupported-fixtures.json',
    'tests/test_product_strategy_context_version.py',
    'src/pvz_agent/readonly_pipeline.py', 'tests/test_capture_schedule_wait.py',
    'evidence/product-review-race-v1/startup-manifest-v4.json',
    'evidence/capture-schedule-v1/startup-manifest-v5.json',
    'src/pvz_agent/product_card_strategy.py', 'tests/test_product_card_strategy.py',
    'src/pvz_agent/card_identity.py', 'src/pvz_agent/loadout_runtime.py',
    'src/pvz_agent/loadout_preflight.py', 'src/pvz_agent/action_preflight.py',
    'src/pvz_agent/strategy_profile.py', 'src/pvz_agent/llm_advice.py',
    'src/pvz_agent/llm_provider.py', 'config/card-identity-v1.json',
    'evidence/day06/card-identity-v1/freeze-v1.json',
    'evidence/product-card-strategy-v1/startup-manifest-v6.json',
    'evidence/product-card-strategy-v1/startup-manifest-v7.json',
    'evidence/product-card-strategy-v1/startup-manifest-v8.json',
    PREVIOUS_MANIFEST,
    'src/pvz_agent/resource_sidebar.py', 'src/pvz_agent/resource_sidebar_tk.py',
    'scripts/day06_product_resource_live.py', 'src/pvz_agent/resource_history.py',
    'src/pvz_agent/run_diagnostics.py', *RESOURCE_ARTIFACTS,
})


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate manifest key')
        value[key] = item
    return value


def _sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _inside(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path outside project')
    return path


def verify_product_startup(root):
    root = Path(root).resolve()
    if not (root / 'evidence/day04/STOP').exists():
        raise ValueError('STOP must remain latched')
    with _inside(root, MANIFEST).open(encoding='utf-8') as stream:
        data = json.load(stream, object_pairs_hook=_unique_object)
    if type(data) is not dict or data.get('schema') != 'pvz-product-startup/10':
        raise ValueError('Invalid product startup manifest')
    if data.get('historical_freeze') != {'path': OLD_FREEZE, 'sha256': OLD_FREEZE_SHA256}:
        raise ValueError('Historical freeze binding changed')
    if _sha256(_inside(root, OLD_FREEZE)) != OLD_FREEZE_SHA256:
        raise ValueError('Historical freeze changed')
    previous_binding = {'path': PREVIOUS_MANIFEST, 'sha256': PREVIOUS_MANIFEST_SHA256}
    if data.get('previous_manifest') != previous_binding:
        raise ValueError('Previous startup manifest binding changed')
    previous_bytes = _inside(root, PREVIOUS_MANIFEST).read_bytes()
    if hashlib.sha256(previous_bytes).hexdigest() != PREVIOUS_MANIFEST_SHA256:
        raise ValueError('Previous startup manifest changed')
    expected = json.loads(previous_bytes.decode('utf-8'), object_pairs_hook=_unique_object)
    expected['schema'] = 'pvz-product-startup/10'
    expected['previous_manifest'] = previous_binding
    artifacts = data.get('artifacts')
    if type(artifacts) is not dict or set(artifacts) != REQUIRED:
        raise ValueError('Required product startup artifact set changed')
    # Preserve the frozen v9 digests except for the reviewed resource loader
    # and selector. Bind the new inner freeze and retain the previous chain.
    expected['artifacts'][PREVIOUS_MANIFEST] = PREVIOUS_MANIFEST_SHA256
    expected['artifacts'].update(RESOURCE_ARTIFACTS)
    # Keep the existing manifest-supplied check of the selector itself.
    expected['artifacts'][SELECTOR] = artifacts[SELECTOR]
    if data != expected:
        raise ValueError('Product startup manifest differs from reviewed revision')
    for rel, digest in artifacts.items():
        if type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid artifact digest')
        if _sha256(_inside(root, rel)) != digest:
            raise ValueError('Product startup artifact changed: ' + rel)
    return {'scope': 'file version check only', 'artifacts_checked': len(artifacts)}
