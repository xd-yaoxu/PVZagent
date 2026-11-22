"""Current product model selection, separate from frozen experiment configs."""
import json
from pathlib import Path

from .llm_provider import CompatibleProvider
from .strategy_profile import ProfiledStrategyProvider


CURRENT_MODEL = 'gpt-6.1-terra'
CURRENT_SELECTION = 'config/product-llm-current.json'


def load_current_config(root, kind):
    if kind not in ('provider', 'strategy'):
        raise ValueError('unknown product provider kind')
    root = Path(root).resolve()
    selection = json.loads((root / CURRENT_SELECTION).read_text(encoding='utf-8-sig'))
    if selection.get('model') != CURRENT_MODEL:
        raise ValueError('current product model mismatch')
    relative = selection.get(kind + '_config')
    if not isinstance(relative, str) or not relative:
        raise ValueError('missing product provider configuration')
    path = (root / relative).resolve()
    if not path.is_relative_to(root / 'config'):
        raise ValueError('product configuration must be inside config')
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    if cfg.get('model') != CURRENT_MODEL:
        raise ValueError('product provider model mismatch')
    return cfg


def current_provider(root, kind, *, transport=None, key_reader=None):
    cfg = load_current_config(root, kind)
    options = {}
    if transport is not None:
        options['transport'] = transport
    if key_reader is not None:
        options['key_reader'] = key_reader
    provider_class = CompatibleProvider if kind == 'provider' else ProfiledStrategyProvider
    return provider_class(cfg, **options)
