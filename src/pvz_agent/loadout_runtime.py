"""Opt-in composition of frozen resource stream and per-frame card evidence."""
import hashlib
import json
from pathlib import Path
from .card_identity import CardIdentityPerception,CardIdentityStream
from .resource_stream import create_resource_stream


def create_card_stream(root):
    root=Path(root).resolve()
    freeze=json.loads((root/'evidence/day06/card-identity-v1/freeze-v1.json').read_text(encoding='utf-8'))
    for rel,digest in freeze['artifacts'].items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen card artifact changed')
    config=json.loads((root/'config/card-identity-v1.json').read_text(encoding='utf-8'))
    return CardIdentityStream(create_resource_stream(root),CardIdentityPerception(root,config))
