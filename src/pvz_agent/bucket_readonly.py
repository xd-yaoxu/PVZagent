"""Explicit frozen v40 stream factory; no capture, networking or input here."""
import hashlib
import json
from pathlib import Path
from .bucket_water_interior import InteriorBucketWaterPerception
from .edge_context import EdgeObservationStream
from .aligned_hud_guard import GuardedAlignedHudPerception


def create_bucket_stream(root,*,with_fields=True):
    root=Path(root).resolve()
    if type(with_fields) is not bool:raise ValueError('Explicit boolean field choice required')
    if not (root/'evidence/day04/STOP').exists():raise ValueError('STOP must remain latched')
    frozen=json.loads((root/'evidence/day06/bucket-angle-v1/freeze-v40.json').read_text(encoding='utf-8'))
    for rel,digest in frozen['artifacts'].items():
        with (root/rel).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:raise ValueError('Frozen bucket artifact changed')
    cfg=json.loads((root/'config/zombie-perception-bucket-v40.json').read_text(encoding='utf-8'))
    fields=None
    if with_fields:
        fc=json.loads((root/'config/perception-fields-alignment-v23.json').read_text(encoding='utf-8'))
        fields=GuardedAlignedHudPerception(root/fc['model'],fc)
    return EdgeObservationStream(InteriorBucketWaterPerception(root/cfg['model'],cfg),fields)
