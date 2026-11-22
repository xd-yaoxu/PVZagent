"""Development-only native partial templates; valid pixels stored at native width."""
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .boundary_perception import head_similarity, row_at
from .boundary_perception_v9 import CoarseBoundaryPerception
from .crossgame_perception import build_crossgame_model
from .perception import fingerprint


def build_edge_bank(root, dataset, config, output):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Retain model; select another revision')
    stage = output.with_name(output.stem+'-full-bank.npz')
    provenance = build_crossgame_model(root, dataset, config, stage)
    w, h = config['patch_wh']
    ax, ay = config['anchor_xy']
    arrays = {}
    refs = []
    for sample in dataset['samples']:
        if sample['split'] != 'development':
            continue
        path = root/sample['image']
        if fingerprint(path) != sample['sha256']:
            raise ValueError('Partial development image hash mismatch')
        with Image.open(path) as image:
            gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        for truth in sample['labels']['zombies']:
            x, y = truth['head_xy']
            width = 800-(x-ax)
            if not config['right_edge_search']['min_visible_width'] <= width < w:
                continue
            patch = gray[y-ay:y-ay+h, x-ax:800]
            if patch.shape != (h, width) or float(np.std(patch)) < config['right_edge_search']['template_std_min']:
                continue
            key = f'edge_{len(refs)}'
            arrays[key] = patch.copy()  # No fabricated columns in the stored template.
            refs.append({'key':key, 'sample_id':sample['id'], 'head_xy':[x, y],
                         'source_row':truth['row'], 'valid_width':width, 'pose':truth.get('pose', 'ordinary'),
                         'source_image_sha256':sample['sha256']})
    if not refs:
        raise ValueError('No native right-edge development examples')
    with np.load(stage, allow_pickle=False) as bundle:
        arrays.update({k:bundle[k].copy() for k in bundle.files})
    provenance.update(edge_template_refs=refs, partial_bank_completed_at=datetime.now(timezone.utc).isoformat(),
                      partial_source_scope='Development labels only; intentionally includes visible border points previously skipped as full templates.')
    arrays['edge_refs'] = np.array(json.dumps(refs))
    arrays['provenance'] = np.array(json.dumps(provenance))
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    return provenance


class EdgeBankPerception(CoarseBoundaryPerception):
    def __init__(self, model_file, config):
        super().__init__(model_file, config)
        if config['native_edge_bank']['source_rows_only'] is not True:
            raise ValueError('Native partial templates are currently limited to their labeled source row')
        self.edge_refs = json.loads(str(self.model['edge_refs']))

    def _search(self, gray):
        proposals = super()._search(gray)
        w, h = self.config['patch_wh']
        ax, ay = self.config['anchor_xy']
        settings = self.config['native_edge_bank']
        for i, ref in enumerate(self.edge_refs):
            row = ref['source_row']
            bands = self.config['submerged_head_bands_y'] if ref['pose'] == 'submerged' else self.config['head_bands_y']
            band = bands[row-1]
            if band is None:
                continue
            top, bottom = max(0, band[0]-ay), min(600, band[1]-ay+h)
            for width in self.config['right_edge_search']['visible_widths']:
                if width > ref['valid_width']:
                    continue
                partial = self.model[ref['key']][:, :width]
                x = 800-width+ax
                scores = cv2.matchTemplate(gray[top:bottom, 800-width:800], partial, cv2.TM_CCOEFF_NORMED)
                maxima = ((scores >= settings['threshold']) &
                          (scores >= cv2.dilate(scores, np.ones((9, 1), np.uint8))))
                for sy in np.where(maxima[:, 0])[0]:
                    y = int(sy+top+ay)
                    if row_at(y, [band]) is None:
                        continue
                    patch = gray[y-ay:y-ay+h, x-ax:800]
                    support = head_similarity(patch, partial, self.config['head_support_rows'])
                    if support < self.config['right_edge_search']['head_support_min']:
                        continue
                    proposals.append({'row':row, 'head_xy':[x, y], 'box_xyxy':[x-ax, y-ay, 800, y-ay+h],
                        'similarity':round(float(scores[sy, 0]), 5), 'template_index':i,
                        'template_bank':'native_right_edge', 'pose':ref['pose'],
                        'head_similarity':round(support, 5), 'truncated':True,
                        'visible_template_fraction':round(width/w, 5), 'visible_template_width':width,
                        'position_semantics':'partial_template_reference_with_unquantified_localization_uncertainty'})
        return proposals
