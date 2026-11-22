"""Color-aware counter ink, checked against the frozen neutral-ink reader."""
from datetime import datetime, timezone
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .crossgame_field_perception import CrossgameFieldPerception
from .perception import classify, fingerprint


def counter_glyphs(image, config, threshold):
    rgb = np.asarray(image.convert('RGB').crop(config['sun_roi_xyxy'])).astype('int16')
    glow = (rgb[:,:,0] > 215) & (rgb[:,:,1] > 190) & (rgb[:,:,2] < 115)
    if glow.mean() > .08:
        return [], 'sun_glow_occlusion'
    # Brown tray shadows have more red than green. Preserve genuinely dark ink
    # even when compression adds a little chroma, plus neutral antialiased ink.
    color_ok = ((rgb.max(2) < config['counter_dark_core_max']) |
                (rgb[:,:,0] - rgb[:,:,1] <= config['counter_red_green_max']))
    ink = ((rgb.max(2) < threshold) & (np.ptp(rgb, axis=2) < 50) & color_ok).astype('uint8')
    ink[:config['sun_ink_top_rows_ignore']] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
    for i, (x,y,w,h,area) in enumerate(stats[1:], 1):
        if h <= 2 and area <= 5:
            ink[labels == i] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
    groups = []
    for left, right in sorted((int(x),int(x+w)) for x,y,w,h,area in stats[1:] if area > 0):
        if groups and left < groups[-1][1]:
            groups[-1][1] = max(right, groups[-1][1])
        else:
            groups.append([left, right])
    boxes = []
    for x1, x2 in groups:
        ys = np.flatnonzero(ink[:,x1:x2].any(1)); y1,y2 = int(ys[0]),int(ys[-1]+1)
        if x2-x1 > 16 or y2-y1 < 7 or ink[y1:y2,x1:x2].sum() < 8:
            return [], 'incomplete_or_merged_glyph'
        boxes.append((x1,y1,x2,y2))
    if not 1 <= len(boxes) <= 4:
        return [], 'digit_component_count'
    mask = Image.fromarray(ink*255)
    return [np.asarray(mask.crop(b).resize((14,20),Image.Resampling.BILINEAR),
                       dtype='float32').reshape(-1)/255 for b in boxes], None


def combine_counter_reads(legacy, color):
    """Never resolve disagreement by reader priority or historical spending."""
    readers = {'neutral_ink':legacy, 'color_ink':color}
    accepted = [r for r in readers.values() if r['value'] is not None]
    values = {r['value'] for r in accepted}
    value = next(iter(values)) if len(values) == 1 else None
    return dict(value=value, similarity=min((r['similarity'] for r in accepted), default=0.) if value is not None else 0.,
                reason=None if value is not None else 'counter_readers_disagree' if len(values)>1 else 'counter_consensus_missing',
                readers=readers, method='two_ink_masks_each_with_threshold_consensus',
                evidence_scope='Both readers use this same screenshot; agreement is not independent validation.',
                number_template_required=False)


def build_counter_fields(root, config, output):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Retain model')
    if fingerprint(root/config['counter_parent_model']) != config['counter_parent_sha256']:
        raise ValueError('Counter parent changed')
    with np.load(root/config['counter_parent_model'], allow_pickle=False) as bundle:
        arrays = {k:bundle[k].copy() for k in bundle.files}
    data = json.loads((root/config['counter_training_dataset']).read_text(encoding='utf-8'))
    xs, ys, refs = [], [], []
    permitted = {'evidence/day02/gameplay-3-7.mp4', 'evidence/day06/new-games-20261023/game-a.mp4'}
    for sample in data['samples']:
        if sample['split'] != 'development' or sample['labels']['scene'] != 'playing':
            continue
        if sample['source'] not in permitted:
            raise ValueError('Only old and A development sources permitted')
        if fingerprint(root/sample['image']) != sample['sha256']:
            raise ValueError('Development image changed')
        sun = sample['labels']['sun']
        if sun is None:
            continue
        with Image.open(root/sample['image']) as source:
            image = source.convert('RGB')
        if image.size != (800,600):
            raise ValueError('Canonical screenshot required')
        for threshold in config['glyph_thresholds']:
            patches, reason = counter_glyphs(image, config, threshold)
            if reason is None and len(patches) == len(str(sun)):
                for i,(patch,digit) in enumerate(zip(patches,str(sun))):
                    xs.append(patch); ys.append(digit)
                    refs.append(dict(sample_id=sample['id'],sha256=sample['sha256'],threshold=threshold,digit_index=i))
    if not xs:
        raise ValueError('Development digit templates required')
    provenance = json.loads(str(arrays['provenance']))
    provenance.update(config=config, counter_digit_refs=refs,
                      counter_training_dataset_sha256=fingerprint(root/config['counter_training_dataset']),
                      counter_parent_sha256=config['counter_parent_sha256'],
                      counter_completed_at=datetime.now(timezone.utc).isoformat(),
                      counter_limits='Only old and A development. B used for known diagnostics, never templates. Card and cell arrays inherited unchanged from fields19.')
    arrays.update(color_digit_x=np.stack(xs),color_digit_y=np.array(ys),
                  provenance=np.array(json.dumps(provenance)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return provenance


class CounterFieldPerception(CrossgameFieldPerception):
    def read_color_sun(self, image):
        reads = []
        for threshold in self.config['glyph_thresholds']:
            patches, reason = counter_glyphs(image, self.config, threshold)
            matches = [classify(p,self.model['color_digit_x'],self.model['color_digit_y'],
                                self.config['field_digit_distance_max'],self.config['field_digit_margin_min']) for p in patches]
            for match in matches:
                limit = self.config.get('digit_distance_overrides',{}).get(match['value'])
                if limit is not None and match['distance'] > limit:
                    match.update(value=None,reason='digit_specific_shape_distance')
            text = ''.join(m['value'] for m in matches) if matches and all(m['value'] is not None for m in matches) else None
            if text is not None and len(text)>1 and text.startswith('0'):
                text=None; reason='leading_zero_or_artifact'
            reads.append(dict(threshold=threshold,text=text,reason=reason,matches=matches))
        texts = {r['text'] for r in reads if r['text'] is not None}
        accepted = len(texts)==1 and sum(r['text'] is not None for r in reads)>=2
        return dict(value=int(next(iter(texts))) if accepted else None,
                    similarity=min((m['similarity'] for r in reads if r['text'] is not None for m in r['matches']),default=0.) if accepted else 0.,
                    reason=None if accepted else 'glyph_consensus_missing_or_conflicting',threshold_reads=reads,
                    method='color_ink_old_and_A_digit_templates')

    def read_sun(self, image):
        return combine_counter_reads(super().read_sun(image), self.read_color_sun(image))
