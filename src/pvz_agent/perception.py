"""Screenshot-only experimental perception with explicit abstention.

No input, strategy, game memory, cooldown-duration inference or plant identity
recognition. Similarity scores are not calibrated confidence probabilities.
"""
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature(image, size=(16, 20)):
    return np.asarray(image.convert('RGB').resize(size, Image.Resampling.BILINEAR),
                      dtype=np.float32).reshape(-1) / 255


def card_patch(image, slot):
    x = 90 + (slot - 1) * 51
    return image.crop((x + 4, 12, x + 43, 62))


def cell_patch(image, row, column):
    x = 40 + (column - .5) * 80
    y = 80 + (row - .5) * 85
    return image.crop((round(x - 31), round(y - 43), round(x + 31), round(y + 26)))


def digit_patches(image, roi):
    """Connected dark-ink components; reject overlapping glow and damaged glyphs."""
    crop = image.convert('RGB').crop(roi)
    rgb = np.asarray(crop)
    # A bright yellow sun crossing the number is an explicit occlusion.
    glow = (rgb[:, :, 0] > 215) & (rgb[:, :, 1] > 190) & (rgb[:, :, 2] < 115)
    if glow.mean() > .08:
        return [], 'sun_glow_occlusion'
    # Brown tray borders are dark too; keep near-neutral black glyph ink.
    channels=rgb.astype(np.int16)
    ink = (np.max(channels, axis=2) < 120) & (np.ptp(channels,axis=2)<50)
    seen = np.zeros(ink.shape, dtype=bool)
    components = []
    for y, x in zip(*np.where(ink)):
        if seen[y, x]:
            continue
        stack, points = [(int(x), int(y))], []
        seen[y, x] = True
        while stack:
            px, py = stack.pop()
            points.append((px, py))
            for nx, ny in ((px-1,py),(px+1,py),(px,py-1),(px,py+1)):
                if 0 <= ny < ink.shape[0] and 0 <= nx < ink.shape[1] and ink[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx] = True
                    stack.append((nx,ny))
        xs, ys = zip(*points)
        box = (min(xs), min(ys), max(xs)+1, max(ys)+1)
        if box[3]-box[1] >= 7 and len(points) >= 10:
            components.append(box)
    components.sort()
    if not 1 <= len(components) <= 4:
        return [], 'digit_component_count'
    glyphs=[]
    mask=Image.fromarray((ink*255).astype('uint8'))
    for box in components:
        if box[2]-box[0] > 16 or box[3]-box[1] > 21:
            return [], 'digit_shape_out_of_range'
        glyphs.append(np.asarray(mask.crop(box).resize((14,20),Image.Resampling.BILINEAR),dtype=np.float32).reshape(-1)/255)
    return glyphs, None


def classify(vector, vectors, labels, max_distance, min_margin):
    if len(labels) == 0:
        return {'value':None,'similarity':0.0,'reason':'no_template'}
    distances = np.mean(np.abs(vectors - vector), axis=1)
    by_class = {str(label): float(np.min(distances[labels == label])) for label in np.unique(labels)}
    ranked=sorted(by_class.items(),key=lambda pair:pair[1])
    label, distance=ranked[0]
    margin=ranked[1][1]-distance if len(ranked)>1 else 1.0
    accepted=distance <= max_distance and margin >= min_margin
    return {'value':label if accepted else None,
            'similarity':round(max(0.0,1-distance),4),
            'distance':round(distance,5),'margin':round(margin,5),
            'reason':None if accepted else 'out_of_distribution_or_ambiguous'}


class Perception:
    def __init__(self, model_file, config):
        self.config=config
        with np.load(model_file,allow_pickle=False) as data:
            self.model={k:data[k].copy() for k in data.files}
        self.provenance=json.loads(str(self.model['provenance']))
        if self.provenance['config'] != config:
            raise ValueError('Model/config mismatch; retain and rebuild a versioned atlas')
        self.model_sha256=fingerprint(model_file)

    def _match(self, family, vector):
        return classify(vector,self.model[family+'_x'],self.model[family+'_y'],
                        self.config[family+'_distance_max'],self.config[family+'_margin_min'])

    def observe(self, image, state_id, capture_time, *, card_costs=None, scene_hint=None):
        started=time.perf_counter()
        if image.size != (800,600):
            raise ValueError('Expected canonical 800x600 screenshot; normalize with reviewed source mapping first')
        if not state_id or not capture_time:
            raise ValueError('state_id and capture_time required')
        if card_costs is not None and (len(card_costs)!=10 or any(c is not None and (type(c) is not int or c<0) for c in card_costs)):
            raise ValueError('Need ten non-negative costs or null; never infer identities from slot index')
        scene=self._match('scene',feature(image,(40,30)))
        # A human scene hint can restrict processing, never override an uncertain detector.
        scene_allowed=scene['value']=='playing' and scene_hint in (None,'playing')
        status = 'playing' if scene_allowed else 'nonplaying_or_unknown'
        output={'schema':'pvz-observation/1','state_id':state_id,'capture_time':capture_time,
                'perceived_at':datetime.now(timezone.utc).isoformat(),
                'model_sha256':self.model_sha256,'scene':{**scene,'status':status},
                'sun':{'value':None,'similarity':0.0,'reason':'scene_not_verified'},
                'cards':[],'cells':[], 'input_sent':False,
                'action_eligible':False,
                'confidence_semantics':'uncalibrated_visual_similarity_not_probability'}
        if scene_allowed:
            glyphs,reason=digit_patches(image,self.config['sun_roi_xyxy'])
            digits=[self._match('digit',g) for g in glyphs]
            number=self._match('number',feature(image.crop(self.config['sun_roi_xyxy']),(44,17)))
            if digits and all(d['value'] is not None for d in digits) and number['value']==''.join(d['value'] for d in digits):
                value=int(''.join(d['value'] for d in digits))
                output['sun']={'value':value,'similarity':min([number['similarity']]+[d['similarity'] for d in digits]),'reason':None,'digits':digits,'number_check':number}
            else:
                output['sun']={'value':None,'similarity':0.0,'reason':reason or 'digit_and_number_disagree_or_unknown','digits':digits,'number_check':number}
        for slot in range(1,11):
            match=self._match('card',feature(card_patch(image,slot))) if scene_allowed else {'value':None,'similarity':0.,'reason':'scene_not_verified'}
            visual=match.pop('value')
            cost=card_costs[slot-1] if card_costs else None
            sun=output['sun']['value']
            # Card readiness is separate from location legality and hidden cooldown state.
            usable=False if scene_allowed and (visual=='dim' or cost is not None and sun is not None and sun<cost) else (
                True if scene_allowed and visual=='bright' and cost is not None and sun is not None and sun>=cost else None)
            output['cards'].append({'slot':slot,'visual_state':visual or 'unknown',
                'cost':cost,'usable_with_observed_resources':usable,
                'target_planting_legality':'not_evaluated','cooldown_remaining_seconds':None,**match})
        for row in range(1,7):
            for col in range(1,10):
                if scene_allowed:
                    group='water' if row in (3,4) else 'land'
                    match=classify(feature(cell_patch(image,row,col)), self.model[group+'_x'],self.model[group+'_y'],
                                   self.config['cell_distance_max'],self.config['cell_margin_min'])
                else:
                    match={'value':None,'similarity':0.,'reason':'scene_not_verified'}
                value=match.pop('value')
                occupied=None if value is None else value!='empty'
                base=(value in ('base_only','base_and_upper')) if value is not None and row in (3,4) else None
                upper=None if value is None else value in ('ground_occupied','base_and_upper')
                output['cells'].append({'row':row,'column':col,'occupancy':value or 'unknown',
                    'occupied':occupied,'carrier_present':base,'upper_present':upper,
                    'plant_identity':None,'planting_legality':'not_evaluated',**match})
        output['processing_ms']=round((time.perf_counter()-started)*1000,3)
        return output


def build_model(root, dataset, config, output):
    """Development split only. Held-out annotations cannot enter any atlas."""
    root=Path(root)
    banks={k:[] for k in ('digit','number','card','land','water','scene')}
    training_ids=[]
    skipped_digits=[]
    for sample in dataset['samples']:
        if sample['split']!='development':
            continue
        path=root/sample['image']
        if fingerprint(path)!=sample['sha256']:
            raise ValueError('Training image hash mismatch')
        im=Image.open(path).convert('RGB')
        labels=sample['labels']
        training_ids.append(sample['id'])
        banks['scene'].append((feature(im,(40,30)),labels['scene']))
        if labels['scene']!='playing':
            continue
        sun=labels['sun']
        if sun is not None:
            banks['number'].append((feature(im.crop(config['sun_roi_xyxy']),(44,17)),str(sun)))
            glyphs,reason=digit_patches(im,config['sun_roi_xyxy'])
            if reason is None and len(glyphs)==len(str(sun)):
                banks['digit'].extend(zip(glyphs,str(sun)))
            else:
                skipped_digits.append({'id':sample['id'],'reason':reason or 'label_component_count_mismatch'})
        for slot,label in enumerate(labels['cards'],1):
            if label in ('bright','dim'):
                banks['card'].append((feature(card_patch(im,slot)),label))
        for row,line in enumerate(labels['occupancy_rows'],1):
            for col,symbol in enumerate(line,1):
                if symbol in ('_','?'):
                    continue
                label='empty' if symbol=='.' else 'base_only' if symbol=='l' else 'base_and_upper' if symbol.isupper() else 'ground_occupied'
                group='water' if row in (3,4) else 'land'
                banks[group].append((feature(cell_patch(im,row,col)),label))
    arrays={}
    for group,items in banks.items():
        if not items:
            raise ValueError(f'No {group} training samples')
        arrays[group+'_x']=np.stack([x for x,_ in items])
        arrays[group+'_y']=np.array([str(y) for _,y in items])
    provenance={'training_ids':training_ids,'source_split':'development','skipped_digits':skipped_digits,
        'counts':{k:len(v) for k,v in banks.items()},'config':config,
        'created_at':datetime.now(timezone.utc).isoformat(),
        'limits':'Fixed-source screenshot template baseline; no independent multi-game accuracy claim.'}
    arrays['provenance']=np.array(json.dumps(provenance,ensure_ascii=False))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    return provenance
