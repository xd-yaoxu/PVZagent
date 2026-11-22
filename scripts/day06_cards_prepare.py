"""Freeze reviewed A-only icon/price references before diagnostics."""
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
from day06_water_audit import load, write, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evidence/day06/card-identity-v1'
DATA = ROOT / 'data/day06/card-identity-v1'
if __name__ == '__main__':
    assert not BASE.exists() and not DATA.exists()
    BASE.mkdir(parents=True); DATA.mkdir(parents=True)
    source = 'data/day06/crossgame-v1/a/images/120.000.png'
    reviewed = load(ROOT/'data/day06/crossgame-v1/run-cards.json')['runs'][0]
    ids = ['sunflower','lily_pad','potato_mine','peashooter','cherry_bomb','repeater','tangle_kelp','squash','wallnut','spikeweed']
    catalog = dict(schema='pvz-reviewed-card-catalog/1', created_at=datetime.now(timezone.utc).isoformat(),
                   review='Assistant visual review of native A120 image and existing run-cards record; known-source, not independent labels.',
                   reference=dict(image=source,image_sha256=sha(ROOT/source),run_id='human-20261023-a',timestamp_seconds=120,split='development'),
                   cards=[dict(plant_id=pid,display_name=name,display_cost=cost,reference_slot=i)
                          for i,(pid,(name,cost)) in enumerate(zip(ids,reviewed['cards_left_to_right']),1)])
    write(DATA/'catalog-v1.json', catalog)
    config = dict(schema='pvz-card-identity-config/1',version=1,catalog='data/day06/card-identity-v1/catalog-v1.json',
                  catalog_sha256=sha(DATA/'catalog-v1.json'),icon_roi=[5,19,43,61],price_roi=[8,64,41,76],
                  search_radius=3,icon_similarity_min=.92,identity_margin_min=.06,price_similarity_min=.90,price_margin_min=.04,
                  confidence_semantics='Uncalibrated similarity; not a correctness probability.',
                  status='explicit_experiment_not_default_not_action_eligible')
    write(ROOT/'config/card-identity-v1.json',config)
    # Reviewed alternate known layout. Normalize this derived diagnostic image only;
    # never use this crop as a desktop/input coordinate map or training reference.
    alternate='evidence/day01/video-analysis/coarse-0000.00s.jpg'
    with Image.open(ROOT/alternate) as image:
        image.crop((113,35,789,542)).resize((800,600),Image.Resampling.LANCZOS).save(DATA/'alternate-day01.png')
    write(DATA/'alternate-day01.json',dict(source=alternate,source_sha256=sha(ROOT/alternate),
        crop_xyxy=[113,35,789,542],output='data/day06/card-identity-v1/alternate-day01.png',output_sha256=sha(DATA/'alternate-day01.png'),
        review='Known day01 alternative ten-slot order, visually reviewed before model output; single assistant, not independent validation.',
        expected_ids=['sunflower','wallnut','potato_mine',None,'repeater','lily_pad','tangle_kelp',None,None,None],
        out_of_catalog_slots=[4,8,9,10],known_display_costs=[50,50,25,150,200,25,25,125,125,125],
        not_training=True))
    artifacts=['src/pvz_agent/card_identity.py','scripts/day06_cards_prepare.py','config/card-identity-v1.json',
               'data/day06/card-identity-v1/catalog-v1.json',source]
    write(BASE/'freeze-v1.json',dict(created_at=datetime.now(timezone.utc).isoformat(),artifacts={p:sha(ROOT/p) for p in artifacts}))
    print(dict(reference_cards=10,training_images=1,alternate_unknown_slots=4))
