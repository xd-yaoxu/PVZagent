"""Build explicitly frozen v40 fields/objects plus v2 sun appearance cues."""
import hashlib
import json
from pathlib import Path
from .bucket_readonly import create_bucket_stream
from .sun_cues import ResourceCueStream,SunCuePerception

HISTORICAL_RESOURCE_FREEZE='evidence/day06/resource-review-v1/resource-freeze.json'
HISTORICAL_RESOURCE_SHA256='a257c80c76c1c9f4c315a18d00538a78cc5afff2d8305af4f0a71ff3f57b207d'
PREVIOUS_PRODUCT_RESOURCE_FREEZE='evidence/product-review-race-v1/resource-freeze-v3.json'
PREVIOUS_PRODUCT_RESOURCE_SHA256='b1cfb8309baa4174c2e0ec02f7ae1ffd43b4372ba8c339ebcd3b7b86e44d5208'
PRODUCT_RESOURCE_V4='evidence/product-review-race-v1/resource-freeze-v4.json'
PRODUCT_RESOURCE_V4_SHA256='f2731140f35bee83e8fe94c9a172feb9ec1d31930617e4e03b8c52c198078aa3'
PRODUCT_RESOURCE_FREEZE='evidence/product-review-race-v1/resource-freeze-v5.json'
REVIEWED_RUNTIME_SHA256='afcea519adcd9f3b10fd2b31cefc61fc6f4347d645edbbb1f0db049029d5f980'


def create_resource_stream(root):
    return create_product_resource_stream(root)


def create_product_resource_stream(root):
    """Current product manifest; retain and bind the unmodified historical freeze."""
    root=Path(root).resolve()
    historical=(root/HISTORICAL_RESOURCE_FREEZE).read_bytes()
    if hashlib.sha256(historical).hexdigest()!=HISTORICAL_RESOURCE_SHA256:raise ValueError('Historical resource freeze changed')
    old=json.loads(historical)
    previous=(root/PREVIOUS_PRODUCT_RESOURCE_FREEZE).read_bytes()
    if hashlib.sha256(previous).hexdigest()!=PREVIOUS_PRODUCT_RESOURCE_SHA256:raise ValueError('Previous resource freeze changed')
    previous_data=json.loads(previous)
    if previous_data.get('historical_sha256')!=HISTORICAL_RESOURCE_SHA256 or set(previous_data['artifacts'])!=set(old['artifacts']):
        raise ValueError('Previous resource freeze binding invalid')
    v4=(root/PRODUCT_RESOURCE_V4).read_bytes()
    if hashlib.sha256(v4).hexdigest()!=PRODUCT_RESOURCE_V4_SHA256:
        raise ValueError('Resource v4 predecessor changed')
    expected=json.loads(v4)
    if (expected.get('schema')!='pvz-product-resource/4'
            or expected.get('historical_sha256')!=HISTORICAL_RESOURCE_SHA256
            or expected.get('previous_sha256')!=PREVIOUS_PRODUCT_RESOURCE_SHA256
            or set(expected['artifacts'])!=set(old['artifacts'])):
        raise ValueError('Resource v4 predecessor binding invalid')
    freeze=json.loads((root/PRODUCT_RESOURCE_FREEZE).read_text(encoding='utf-8'))
    artifacts=freeze.get('artifacts')
    if type(artifacts) is not dict or set(artifacts)!=set(old['artifacts']):raise ValueError('Product resource artifact set changed')
    expected.update(schema='pvz-product-resource/5', previous_sha256=PRODUCT_RESOURCE_V4_SHA256)
    expected['artifacts']['src/pvz_agent/resource_runtime.py']=REVIEWED_RUNTIME_SHA256
    expected['artifacts']['src/pvz_agent/resource_stream.py']=artifacts['src/pvz_agent/resource_stream.py']
    if freeze!=expected:raise ValueError('Resource v5 freeze differs from reviewed revision')
    for rel,digest in artifacts.items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or type(digest) is not str or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Product resource artifact changed')
    cfg=json.loads((root/'config/sun-cues-v2.json').read_text(encoding='utf-8'))
    return ResourceCueStream(create_bucket_stream(root),SunCuePerception(root,cfg))
