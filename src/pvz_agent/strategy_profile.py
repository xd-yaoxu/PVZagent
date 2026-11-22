"""Opt-in scalar timing only; payloads, credentials and headers are never logged."""
import json
import time
from .strategy_review import StrategyProvider,recommend_strategy
from .llm_provider import http_transport


class ProfiledStrategyProvider(StrategyProvider):
    def __init__(self,config,*,transport=http_transport,key_reader=None,clock=time.perf_counter):
        self.profiles=[];self.clock=clock;self._active=None
        def measured(url,body,key,timeout):
            tick=clock()
            # Only the serialized length is retained; this is not server latency.
            payload_bytes=len(json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8'))
            serialization_ms=(clock()-tick)*1000
            started=clock()
            try:return transport(url,body,key,timeout)
            finally:
                self._active.update(transport_roundtrip_ms=(clock()-started)*1000,
                    payload_utf8_bytes=payload_bytes,size_measurement_ms=serialization_ms)
        kwargs=dict(transport=measured)
        if key_reader is not None:kwargs['key_reader']=key_reader
        super().__init__(config,**kwargs)

    def complete(self,*args,**kwargs):
        if self._active is not None:raise RuntimeError('One profiled request at a time')
        self._active={};started=self.clock()
        try:return super().complete(*args,**kwargs)
        finally:
            self._active['provider_complete_ms']=(self.clock()-started)*1000
            self.profiles.append(self._active);self._active=None


def recommend_profiled(provider,root,state):
    started=provider.clock();before=len(provider.profiles)
    result=recommend_strategy(provider,root,state)
    total=(provider.clock()-started)*1000
    if len(provider.profiles)!=before+1:raise RuntimeError('Exactly one transport profile required')
    profile=dict(provider.profiles[-1]);profile['recommendation_total_ms']=total
    profile['outside_provider_ms']=max(0.,total-profile['provider_complete_ms'])
    profile['limits']='Client measurements only: transport includes local HTTP/JSON, network, service queue and generation. No server-stage breakdown or isolated model compute measurement.'
    return dict(result,client_timing=profile)
