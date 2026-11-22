"""Atomic, isolated snapshots over the frozen latest-frame handoff."""
import copy
from .latest_frames import LatestFrames,finite_nonnegative


class SnapshotFrames(LatestFrames):
    def publish_observation(self,ticket,state,*,captured_at,now,max_age,input_age_ms):
        if not all(finite_nonnegative(v) for v in [captured_at,now,max_age,input_age_ms]) or max_age==0:
            raise ValueError('Finite observation clocks and positive age required')
        with self._condition:
            if self._error is not None:result='source_failed'
            elif ticket.generation!=self._generation:result='superseded'
            elif not 0<=now-captured_at<=max_age:result='expired'
            else:result='published'
            value=copy.deepcopy(state)
            value['readonly_runtime']=dict(generation=ticket.generation,input_age_ms=input_age_ms,
                result_age_ms=(now-captured_at)*1000,publication_at_completion=result,
                expires_monotonic=captured_at+max_age,
                currentness_limits='Archival snapshot; query live registry with generation and TTL. Never execution eligibility.')
            if result=='published':self._published=(ticket.generation,value,captured_at,captured_at+max_age)
            # Archive caller and subsequent readers never share the registry's dictionary.
            return copy.deepcopy(value)

    def publish(self,ticket,value,**kwargs):
        return super().publish(ticket,copy.deepcopy(value),**kwargs)

    def current(self,now):
        with self._condition:return copy.deepcopy(super().current(now))
