"""Read-only v2 state gate for the product resource supervisor.

Only published snapshots are available to its reviewer. This checks structured
field integrity, not capture authenticity or actual absence of threats.
"""
import copy
import math

from .continuity_frames import ContinuityFrames


def validate_threat_context(value):
    if type(value) is not dict:raise ValueError('state unknown')
    vehicle=value.get('vehicles');zombies=value.get('zombies')
    vehicle_keys=('candidates','tracks','uncertain_candidates')
    zombie_keys=('candidates','uncertain_edge_presence','uncertain_water_candidates')
    if (type(vehicle) is not dict or any(type(vehicle.get(k)) is not list for k in vehicle_keys)
            or type(zombies) is not dict or any(type(zombies.get(k)) is not list for k in zombie_keys)):
        raise ValueError('threat channels unknown')
    for k in vehicle_keys:
        if any(type(item) is not dict for item in vehicle[k]):raise ValueError('vehicle item invalid')
    for k in zombie_keys[1:]:
        if any(type(item) is not dict for item in zombies[k]):raise ValueError('uncertain item invalid')
    for item in zombies['candidates']:
        if type(item) is not dict:raise ValueError('candidate invalid')
        row=item.get('row');point=item.get('head_xy')
        if (type(row) is not int or not 1<=row<=6 or type(point) not in (list,tuple) or len(point)!=2
                or any(type(n) not in (int,float) or not math.isfinite(n) for n in point)
                or not 0<=point[0]<800 or not 0<=point[1]<600):
            raise ValueError('candidate geometry invalid')


class ThreatCheckedContinuityFrames(ContinuityFrames):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self._strategy_context_version=0

    def _after_publish_locked(self,value):
        vehicle=value['vehicles'];zombies=value['zombies']
        if (any(vehicle[k] for k in ('candidates','tracks','uncertain_candidates'))
                or any(zombies[k] for k in ('uncertain_edge_presence','uncertain_water_candidates'))):
            self._strategy_context_version+=1

    def publish_observation(self,ticket,state,**kwargs):
        try:
            value=copy.deepcopy(state)
            validate_threat_context(value)
        except Exception:
            self.session_changed('threat_context_invalid')
            try:rejected=copy.deepcopy(state) if type(state) is dict else {}
            except Exception:rejected={}
            rejected['readonly_runtime']={'publication_at_completion':'continuity_rejected'}
            return rejected
        return super().publish_observation(ticket,value,**kwargs)
