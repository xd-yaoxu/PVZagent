"""Read-only timestamped state assembly for later decision/log integration."""
from datetime import datetime, timezone
import math
import time
from .tracking import ShortTermTracker


class ObservationStream:
    def __init__(self,detector,field_perception=None):
        self.detector=detector;self.fields=field_perception
        self.tracker=ShortTermTracker(detector.config['tracker'])

    def observe(self,image,*,run_id,state_id,capture_time,frame_time_seconds,evidence,tracking_epoch=None):
        if tracking_epoch is not None and (type(tracking_epoch) is not int or tracking_epoch<0):
            raise ValueError('Tracking epoch must be a nonnegative integer')
        if tracking_epoch is not None and (not isinstance(run_id,str) or not run_id.strip()):
            raise ValueError('Nonempty source run_id required')
        if not capture_time or not evidence.get('image_sha256') or not evidence.get('image'):
            raise ValueError('Capture time and original image evidence required')
        if type(frame_time_seconds) not in (float,int) or not math.isfinite(frame_time_seconds) or frame_time_seconds<0:
            raise ValueError('Explicit numeric frame clock required')
        start=time.perf_counter()
        detected=self.detector.observe(image)
        tracking_run_id=run_id if tracking_epoch is None else f'{run_id}/continuity-{tracking_epoch}'
        assoc=self.tracker.update(run_id=tracking_run_id,state_id=state_id,frame_time=frame_time_seconds,
                                 candidates=detected['candidates'],scene=detected['scene']['value'])
        for c in assoc['candidates']:
            c['detection_id']=f'{state_id}:{c["detection_id"]}'
        for i,event in enumerate(assoc['events'],1):
            event.update(event_id=f'{state_id}:e{i}',capture_time=capture_time,evidence=dict(evidence))
            if event.get('detection_id'):
                event['detection_id']=f'{state_id}:{event["detection_id"]}'
        result={'schema':'pvz-state/2','run_id':run_id,'state_id':state_id,
            'capture_time':capture_time,'frame_time_seconds':frame_time_seconds,
            'perceived_at':datetime.now(timezone.utc).isoformat(),'scene':detected['scene'],
            'zombies':{**detected,**assoc},'evidence':dict(evidence),
            'input_sent':False,'action_eligible':False,
            'limitations':['Partial visible-headed candidates; no empty-lane assurance.',
                           'Day05 fields remain experimental and cannot authorize input.']}
        if tracking_epoch is not None:
            result['tracking_context']=dict(epoch=tracking_epoch,source_run_id=run_id,tracking_run_id=tracking_run_id)
        if self.fields is not None:
            # New guard can veto Day05 processing, never force its uncertain scene to playing.
            result['fields']=self.fields.observe(image,state_id,capture_time,
                scene_hint=None if detected['scene']['value']=='playing' else 'nonplaying')
        result['processing_ms']=round((time.perf_counter()-start)*1000,3)
        return result
