"""Continuity epochs attached to the real capture/perception snapshot registry.

Ordinary replacement of a pending frame is not a continuity failure. Observable
capture/scene/session failures are; callbacks always run outside registry locks.
"""
import copy
import json
import math
import time

from .latest_frames import finite_nonnegative
from .readonly_pipeline import capture_valid
from .snapshot_frames import SnapshotFrames


def geometry_identity(meta):
    windows=meta['windows'];mapping=meta['display_mapping']
    if type(windows) is not dict or type(mapping) is not dict:
        raise ValueError('Window and mapping metadata required')
    def rectangle(value, *, allow_float=False):
        types=(int,float) if allow_float else (int,)
        if (type(value) is not dict or any(type(value.get(k)) not in types or not math.isfinite(value[k])
                                         for k in ('x','y','width','height'))
                or value['width']<=0 or value['height']<=0):
            raise ValueError('Valid rectangle required')
    for key in ('game','scaler'):
        window=windows.get(key)
        if window is None and key=='scaler':continue
        if (type(window) is not dict or type(window.get('hwnd')) is not int or window['hwnd']<=0
                or type(window.get('process')) is not str or not window['process']
                or type(window.get('dpi')) is not int or window['dpi']<=0):
            raise ValueError('Valid window metadata required')
        rectangle(window.get('client_screen'))
    if type(mapping.get('mode')) is not str or not mapping['mode']:
        raise ValueError('Mapping mode required')
    rectangle(mapping.get('game_viewport_screen'),allow_float=True)
    value=dict(windows={key:({field:windows[key][field] for field in
        ['hwnd','process','client_screen','dpi']} if windows.get(key) else None) for key in ['game','scaler']},
        mapping={key:mapping[key] for key in ['mode','game_viewport_screen']},pixels=meta['image_pixels'])
    if value['pixels'] != [800,600] or value['windows']['game'] is None:
        raise ValueError('Canonical game geometry required')
    return json.dumps(value,sort_keys=True,allow_nan=False)


class ContinuityFrames(SnapshotFrames):
    def __init__(self,*,clock=time.monotonic,max_capture_gap=2.):
        super().__init__()
        if not finite_nonnegative(max_capture_gap) or not 0<max_capture_gap<=2:
            raise ValueError('Capture continuity gap must be at most2s')
        self.clock=clock;self.max_capture_gap=max_capture_gap;self._epoch=0
        self._last_capture=None;self._identity=None;self._source_run=None
        self._continuity_events=[];self._listener=None;self._gap_reported=False
        self._source_ended=False

    def bind_invalidator(self,listener):
        with self._condition:
            if self._listener is not None:raise RuntimeError('One invalidator required')
            self._listener=listener

    def _break_locked(self,reason,state_id=None):
        self._epoch+=1;self._published=None
        self._continuity_events.append(dict(kind='continuity_invalidated',epoch=self._epoch,
            reason=reason,state_id=state_id,monotonic=self.clock()))
        return self._listener

    @staticmethod
    def _notify(listener,reason):
        if listener is not None:listener(reason)

    def _after_publish_locked(self,value):
        """Internal subclass hook; called only for an accepted publication."""
        pass

    def put(self,item):
        if type(item) is not dict or type(item.get('metadata')) is not dict:
            with self._condition:
                listener=self._break_locked('capture_metadata_invalid')
            self._notify(listener,'capture_metadata_invalid')
            raise ValueError('Capture metadata required')
        notification=None;reason=None
        with self._condition:
            meta=item['metadata'];sid=meta.get('state_id')
            try:
                timestamp=meta['monotonic_ns']/1e9
                if type(sid) is not str or not sid:
                    reason='capture_metadata_invalid'
                elif type(meta['monotonic_ns']) is not int or not finite_nonnegative(timestamp) or not 0<=self.clock()-timestamp<=1.5:
                    reason='capture_timestamp_invalid'
                elif (any(meta[key] is not True for key in ['geometry_stable_during_capture','foreground_stable','foreground_game_or_scaler'])
                      or meta['display_mapping']['validated'] is not True or not capture_valid(meta)):
                    reason='capture_foreground_or_mapping_invalid'
                else:
                    identity=geometry_identity(meta)
                    if self._last_capture is not None and not 0<timestamp-self._last_capture<=self.max_capture_gap:
                        reason='capture_gap_or_nonmonotonic'
                    elif self._identity is not None and identity!=self._identity:
                        reason='window_identity_or_geometry_changed'
                    self._identity=identity
                if reason is None or reason in ['capture_gap_or_nonmonotonic','window_identity_or_geometry_changed']:
                    self._last_capture=timestamp;self._gap_reported=False
            except Exception:reason='capture_metadata_invalid'
            if reason:notification=self._break_locked(reason,sid)
            stamped=dict(item,_source_binding=(sid,meta.get('monotonic_ns')),
                _continuity_epoch=self._epoch,_continuity_capture_valid=reason in
                [None,'capture_gap_or_nonmonotonic','window_identity_or_geometry_changed'])
            result=super().put(stamped)
        self._notify(notification,reason)
        return result

    def publish_observation(self,ticket,state,**kwargs):
        notification=None;reason=None
        with self._condition:
            try:
                sid,ns=ticket.item['_source_binding']
                if (type(sid) is not str or not sid or type(ns) is not int or ns<0
                        or type(state) is not dict or state['state_id']!=sid
                        or type(state['frame_time_seconds']) not in (int,float)
                        or type(kwargs['captured_at']) not in (int,float)):
                    raise ValueError('invalid binding')
                captured=ns/1e9
                if (not math.isfinite(captured) or not math.isfinite(state['frame_time_seconds'])
                        or not math.isfinite(kwargs['captured_at'])
                        or state['frame_time_seconds']!=captured or kwargs['captured_at']!=captured):
                    raise ValueError('invalid binding')
            except (KeyError,TypeError,ValueError,AttributeError,OverflowError):
                notification=self._break_locked('source_binding_invalid')
                reason='source_binding_invalid'
            if reason:
                value=copy.deepcopy(state) if type(state) is dict else {}
                value['readonly_runtime']=dict(publication_at_completion='continuity_rejected')
            else:
                source_run=state.get('run_id');scene_data=state.get('scene')
                scene=scene_data.get('value') if type(scene_data) is dict else None
                if scene!='playing':reason='nonplaying_scene'
                elif not isinstance(source_run,str) or not source_run:reason='source_run_unknown'
                elif self._source_run is not None and source_run!=self._source_run:reason='source_run_changed'
                if source_run:self._source_run=source_run
                if reason:notification=self._break_locked(reason,state.get('state_id'))
                value=copy.deepcopy(state)
                value['run_id']=f'{source_run}/continuity-{self._epoch}'
                value['continuity']=dict(epoch=self._epoch,source_run_id=source_run,
                    capture_epoch=ticket.item['_continuity_epoch'],scope='Observable capture/scene/session continuity, not hidden restart detection.')
                value=super().publish_observation(ticket,value,**kwargs)
                blocked=(reason is not None or self._source_ended or not ticket.item['_continuity_capture_valid']
                         or ticket.item['_continuity_epoch']!=self._epoch)
                if blocked and value['readonly_runtime']['publication_at_completion']=='published':
                    self._published=None;value['readonly_runtime']['publication_at_completion']='continuity_rejected'
                if value['readonly_runtime']['publication_at_completion']=='published':
                    self._after_publish_locked(value)
        self._notify(notification,reason)
        return value

    def current(self,now):
        notification=None
        with self._condition:
            if self._last_capture is not None and finite_nonnegative(now) and now-self._last_capture>self.max_capture_gap and not self._gap_reported:
                self._gap_reported=True;notification=self._break_locked('capture_stalled')
            result=None if self._source_ended else super().current(now)
        self._notify(notification,'capture_stalled')
        return result

    def session_changed(self,reason='explicit_session_restart'):
        with self._condition:listener=self._break_locked(reason)
        self._notify(listener,reason)

    def fail(self,error):
        with self._condition:
            listener=self._break_locked('capture_or_runtime_failed');super().fail(error)
        self._notify(listener,'capture_or_runtime_failed')

    def close(self):
        with self._condition:
            if self._source_ended:return
            self._source_ended=True;listener=self._break_locked('capture_source_closed');super().close()
        self._notify(listener,'capture_source_closed')

    def continuity_events(self):
        with self._condition:return copy.deepcopy(self._continuity_events)

    @property
    def source_ended(self):
        with self._condition:return self._source_ended
