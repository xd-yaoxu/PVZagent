"""Conservative short-term association; misses never imply a zombie died."""
import copy
import math


class ShortTermTracker:
    def __init__(self,config):
        self.config=config;self.run_id=None;self.tracks={};self.counter=0
        self.last_time=None;self.seen_states=set()

    def update(self,*,run_id,state_id,frame_time,candidates,scene):
        if not isinstance(run_id,str) or not run_id.strip() or not isinstance(state_id,str) or not state_id.strip():
            raise ValueError('Nonempty run_id and state_id required')
        if type(frame_time) not in (float,int) or not math.isfinite(frame_time) or frame_time<0:
            raise ValueError('Finite nonnegative frame time required')
        # Validate before mutation so an invalid update cannot damage the active state.
        for c in candidates:
            p=c.get('head_xy',[])
            if type(c.get('row')) is not int or not 1<=c['row']<=6 or len(p)!=2 or not all(type(v) in (float,int) and math.isfinite(v) for v in p) or not (0<=p[0]<800 and 0<=p[1]<600):
                raise ValueError('Invalid detection geometry')
        changed=run_id!=self.run_id
        if not changed and (state_id in self.seen_states or self.last_time is not None and frame_time<=self.last_time):
            raise ValueError('Duplicate or out-of-order frame; do not replay')
        events=[]
        if changed:
            if self.run_id is not None:
                events.append({'type':'run_reset','previous_run_id':self.run_id})
            self.run_id=run_id;self.tracks={};self.seen_states=set();self.last_time=None
        self.seen_states.add(state_id);self.last_time=frame_time
        def event(kind,track_id=None,**fields):
            events.append({'type':kind,'track_id':track_id,**fields})
        for tid,t in list(self.tracks.items()):
            if scene!='playing' or frame_time-t['last_seen_time']>self.config['max_gap_seconds']:
                event('track_ended',tid,reason='scene_unverified' if scene!='playing' else 'observation_gap',death_confirmed=False)
                del self.tracks[tid]
        observations=copy.deepcopy(candidates)
        if scene!='playing':
            observations=[]
        distances={}
        for tid,t in self.tracks.items():
            dt=frame_time-t['last_seen_time']
            velocity=max(-self.config['max_speed_pixels_per_second'],
                         min(self.config['max_speed_pixels_per_second'],t['velocity_x'] or 0))
            px=t['head_xy'][0]+velocity*dt
            for i,c in enumerate(observations):
                if c['row']!=t['row']:
                    continue
                dx=c['head_xy'][0]-t['head_xy'][0]
                if abs(dx)>self.config['max_speed_pixels_per_second']*dt+self.config['position_slack_pixels']:
                    continue
                dist=math.hypot(c['head_xy'][0]-px,(c['head_xy'][1]-t['head_xy'][1])*.5)
                distances[tid,i]=dist
        margin=self.config['ambiguity_margin_pixels'];matched={};used=set()
        for tid in self.tracks:
            ranked=sorted((d,i) for (t,i),d in distances.items() if t==tid)
            if not ranked or len(ranked)>1 and ranked[1][0]-ranked[0][0]<margin:
                continue
            d,i=ranked[0]
            reverse=sorted((v,t) for (t,j),v in distances.items() if j==i)
            if reverse[0][1]!=tid or len(reverse)>1 and reverse[1][0]-reverse[0][0]<margin:
                continue
            matched[tid]=i;used.add(i)
        for tid,t in self.tracks.items():
            if tid not in matched:
                if t['status']!='lost':
                    event('track_lost',tid,reason='not_observed_or_ambiguous',death_confirmed=False)
                t['status']='lost'
        for i,c in enumerate(observations):
            tid=next((t for t,j in matched.items() if j==i),None)
            if tid is None and any(j==i for _,j in distances):
                c.update(track_id=None,track_status='unassociated',association_reason='ambiguous_or_nonmutual_match')
                event('association_uncertain',detection_id=c.get('detection_id'))
                continue
            if tid is None:
                self.counter+=1;tid=f'{run_id}:z{self.counter}'
                self.tracks[tid]={'track_id':tid,'row':c['row'],'head_xy':list(c['head_xy']),
                    'last_seen_time':frame_time,'last_seen_state_id':state_id,'velocity_x':None,
                    'hits':0,'status':'tentative','history':[]}
                event('track_created',tid)
            t=self.tracks[tid];old_status=t['status'];dt=frame_time-t['last_seen_time']
            if dt>0 and t['hits']:
                t['velocity_x']=(c['head_xy'][0]-t['head_xy'][0])/dt
            t.update(head_xy=list(c['head_xy']),last_seen_time=frame_time,last_seen_state_id=state_id,hits=t['hits']+1)
            t['status']='confirmed' if t['hits']>=self.config['confirm_hits'] else 'tentative'
            t['history']=(t['history']+[{'state_id':state_id,'time':frame_time,'head_xy':list(c['head_xy'])}])[-8:]
            if old_status=='lost':
                event('track_reacquired',tid)
            c.update(track_id=tid,track_status=t['status'],association_reason='observed_geometry')
        tracks=[]
        for t in self.tracks.values():
            item=copy.deepcopy(t)
            item.update(observed_this_frame=t['last_seen_state_id']==state_id,
                age_since_seen_seconds=round(frame_time-t['last_seen_time'],4),
                position_is_current_observation=t['last_seen_state_id']==state_id,
                velocity_semantics='apparent_head_x_change_not_measured_game_speed',
                confidence_semantics='association_status_not_identity_probability')
            tracks.append(item)
        for e in events:
            e.update(run_id=run_id,state_id=state_id,frame_time_seconds=frame_time)
        return {'candidates':observations,'tracks':tracks,'events':events}
