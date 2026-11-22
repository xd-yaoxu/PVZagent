"""Bounded latest-frame handoff and generation/age checked observation publication."""
from dataclasses import dataclass
import math
import threading


def finite_nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


@dataclass(frozen=True)
class FrameTicket:
    generation:int
    item:object


class LatestFrames:
    """One pending frame plus one in-flight consumer; no FIFO backlog."""
    def __init__(self):
        self._condition=threading.Condition();self._pending=None;self._generation=0
        self._closed=False;self._error=None;self._published=None

    def put(self,item):
        with self._condition:
            if self._closed:raise RuntimeError('Frame source closed')
            replaced=self._pending;self._generation+=1
            self._pending=FrameTicket(self._generation,item)
            # A newer observation, including an invalid one, invalidates the old current state.
            self._published=None;self._condition.notify_all()
            return replaced

    def take(self):
        with self._condition:
            self._condition.wait_for(lambda:self._pending is not None or self._closed)
            ticket=self._pending;self._pending=None;return ticket

    def publish(self,ticket,value,*,captured_at,now,max_age):
        if not finite_nonnegative(captured_at) or not finite_nonnegative(now) or not finite_nonnegative(max_age) or max_age==0:
            raise ValueError('Finite monotonic clocks and positive age budget required')
        with self._condition:
            if self._error is not None:return 'source_failed'
            if ticket.generation!=self._generation:return 'superseded'
            if not 0<=now-captured_at<=max_age:return 'expired'
            self._published=(ticket.generation,value,captured_at,captured_at+max_age)
            return 'published'

    def current(self,now):
        if not finite_nonnegative(now):raise ValueError('Finite monotonic clock required')
        with self._condition:
            if self._published is None or self._error is not None:return None
            generation,value,captured_at,expires=self._published
            return value if generation==self._generation and captured_at<=now<=expires else None

    def close(self):
        with self._condition:self._closed=True;self._condition.notify_all()

    def fail(self,error):
        with self._condition:
            self._error=error;self._closed=True;self._pending=None;self._published=None;self._condition.notify_all()


class SkippingSchedule:
    """Advance past missed capture slots; never replay a backlog of deadlines."""
    def __init__(self,start,interval):
        if not finite_nonnegative(start) or not finite_nonnegative(interval) or interval==0:
            raise ValueError('Finite start and positive interval required')
        self.next_due=start;self.interval=interval

    def completed(self,now):
        if not finite_nonnegative(now) or now<self.next_due:raise ValueError('Completion precedes capture slot')
        self.next_due+=self.interval;skipped=0
        if self.next_due<now:
            skipped=math.floor((now-self.next_due)/self.interval)+1
            self.next_due+=skipped*self.interval
        return skipped
