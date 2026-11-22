"""Capture a deliberately pinned, fully discovered game/scaler window pair.

Each inspect verifies HWND existence, owning executable, class, visibility and
fresh physical geometry. A lost pin fails the run; it never silently retargets.
"""
import ctypes
from ctypes import wintypes as w
from pathlib import Path

from .profiled_capture import ProfiledWindowsCapture
from .windows_capture import MonitorInfo,_rect


def validate_pinned_window(info,expected_exe,kind):
    if str(Path(info['process']).resolve()).casefold()!=expected_exe:
        raise RuntimeError('Pinned executable identity changed')
    expected_class=info['class']=='MainWindow' if kind=='game' else info['class'].startswith('Window_Magpie_')
    if not expected_class or info['visible'] is not True or info['minimized'] is not False:
        raise RuntimeError('Pinned window class/visibility invalid')
    return info


class PinnedWindowsCapture(ProfiledWindowsCapture):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._pinned=None;self._pinned_pids={}
        self.u.IsWindow.argtypes=[w.HWND];self.u.IsWindow.restype=w.BOOL

    def _owner_pid(self,hwnd):
        pid=w.DWORD();self.u.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        if not pid.value:raise RuntimeError('Pinned window owner unavailable')
        return pid.value

    def _read_pinned(self,hwnd,expected_exe,kind):
        if not self.u.IsWindow(hwnd):raise RuntimeError('Pinned HWND no longer exists')
        pid=self._owner_pid(hwnd)
        if pid!=self._pinned_pids[hwnd]:raise RuntimeError('Pinned window owner changed')
        path=self._process_path(pid)
        return validate_pinned_window(self._window_info(hwnd,path),expected_exe,kind)

    def inspect(self):
        if self._pinned is None:
            result=super().inspect()
            self._pinned=dict(game=result['game']['hwnd'],scaler=result['scaler']['hwnd'] if result['scaler'] else None)
            self._pinned_pids={hwnd:self._owner_pid(hwnd) for hwnd in self._pinned.values() if hwnd is not None}
            return result
        with self.physical_coordinates():
            game=self._read_pinned(self._pinned['game'],self.game_exe,'game')
            scaler=self._read_pinned(self._pinned['scaler'],self.scaler_exe,'scaler') if self._pinned['scaler'] else None
            display=scaler or game;monitor=MonitorInfo();monitor.cbSize=ctypes.sizeof(monitor)
            handle=self.u.MonitorFromWindow(display['hwnd'],2)
            if not self.u.GetMonitorInfoW(handle,ctypes.byref(monitor)):
                raise ctypes.WinError(ctypes.get_last_error())
            return dict(game=game,scaler=scaler,
                display_monitor=dict(screen=_rect(monitor.rcMonitor),work_area=_rect(monitor.rcWork)))
