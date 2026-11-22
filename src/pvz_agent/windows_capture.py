"""Read-only window capture and geometry. No mouse, keyboard or game-memory access."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import ctypes
from ctypes import wintypes as w
import hashlib
import json
import time
import uuid

from PIL import Image, ImageChops, ImageGrab, ImageStat


class MonitorInfo(ctypes.Structure):
    _fields_ = [('cbSize', w.DWORD), ('rcMonitor', w.RECT), ('rcWork', w.RECT), ('dwFlags', w.DWORD)]


def _rect(r):
    return dict(x=r.left, y=r.top, width=r.right - r.left, height=r.bottom - r.top)


class WindowEnumerationError(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__('window_enumeration_failed')
        self.diagnostic = diagnostic


def enumerate_windows_checked(user32, visitor, callback_stage=lambda: 'visitor'):
    """Distinguish native failure from callback failure without exposing window data."""
    count, failed = 0, False
    failure = {'stage': 'callback', 'function': 'unknown', 'winerror': None,
               'exception_type': 'OtherException', 'callback_count': 0}
    @ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    def callback(hwnd, _):
        nonlocal count, failed
        count += 1
        try:
            visitor(hwnd)
            return True
        except BaseException as exc:
            failed = True
            # Lock failure before touching exception attributes or diagnostic callbacks.
            failure['callback_count'] = count
            try:
                stage = callback_stage()
                if type(stage) is str and stage in ('visitor', 'GetWindowThreadProcessId',
                                                     '_process_path', 'path_match', '_window_info', 'IsWindow'):
                    failure['function'] = stage
                kind = type(exc)
                known = (Exception, RuntimeError, ValueError, OSError, KeyboardInterrupt, SystemExit)
                if any(kind is allowed for allowed in known):
                    failure['exception_type'] = kind.__name__
                code = getattr(exc, 'winerror', None)
                if type(code) is int and 0 < code < 2**32:
                    failure['winerror'] = code
            except BaseException:
                pass
            return False
    ctypes.set_last_error(0)
    ok = user32.EnumWindows(callback, 0)
    error = ctypes.get_last_error()
    if failed:
        raise WindowEnumerationError(failure)
    if not ok:
        raise WindowEnumerationError({'stage': 'enum_windows', 'function': 'EnumWindows',
                                      'winerror': error or None, 'exception_type': None,
                                      'callback_count': count})


class WindowsCapture:
    def __init__(self, game_exe, scaler_exe, scaler_config):
        self.game_exe = str(Path(game_exe).resolve()).casefold()
        self.scaler_exe = str(Path(scaler_exe).resolve()).casefold()
        self.scaler_config = Path(scaler_config)
        self.u = ctypes.WinDLL('user32', use_last_error=True)
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        self.dwm = ctypes.WinDLL('dwmapi', use_last_error=True)
        u, k = self.u, self.k
        for name in ('GetWindowRect', 'GetClientRect'):
            getattr(u, name).argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
        u.ClientToScreen.argtypes = [w.HWND, ctypes.POINTER(w.POINT)]
        u.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
        u.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
        u.GetClassNameW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
        u.IsWindowVisible.argtypes = [w.HWND]
        u.IsIconic.argtypes = [w.HWND]
        u.GetDpiForWindow.argtypes = [w.HWND]
        u.GetDpiForWindow.restype = w.UINT
        u.SetThreadDpiAwarenessContext.argtypes = [w.HANDLE]
        u.SetThreadDpiAwarenessContext.restype = w.HANDLE
        u.EnumWindows.argtypes = [ctypes.c_void_p, w.LPARAM]
        u.EnumWindows.restype = w.BOOL
        u.MonitorFromWindow.argtypes = [w.HWND, w.DWORD]
        u.MonitorFromWindow.restype = w.HANDLE
        u.GetMonitorInfoW.argtypes = [w.HANDLE, ctypes.POINTER(MonitorInfo)]
        k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k.OpenProcess.restype = w.HANDLE
        k.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]
        k.CloseHandle.argtypes = [w.HANDLE]
        self.dwm.DwmGetWindowAttribute.argtypes = [w.HWND, w.DWORD, ctypes.c_void_p, w.DWORD]

    @contextmanager
    def physical_coordinates(self):
        # Thread-local awareness; this changes no Windows display setting.
        previous = self.u.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        if not previous:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            self.u.SetThreadDpiAwarenessContext(previous)

    def _process_path(self, pid):
        process = self.k.OpenProcess(0x1000, False, pid)
        if not process:
            return ''
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            length = w.DWORD(len(buffer))
            return buffer.value if self.k.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(length)) else ''
        finally:
            self.k.CloseHandle(process)

    def _window_info(self, hwnd, process_path):
        title, cls = ctypes.create_unicode_buffer(512), ctypes.create_unicode_buffer(256)
        self.u.GetWindowTextW(hwnd, title, len(title))
        self.u.GetClassNameW(hwnd, cls, len(cls))
        outer, client, frame = w.RECT(), w.RECT(), w.RECT()
        origin = w.POINT(0, 0)
        for fn, target in [(self.u.GetWindowRect, outer), (self.u.GetClientRect, client)]:
            if not fn(hwnd, ctypes.byref(target)):
                raise ctypes.WinError(ctypes.get_last_error())
        if not self.u.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise ctypes.WinError(ctypes.get_last_error())
        frame_available = self.dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(frame), ctypes.sizeof(frame)) == 0
        return {'hwnd': int(hwnd), 'process': process_path, 'title': title.value, 'class': cls.value,
                'visible': bool(self.u.IsWindowVisible(hwnd)), 'minimized': bool(self.u.IsIconic(hwnd)),
                'window_screen': _rect(outer), 'extended_frame_screen': _rect(frame) if frame_available else None,
                'client_screen': dict(x=origin.x, y=origin.y, width=client.right, height=client.bottom),
                'dpi': self.u.GetDpiForWindow(hwnd)}

    def inspect(self):
        records, cache = [], {}
        @ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        def visit(hwnd, _):
            pid = w.DWORD()
            self.u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value not in cache:
                cache[pid.value] = self._process_path(pid.value)
            process = cache[pid.value]
            if str(Path(process).resolve()).casefold() in (self.game_exe, self.scaler_exe):
                records.append(self._window_info(hwnd, process))
            return True
        with self.physical_coordinates():
            if not self.u.EnumWindows(visit, 0):
                raise ctypes.WinError(ctypes.get_last_error())
        games = [r for r in records if str(Path(r['process']).resolve()).casefold() == self.game_exe and r['class'] == 'MainWindow' and r['visible']]
        if len(games) != 1 or games[0]['minimized']:
            raise RuntimeError('Expected one visible, non-minimized PVZ MainWindow')
        scalers = [r for r in records if str(Path(r['process']).resolve()).casefold() == self.scaler_exe and r['class'].startswith('Window_Magpie_') and r['visible'] and not r['minimized']]
        if len(scalers) > 1:
            raise RuntimeError('Multiple Magpie outputs; cannot identify game display safely')
        display = scalers[0] if scalers else games[0]
        monitor = MonitorInfo()
        monitor.cbSize = ctypes.sizeof(monitor)
        with self.physical_coordinates():
            handle = self.u.MonitorFromWindow(display['hwnd'], 2)
            if not self.u.GetMonitorInfoW(handle, ctypes.byref(monitor)):
                raise ctypes.WinError(ctypes.get_last_error())
        return {'game': games[0], 'scaler': scalers[0] if scalers else None,
                'display_monitor': {'screen': _rect(monitor.rcMonitor), 'work_area': _rect(monitor.rcWork)}}

    def _mapping(self, windows):
        game, scaler = windows['game'], windows['scaler']
        if not scaler:
            return {'mode': 'native_client', 'game_viewport_screen': game['client_screen'], 'validated': False}
        settings = json.loads(self.scaler_config.read_text(encoding='utf-8-sig'))
        profiles = [p for p in settings.get('profiles', []) if str(Path(p.get('pathRule', '')).resolve()).casefold() == self.game_exe]
        if len(profiles) != 1:
            raise RuntimeError('Need one exact Magpie game profile')
        profile = profiles[0]
        if profile.get('croppingEnabled') or profile.get('3DGameMode'):
            raise RuntimeError('Magpie cropping/3D mode requires separate calibration')
        source = game['extended_frame_screen'] if profile.get('captureTitleBar') else game['client_screen']
        if not source:
            raise RuntimeError('Extended frame bounds unavailable for title-bar scaling')
        target, client = scaler['client_screen'], game['client_screen']
        sx, sy = target['width'] / source['width'], target['height'] / source['height']
        if abs(sx / sy - 1) > .01:
            raise RuntimeError('Magpie output aspect differs; mapping requires recalibration')
        viewport = dict(x=target['x'] + (client['x'] - source['x']) * sx,
                        y=target['y'] + (client['y'] - source['y']) * sy,
                        width=client['width'] * sx, height=client['height'] * sy)
        return {'mode': 'magpie_windowed', 'capture_title_bar': bool(profile.get('captureTitleBar')),
                'source_frame_screen': source, 'output_client_screen': target,
                'game_viewport_screen': viewport, 'validated': False}

    def capture(self, output_dir, label='frame', expected_size=(800, 600)):
        if not label.replace('-', '').isalnum():
            raise ValueError('Invalid capture label')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        tick = time.monotonic_ns()
        before = self.inspect()
        # A real HWND discovered and process-verified above, not a tool's opaque ID.
        image = ImageGrab.grab(window=before['game']['hwnd']).convert('RGB')
        if image.size != tuple(expected_size):
            raise RuntimeError(f'Unexpected game capture {image.size}; recalibration required')
        if max(ImageStat.Stat(image).var) < 10:
            raise RuntimeError('Black/blank game capture; do not use for planning')
        mapping = self._mapping(before)
        target = mapping['game_viewport_screen']
        bbox = (round(target['x']), round(target['y']), round(target['x'] + target['width']), round(target['y'] + target['height']))
        with self.physical_coordinates():
            display = ImageGrab.grab(bbox=bbox, all_screens=True).convert('RGB')
        after = self.inspect()
        stable = before == after
        # Independently align the visible display with the native source. This is
        # a coarse guard for crop errors/occlusion, not a semantic scene detector.
        normalized = display.resize(image.size, Image.Resampling.LANCZOS)
        difference = ImageChops.difference(image, normalized)
        error = sum(ImageStat.Stat(difference).mean) / 3
        mapping['alignment_mean_absolute_error'] = error
        mapping['validated'] = bool(stable and error <= 18)
        mapping['validation_threshold'] = 18
        stamp = datetime.now(timezone.utc).isoformat()
        state_id = f"{label}-{stamp.replace(':', '-').replace('.', '-')}-{uuid.uuid4().hex[:6]}"
        native_file = output_dir / f'{state_id}.png'
        display_file = output_dir / f'{state_id}-display.png'
        image.save(native_file)
        display.save(display_file)
        metadata = {'schema_version': 1, 'state_id': state_id, 'capture_started_at': started,
                    'capture_time': stamp, 'monotonic_ns': tick, 'image_pixels': list(image.size),
                    'image_file': native_file.name, 'display_file': display_file.name,
                    'source': 'Pillow_single_window_image_capture', 'windows': before,
                    'geometry_stable_during_capture': stable, 'display_mapping': mapping,
                    'image_sha256': hashlib.sha256(native_file.read_bytes()).hexdigest(),
                    'capture_duration_ms': (time.monotonic_ns() - tick) / 1e6,
                    'limitations': ['Alignment check is not full occlusion or scene detection.',
                                    'No input sent; targets must be freshly revalidated before future execution.']}
        metadata_file = output_dir / f'{state_id}.json'
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return native_file, metadata_file, metadata
