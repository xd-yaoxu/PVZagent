"""Continuous client-only WGC capture and original PVZ window identity selector.

Runs WGC on a dedicated MTA worker thread so Tk can stay on the STA main thread.
Zero pixels are written to disk, and zero game inputs are ever sent.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.wintypes as wt
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Callable

import numpy as np

ALLOWED_EXE_NAMES = ("PVZ-Windowed.exe", "PlantsVsZombies.exe", "popcapgame1.exe")
EXPECTED_CLASS = "MainWindow"
DLL_PATH = Path(__file__).resolve().parent / "native" / "build" / "pvz_wgc_continuous.dll"


def matches_pvz_exe_path(full_path: str) -> bool:
    """Match an original PVZ executable path (honors optional PVZ_GAME_EXE exact-path pin)."""
    cleaned = (full_path or "").strip()
    if not cleaned:
        return False
    override = os.environ.get("PVZ_GAME_EXE", "").strip()
    if override:
        return cleaned.casefold() == override.casefold()
    base = cleaned.replace("/", "\\").rsplit("\\", 1)[-1].casefold()
    return any(base == name.casefold() for name in ALLOWED_EXE_NAMES)
QUERY_AND_SYNCHRONIZE = 0x101000
WAIT_TIMEOUT = 258
LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x800
SUPPORTED_CLIENT_SIZES = frozenset({(800, 600), (1200, 900)})

RECT_FIELDS = tuple(
    (name, ct.c_int32)
    for name in (
        "client_left", "client_top", "client_right", "client_bottom",
        "dwm_left", "dwm_top", "dwm_right", "dwm_bottom",
    )
)
ALPHA_DIAG_FIELDS = (
    "crop_x", "crop_y", "alpha_0", "alpha_1_250", "alpha_251_254", "alpha_255", "alpha_min",
    "min_x", "min_y", "max_x", "max_y", "corner_tl", "corner_tr", "corner_bl", "corner_br",
    "edge_top", "edge_bottom", "edge_left", "edge_right", "interior", "outer_1px",
    "hit_sun", "hit_cards", "hit_hud_header", "hit_board",
)


class AlphaDiag(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in ALPHA_DIAG_FIELDS]


class WgcRequest(ct.Structure):
    _fields_ = (
        [(n, ct.c_uint32) for n in ("size", "version", "pid", "tid")]
        + [(n, ct.c_uint64) for n in ("hwnd", "creation_time_100ns", "deadline_tick_ms")]
        + list(RECT_FIELDS)
        + [("class_name", ct.c_wchar * 128)]
    )


class WgcResult(ct.Structure):
    _fields_ = (
        [(n, ct.c_uint32) for n in (
            "size", "status", "identity_before", "identity_after",
            "content_w", "content_h", "texture_w", "texture_h",
            "client_w", "client_h", "stride", "frame_index",
        )]
        + [(n, ct.c_uint64) for n in ("frame_qpc_100ns", "bytes_written")]
        + list(RECT_FIELDS)
        + [
            ("hresult", ct.c_uint32),
            ("cleanup_state", ct.c_uint32),
            ("alpha_nonopaque", ct.c_uint32),
            ("open_stage", ct.c_uint32),
            ("apartment_hr", ct.c_uint32),
            ("apartment_type", ct.c_uint32),
            ("apartment_qualifier", ct.c_uint32),
            ("ts_reject_code", ct.c_uint32),
            ("stale_preopen_dropped", ct.c_uint32),
            ("ts_lead_us", ct.c_uint32),
            ("reserved0", ct.c_uint32),
            ("alpha_diag", AlphaDiag),
        ]
    )


def perimeter_alpha_ok(alpha_nonopaque: int, diag: dict[str, int], size: tuple[int, int]) -> bool:
    """Evidence-bounded perimeter alpha gate (identical to V5-V10 / v1r7)."""
    if (
        size not in SUPPORTED_CLIENT_SIZES
        or type(diag) is not dict
        or type(alpha_nonopaque) is not int
        or alpha_nonopaque < 0
    ):
        return False
    if any(k not in diag or type(diag[k]) is not int or diag[k] < 0 for k in ALPHA_DIAG_FIELDS):
        return False
    width, height = size
    band = 8 if width == 800 else 12
    total_pixels = width * height
    zones_sum = (
        diag["corner_tl"] + diag["corner_tr"] + diag["corner_bl"] + diag["corner_br"]
        + diag["edge_top"] + diag["edge_bottom"] + diag["edge_left"] + diag["edge_right"]
        + diag["interior"]
    )
    if alpha_nonopaque == 0:
        return (
            diag["alpha_0"] == 0 and diag["alpha_1_250"] == 0 and diag["alpha_251_254"] == 0
            and diag["alpha_255"] == total_pixels and diag["alpha_min"] == 255
            and zones_sum == 0 and diag["outer_1px"] == 0
            and diag["hit_sun"] == 0 and diag["hit_cards"] == 0
            and diag["hit_hud_header"] == 0 and diag["hit_board"] == 0
            and diag["min_x"] == 65535 and diag["min_y"] == 65535
            and diag["max_x"] == 65535 and diag["max_y"] == 65535
        )
    bins_sum = diag["alpha_0"] + diag["alpha_1_250"] + diag["alpha_251_254"]
    return (
        bins_sum == alpha_nonopaque
        and alpha_nonopaque + diag["alpha_255"] == total_pixels
        and zones_sum == alpha_nonopaque
        and diag["alpha_0"] == 0
        and (1 <= diag["alpha_min"] <= 250 if diag["alpha_1_250"] > 0 else 251 <= diag["alpha_min"] <= 254)
        and 0 <= diag["min_x"] <= diag["max_x"] < width
        and 0 <= diag["min_y"] <= diag["max_y"] < height
        and (diag["max_x"] - diag["min_x"] + 1) * (diag["max_y"] - diag["min_y"] + 1) >= alpha_nonopaque
        and diag["outer_1px"] <= alpha_nonopaque
        and diag["outer_1px"] <= 2 * (width + height) - 4
        and diag["interior"] == 0
        and diag["hit_sun"] == 0 and diag["hit_cards"] == 0
        and diag["hit_hud_header"] == 0 and diag["hit_board"] == 0
        and diag["corner_tl"] == 0 and diag["edge_top"] == 0
        and diag["edge_left"] == 0 and diag["edge_bottom"] == 0
        and diag["corner_tr"] <= band
        and diag["edge_right"] <= height - 2 * band
        and diag["corner_bl"] <= min(64, band * band)
        and diag["corner_br"] <= min(64 + band, band * band)
    )


@dataclass
class WindowLease:
    hwnd: int
    pid: int
    tid: int
    process_handle: int
    birth_100ns: int
    closed: bool = False


class OriginalPvzWindowSelector:
    """Win32 metadata verifier for a single operator-presented original PVZ-Windowed.exe HWND."""

    def __init__(self):
        if sys.platform != "win32" or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("x64 Windows required")
        user = ct.WinDLL("user32.dll", use_last_error=True, winmode=LOAD_LIBRARY_SEARCH_SYSTEM32)
        kernel = ct.WinDLL("kernel32.dll", use_last_error=True, winmode=LOAD_LIBRARY_SEARCH_SYSTEM32)

        def bind(lib, name, restype, *argtypes):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            return fn

        self._fg = bind(user, "GetForegroundWindow", wt.HWND)
        self._is_win = bind(user, "IsWindow", wt.BOOL, wt.HWND)
        self._class = bind(user, "GetClassNameW", ct.c_int, wt.HWND, wt.LPWSTR, ct.c_int)
        self._visible = bind(user, "IsWindowVisible", wt.BOOL, wt.HWND)
        self._iconic = bind(user, "IsIconic", wt.BOOL, wt.HWND)
        self._ancestor = bind(user, "GetAncestor", wt.HWND, wt.HWND, wt.UINT)
        self._owner = bind(user, "GetWindow", wt.HWND, wt.HWND, wt.UINT)
        self._style = bind(user, "GetWindowLongPtrW", ct.c_ssize_t, wt.HWND, ct.c_int)
        self._tp = bind(user, "GetWindowThreadProcessId", wt.DWORD, wt.HWND, ct.POINTER(wt.DWORD))
        self._open = bind(kernel, "OpenProcess", wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
        self._pid = bind(kernel, "GetProcessId", wt.DWORD, wt.HANDLE)
        self._wait = bind(kernel, "WaitForSingleObject", wt.DWORD, wt.HANDLE, wt.DWORD)
        self._times = bind(kernel, "GetProcessTimes", wt.BOOL, wt.HANDLE, *([ct.POINTER(wt.FILETIME)] * 4))
        self._path = bind(kernel, "QueryFullProcessImageNameW", wt.BOOL, wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD))
        self._compare = bind(kernel, "CompareStringOrdinal", ct.c_int, wt.LPCWSTR, ct.c_int, wt.LPCWSTR, ct.c_int, wt.BOOL)
        self._close = bind(kernel, "CloseHandle", wt.BOOL, wt.HANDLE)

    def _inspect_window(self, hwnd: int) -> tuple[str, tuple[int, int] | None]:
        if not hwnd or not self._is_win(hwnd):
            return "window_closed", None
        buf = ct.create_unicode_buffer(128)
        n = self._class(hwnd, buf, len(buf))
        if n <= 0 or n >= len(buf) - 1:
            return "metadata_unavailable", None
        if buf.value != EXPECTED_CLASS:
            return "outside_scope", None
        if not self._visible(hwnd) or self._iconic(hwnd):
            return "minimized_or_hidden", None
        if self._ancestor(hwnd, 2) != hwnd:
            return "outside_scope", None
        ct.set_last_error(0)
        if self._owner(hwnd, 4):
            return "outside_scope", None
        style = self._style(hwnd, -16)
        if style & 0x40000000:
            return "outside_scope", None
        pid = wt.DWORD()
        tid = self._tp(hwnd, ct.byref(pid))
        if not tid or not pid.value:
            return "metadata_unavailable", None
        return "ok", (int(pid.value), int(tid))

    def _inspect_process(self, handle: int, expected_pid: int) -> tuple[str, int | None]:
        if self._pid(handle) != expected_pid:
            return "process_changed", None
        if self._wait(handle, 0) != WAIT_TIMEOUT:
            return "process_exited", None
        buf = ct.create_unicode_buffer(32768)
        length = wt.DWORD(len(buf))
        if not self._path(handle, 0, buf, ct.byref(length)) or not 0 < length.value < len(buf):
            return "identity_unknown", None
        if not matches_pvz_exe_path(buf.value[: length.value]):
            return "outside_scope", None
        times = [wt.FILETIME() for _ in range(4)]
        if not self._times(handle, *(ct.byref(t) for t in times)):
            return "identity_unknown", None
        born = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        return ("ok", born) if born else ("identity_unknown", None)

    def inspect_foreground(self) -> tuple[str, WindowLease | None]:
        """Inspect the current foreground window and return a verified WindowLease if it is original PVZ."""
        hwnd = int(self._fg() or 0)
        if not hwnd:
            return "no_foreground", None
        st, ids = self._inspect_window(hwnd)
        if st != "ok" or ids is None:
            return st, None
        pid, tid = ids
        handle = self._open(QUERY_AND_SYNCHRONIZE, False, pid)
        if not handle:
            return "identity_unknown", None
        st_p, born = self._inspect_process(handle, pid)
        if st_p != "ok" or born is None:
            self._close(handle)
            return st_p, None
        st_w2, ids2 = self._inspect_window(hwnd)
        if st_w2 != "ok" or ids2 != (pid, tid):
            self._close(handle)
            return "window_changed", None
        return "ok", WindowLease(hwnd=hwnd, pid=pid, tid=tid, process_handle=int(handle), birth_100ns=born)

    def is_lease_alive(self, lease: WindowLease) -> tuple[bool, bool, str]:
        """Return (window_alive, is_foreground, status_code) without mutating lease."""
        if lease is None or lease.closed:
            return False, False, "lease_closed"
        st_p, born = self._inspect_process(lease.process_handle, lease.pid)
        if st_p != "ok" or born != lease.birth_100ns:
            return False, False, st_p
        st_w, ids = self._inspect_window(lease.hwnd)
        if st_w == "minimized_or_hidden":
            return True, False, "minimized_or_hidden"
        if st_w != "ok" or ids != (lease.pid, lease.tid):
            return False, False, st_w
        is_fg = int(self._fg() or 0) == lease.hwnd
        return True, is_fg, "ok" if is_fg else "not_foreground"

    def release(self, lease: WindowLease | None) -> bool:
        if lease is None or lease.closed:
            return True
        lease.closed = True
        try:
            return bool(self._close(lease.process_handle))
        except Exception:
            return False


@dataclass(frozen=True)
class CapturedFrame:
    bgra: np.ndarray  # shape (h, w, 4), uint8, copied out of staging buffer
    client_size: tuple[int, int]
    rect_signature: tuple[int, ...]
    frame_index: int
    frame_qpc_100ns: int
    captured_monotonic: float
    stale_dropped: int
    alpha_nonopaque: int


class _NativeWgcSession:
    """Lives exclusively on the dedicated MTA worker thread."""

    def __init__(self, lease: WindowLease, dll_path: Path = DLL_PATH):
        self.lease = lease
        self.dll_path = dll_path
        self.ctx = ct.c_void_p()
        self.library = None
        self.buffer: np.ndarray | None = None
        self.size: tuple[int, int] | None = None
        self.expected_rect: tuple[int, ...] | None = None
        self.last_qpc = 0
        self.accepted_frames = 0
        self.dpi_old = None
        self.dpi_user = None

    def open(self) -> tuple[tuple[int, int], tuple[int, ...]]:
        user = ct.WinDLL("user32.dll", winmode=0x800, use_last_error=True)
        user.SetThreadDpiAwarenessContext.argtypes = [ct.c_void_p]
        user.SetThreadDpiAwarenessContext.restype = ct.c_void_p
        old = user.SetThreadDpiAwarenessContext(ct.c_void_p(-4))
        if not old:
            raise RuntimeError("dpi_context_failed")
        self.dpi_old = old
        self.dpi_user = user

        dwm = ct.WinDLL("dwmapi.dll", winmode=0x800, use_last_error=True)
        user.GetClientRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
        user.GetClientRect.restype = wt.BOOL
        user.ClientToScreen.argtypes = [wt.HWND, ct.POINTER(wt.POINT)]
        user.ClientToScreen.restype = wt.BOOL
        dwm.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ct.c_void_p, wt.DWORD]
        dwm.DwmGetWindowAttribute.restype = ct.c_int32

        rect = wt.RECT()
        pt_a = wt.POINT()
        pt_b = wt.POINT()
        dwm_rect = wt.RECT()
        if not user.GetClientRect(self.lease.hwnd, ct.byref(rect)):
            raise RuntimeError("client_rect_failed")
        pt_b.x, pt_b.y = rect.right, rect.bottom
        if (
            not user.ClientToScreen(self.lease.hwnd, ct.byref(pt_a))
            or not user.ClientToScreen(self.lease.hwnd, ct.byref(pt_b))
            or dwm.DwmGetWindowAttribute(self.lease.hwnd, 9, ct.byref(dwm_rect), ct.sizeof(dwm_rect)) != 0
        ):
            raise RuntimeError("window_geometry_failed")
        size = (pt_b.x - pt_a.x, pt_b.y - pt_a.y)
        if size not in SUPPORTED_CLIENT_SIZES:
            raise RuntimeError(f"unsupported_client_size:{size[0]}x{size[1]}")
        self.size = size

        req = WgcRequest()
        req.size = ct.sizeof(req)
        req.version = 1
        req.pid = self.lease.pid
        req.tid = self.lease.tid
        req.hwnd = self.lease.hwnd
        req.creation_time_100ns = self.lease.birth_100ns
        req.deadline_tick_ms = 0  # open-ended operator-controlled continuous read-only session
        rect_vals = (
            pt_a.x, pt_a.y, pt_b.x, pt_b.y,
            dwm_rect.left, dwm_rect.top, dwm_rect.right, dwm_rect.bottom,
        )
        for (key, _), val in zip(RECT_FIELDS, rect_vals):
            setattr(req, key, val)
        req.class_name = EXPECTED_CLASS
        self.expected_rect = rect_vals

        self.library = ct.WinDLL(str(self.dll_path), winmode=0x900, use_last_error=True)
        self.library.pvz_wgc_open.argtypes = [ct.POINTER(WgcRequest), ct.POINTER(ct.c_void_p), ct.POINTER(WgcResult)]
        self.library.pvz_wgc_open.restype = ct.c_int32
        self.library.pvz_wgc_read.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.POINTER(WgcResult)]
        self.library.pvz_wgc_read.restype = ct.c_int32
        self.library.pvz_wgc_close.argtypes = [ct.c_void_p]
        self.library.pvz_wgc_close.restype = ct.c_int32

        outcome = WgcResult()
        outcome.size = ct.sizeof(outcome)
        status = int(self.library.pvz_wgc_open(ct.byref(req), ct.byref(self.ctx), ct.byref(outcome)))
        if status != 0 or not self.ctx.value:
            raise RuntimeError(f"wgc_open_failed:status={status},stage={int(outcome.open_stage)}")
        self.buffer = np.zeros(size[0] * size[1] * 4, dtype=np.uint8)
        return size, rect_vals

    def read_frame(self) -> CapturedFrame:
        assert self.buffer is not None and self.size is not None and self.expected_rect is not None
        self.buffer.fill(0)
        res = WgcResult()
        res.size = ct.sizeof(res)
        code = int(
            self.library.pvz_wgc_read(
                self.ctx, ct.c_void_p(self.buffer.ctypes.data), self.buffer.nbytes, ct.byref(res)
            )
        )
        if code != 0 or int(res.status) != 0:
            self.buffer.fill(0)
            raise RuntimeError(f"wgc_read_rejected:code={code},status={int(res.status)}")
        width, height = self.size
        alpha_diag = {k: int(getattr(res.alpha_diag, k)) for k in ALPHA_DIAG_FIELDS}
        rect_now = tuple(int(getattr(res, k)) for k, _ in RECT_FIELDS)
        if (
            int(res.size) != ct.sizeof(res)
            or int(res.identity_before) != 1
            or int(res.identity_after) != 1
            or (int(res.client_w), int(res.client_h)) != self.size
            or int(res.stride) != width * 4
            or int(res.bytes_written) != width * height * 4
            or not perimeter_alpha_ok(int(res.alpha_nonopaque), alpha_diag, self.size)
            or int(res.frame_qpc_100ns) <= self.last_qpc
            or rect_now != self.expected_rect
        ):
            self.buffer.fill(0)
            raise RuntimeError("wgc_frame_gate_rejected")
        self.accepted_frames += 1
        self.last_qpc = int(res.frame_qpc_100ns)
        frame_copy = self.buffer.reshape(height, width, 4).copy()
        self.buffer.fill(0)
        return CapturedFrame(
            bgra=frame_copy,
            client_size=self.size,
            rect_signature=rect_now,
            frame_index=self.accepted_frames,
            frame_qpc_100ns=self.last_qpc,
            captured_monotonic=time.monotonic(),
            stale_dropped=int(res.stale_preopen_dropped),
            alpha_nonopaque=int(res.alpha_nonopaque),
        )

    def close(self) -> bool:
        ok = True
        if self.buffer is not None:
            self.buffer.fill(0)
            self.buffer = None
        if self.ctx.value and self.library is not None:
            try:
                ok = int(self.library.pvz_wgc_close(self.ctx)) == 0
            except Exception:
                ok = False
            self.ctx = ct.c_void_p()
        if self.dpi_old is not None and self.dpi_user is not None:
            try:
                ok = bool(self.dpi_user.SetThreadDpiAwarenessContext(self.dpi_old)) and ok
            except Exception:
                ok = False
            self.dpi_old = None
        return ok


class ContinuousWgcSource:
    """Thread-safe controller owning a dedicated MTA thread for pvz_wgc_continuous.dll."""

    def __init__(self, *, session_factory: Callable[[WindowLease], _NativeWgcSession] | None = None):
        self._factory = session_factory or (lambda lease: _NativeWgcSession(lease))
        self._cmds: queue.Queue = queue.Queue(maxsize=1)
        self._resps: queue.Queue = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None
        self._open = False
        self._lock = threading.Lock()

    def _serve(self, lease: WindowLease):
        session = None
        try:
            session = self._factory(lease)
            info = session.open()
            self._resps.put((True, info))
            while True:
                cmd = self._cmds.get()
                if cmd == "read":
                    try:
                        frame = session.read_frame()
                        self._resps.put((True, frame))
                    except BaseException as exc:
                        self._resps.put((False, exc))
                elif cmd == "close":
                    ok = session.close()
                    self._resps.put((True, ok))
                    return
        except BaseException as exc:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            self._resps.put((False, exc))

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open and self._worker is not None and self._worker.is_alive()

    def open(self, lease: WindowLease, timeout: float = 4.0) -> tuple[tuple[int, int], tuple[int, ...]]:
        self.close()
        with self._lock:
            self._cmds = queue.Queue(maxsize=1)
            self._resps = queue.Queue(maxsize=1)
            self._worker = threading.Thread(
                target=self._serve, args=(lease,), name="pvz-wgc-continuous-mta", daemon=True
            )
            self._worker.start()
            ok, value = self._resps.get(timeout=timeout)
            if not ok:
                self._worker.join(timeout=1.0)
                self._worker = None
                self._open = False
                raise value
            self._open = True
            return value

    def read_frame(self, timeout: float = 3.5) -> CapturedFrame:
        with self._lock:
            if not self._open or self._worker is None or not self._worker.is_alive():
                raise RuntimeError("wgc_not_open")
            self._cmds.put("read", timeout=1.0)
            ok, value = self._resps.get(timeout=timeout)
            if not ok:
                raise value
            return value

    def close(self) -> bool:
        with self._lock:
            worker = self._worker
            was_open = self._open
            self._open = False
            self._worker = None
            if worker is None:
                return True
            ok = True
            if was_open and worker.is_alive():
                try:
                    self._cmds.put("close", timeout=0.5)
                    r_ok, r_val = self._resps.get(timeout=2.0)
                    ok = bool(r_ok and r_val)
                except Exception:
                    ok = False
            worker.join(timeout=1.5)
            return ok and not worker.is_alive()
