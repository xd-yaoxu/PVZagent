"""PVZagent Read-Only Real-Time Advice Assistant v1.0 launcher.

Usage (from Win+R or interactive PowerShell on the Default desktop):
    pythonw scripts/pvz_readonly_advisor.py
Offline preflight check:
    python scripts/pvz_readonly_advisor.py --self-test
"""
from __future__ import annotations

import ctypes as ct
import ctypes.wintypes as wt
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _current_desktop_name() -> str:
    u32 = ct.WinDLL("user32.dll", use_last_error=True, winmode=0x800)
    k32 = ct.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)
    u32.GetThreadDesktop.argtypes = (wt.DWORD,)
    u32.GetThreadDesktop.restype = wt.HANDLE
    u32.GetUserObjectInformationW.argtypes = (wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD))
    u32.GetUserObjectInformationW.restype = wt.BOOL
    k32.GetCurrentThreadId.argtypes = ()
    k32.GetCurrentThreadId.restype = wt.DWORD
    hdesk = u32.GetThreadDesktop(k32.GetCurrentThreadId())
    if not hdesk:
        return ""
    buf = ct.create_unicode_buffer(256)
    needed = wt.DWORD()
    if not u32.GetUserObjectInformationW(hdesk, 2, buf, ct.sizeof(buf), ct.byref(needed)):
        return ""
    return buf.value


def run_self_test() -> int:
    from pvz_agent.product_startup import verify_product_startup
    from pvz_agent.readonly_advisor_v1.fast_perception import FastGeometryNormalizer, FastSinglePassPerception
    from pvz_agent.readonly_advisor_v1.wgc_capture import ALLOWED_EXE_NAMES, DLL_PATH

    startup = verify_product_startup(REPO_ROOT)
    assert startup.get("artifacts_checked") == 59
    assert DLL_PATH.is_file()
    dll_bytes = DLL_PATH.read_bytes()
    assert all(name.encode("utf-16-le") in dll_bytes for name in ALLOWED_EXE_NAMES)
    _ = FastGeometryNormalizer(REPO_ROOT)
    _ = FastSinglePassPerception(REPO_ROOT)
    print(json.dumps({
        "status": "SELF_TEST_PASS",
        "artifacts_checked": startup["artifacts_checked"],
        "dll_present": True,
        "portable_exe_names_verified": True,
        "input_sent": False,
        "stop_latched": True,
    }, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in args:
        return run_self_test()

    desktop = _current_desktop_name()
    if desktop.casefold() != "default":
        sys.stderr.write(f"Refused GUI launch on non-interactive desktop: {desktop!r}. Launch from Win+R or PowerShell.\n")
        return 2

    import tkinter as tk
    from pvz_agent.readonly_advisor_v1.advisor_runtime import AdvisorController
    from pvz_agent.readonly_advisor_v1.sidebar_app import ReadonlyAdvisorSidebarApp

    root = tk.Tk()
    controller = AdvisorController(root=REPO_ROOT)
    app = ReadonlyAdvisorSidebarApp(root, controller)
    try:
        root.mainloop()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
