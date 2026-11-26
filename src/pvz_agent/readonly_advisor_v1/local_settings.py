"""Windows DPAPI-encrypted user settings stored strictly outside the project repository.

Stores Base URL, model, protocol, interval, topmost preference, and DPAPI-encrypted
API key under %LOCALAPPDATA%\\PVZagent\\readonly_advisor_v1_settings.json.
Plaintext API keys are NEVER written to disk (neither inside the repo nor in AppData).
"""
from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from typing import Any

from .async_llm_coach import (
    SUPPORTED_PROTOCOLS,
    SUPPORTED_REASONING_EFFORTS,
    SUPPORTED_VISION_MODES,
    validate_base_url,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_DPAPI_ENTROPY = b"PVZagent.ReadonlyAdvisor.v1.ApiKey"
_CRYPTPROTECT_UI_FORBIDDEN = 0x01


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _make_blob(data: bytes) -> tuple[_DATA_BLOB, Any]:
    if not data:
        return _DATA_BLOB(0, None), None
    buf = ctypes.create_string_buffer(data, len(data))
    blob = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    return blob, buf


def _dpapi_dlls():
    crypt32 = ctypes.WinDLL("crypt32.dll", use_last_error=True, winmode=0x800)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DATA_BLOB),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DATA_BLOB),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    return crypt32, kernel32


def dpapi_protect(secret: str) -> str:
    """Encrypt a UTF-8 string using current Windows user DPAPI and return base64 ciphertext."""
    if not secret:
        return ""
    if os.name != "nt":
        raise RuntimeError("dpapi_requires_windows")
    crypt32, kernel32 = _dpapi_dlls()

    raw = secret.encode("utf-8")
    in_blob, _in_buf = _make_blob(raw)
    ent_blob, _ent_buf = _make_blob(_DPAPI_ENTROPY)
    out_blob = _DATA_BLOB()
    ok = crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        "PVZagent ReadonlyAdvisor API Key",
        ctypes.byref(ent_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    try:
        if not ok or not out_blob.pbData or out_blob.cbData == 0:
            raise RuntimeError("dpapi_protect_failed")
        cipher = ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)
    return base64.b64encode(cipher).decode("ascii")


def dpapi_unprotect(cipher_b64: str) -> str:
    """Decrypt a base64 DPAPI blob bound to the current Windows user."""
    if not cipher_b64:
        return ""
    if os.name != "nt":
        raise RuntimeError("dpapi_requires_windows")
    try:
        cipher = base64.b64decode(cipher_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise RuntimeError("dpapi_invalid_base64") from exc
    if not cipher:
        return ""

    crypt32, kernel32 = _dpapi_dlls()

    in_blob, _in_buf = _make_blob(cipher)
    ent_blob, _ent_buf = _make_blob(_DPAPI_ENTROPY)
    out_blob = _DATA_BLOB()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        ctypes.byref(ent_blob),
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    try:
        if not ok or not out_blob.pbData:
            raise RuntimeError("dpapi_unprotect_failed")
        raw = ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        if out_blob.pbData:
            kernel32.LocalFree(out_blob.pbData)
    return raw.decode("utf-8")


def default_user_settings_path() -> Path:
    """Return %LOCALAPPDATA%\\PVZagent\\readonly_advisor_v1_settings.json (outside repo)."""
    local_app = os.environ.get("LOCALAPPDATA")
    base = Path(local_app) if local_app else (Path.home() / "AppData" / "Local")
    return base / "PVZagent" / "readonly_advisor_v1_settings.json"


def _is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


class LocalSettingsStore:
    """Persist user API preferences outside the project directory with DPAPI-encrypted key."""

    def __init__(self, path: Path | None = None):
        target = Path(path) if path is not None else default_user_settings_path()
        if _is_within_repo(target):
            raise ValueError("local_settings_path_must_be_outside_repo")
        self.path = target

    def load(self) -> dict[str, Any]:
        """Load saved settings if present; never raises on corrupt/missing file."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        result: dict[str, Any] = {}
        if isinstance(data.get("remember_enabled"), bool):
            result["remember_enabled"] = data["remember_enabled"]
        if isinstance(data.get("topmost"), bool):
            result["topmost"] = data["topmost"]
        if isinstance(data.get("llm_enabled"), bool):
            result["llm_enabled"] = data["llm_enabled"]
        if isinstance(data.get("base_url"), str) and data["base_url"].strip():
            try:
                result["base_url"] = validate_base_url(data["base_url"])
            except ValueError:
                pass
        if isinstance(data.get("model"), str) and data["model"].strip():
            result["model"] = data["model"].strip()
        if isinstance(data.get("protocol"), str) and data["protocol"] in SUPPORTED_PROTOCOLS:
            result["protocol"] = data["protocol"]
        if isinstance(data.get("vision_mode"), str) and data["vision_mode"] in SUPPORTED_VISION_MODES:
            result["vision_mode"] = data["vision_mode"]
        if isinstance(data.get("reasoning_effort"), str) and data["reasoning_effort"] in SUPPORTED_REASONING_EFFORTS:
            result["reasoning_effort"] = data["reasoning_effort"]
        if isinstance(data.get("min_interval_s"), (int, float)):
            result["min_interval_s"] = max(10.0, min(60.0, float(data["min_interval_s"])))

        enc_key = data.get("api_key_dpapi_b64")
        if isinstance(enc_key, str) and enc_key.strip():
            try:
                decrypted = dpapi_unprotect(enc_key.strip()).strip()
                if decrypted:
                    result["api_key"] = decrypted
            except Exception:
                result["api_key_decrypt_failed"] = True
        return result

    def save(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        protocol: str,
        min_interval_s: float,
        vision_mode: str = "text_only",
        reasoning_effort: str = "low",
        llm_enabled: bool = True,
        topmost: bool = True,
        remember_enabled: bool = True,
    ) -> tuple[bool, str]:
        """Save user preferences atomically; API key is DPAPI-encrypted (never plaintext)."""
        if not remember_enabled:
            self.clear_all()
            return True, "已关闭本机记忆并清除本地配置"

        clean_url = ""
        if (base_url or "").strip():
            try:
                clean_url = validate_base_url(base_url)
            except ValueError as exc:
                return False, str(exc)

        clean_key = (api_key or "").strip()
        enc_key = ""
        if clean_key:
            try:
                enc_key = dpapi_protect(clean_key)
            except Exception:
                return False, "本机 DPAPI 加密失败，为保护密钥已拒绝明文保存"

        clean_proto = protocol if protocol in SUPPORTED_PROTOCOLS else "responses"
        clean_vision = vision_mode if vision_mode in SUPPORTED_VISION_MODES else "text_only"
        clean_effort = reasoning_effort if reasoning_effort in SUPPORTED_REASONING_EFFORTS else "low"
        payload = {
            "schema_version": 1,
            "remember_enabled": True,
            "base_url": clean_url,
            "model": (model or "").strip(),
            "protocol": clean_proto,
            "vision_mode": clean_vision,
            "reasoning_effort": clean_effort,
            "llm_enabled": bool(llm_enabled),
            "min_interval_s": max(10.0, min(60.0, float(min_interval_s))),
            "topmost": bool(topmost),
            "api_key_dpapi_b64": enc_key,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return True, "已加密记忆到本地用户目录（%LOCALAPPDATA%\\PVZagent，不写入项目）"

    def clear_api_key(self) -> None:
        """Remove stored encrypted API key while preserving non-secret preferences if present."""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.clear_all()
            return
        if not isinstance(data, dict):
            self.clear_all()
            return
        if "api_key_dpapi_b64" in data:
            data["api_key_dpapi_b64"] = ""
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def clear_all(self) -> None:
        """Delete the local settings file completely."""
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass
