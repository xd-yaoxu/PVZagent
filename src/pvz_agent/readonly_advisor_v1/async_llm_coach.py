"""Opt-in non-blocking async LLM tactical coach (default OFF per user preference).

Supports user-supplied base_url, api_key (kept strictly in process memory, never written
to repo files or logs), model ID, and protocol (responses / chat_completions).
Falls back to local environment variable PVZ_LLM_API_KEY when the key box is left blank.
Every returned hypothesis is rechecked against the current <=2.0s frame before display.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import http.client
import io
import json
import os
from pathlib import Path
import ssl
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

from ..goal_review import ReviewBlocked, defense_evidence
from ..llm_advice import strict_json
from ..llm_provider import ProviderError, extract_text, load_local_api_key
from ..resource_review import resource_evidence
from ..strategy_review import STRATEGY_SCHEMA, fresh, observed_basis, validate_strategy

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TERRA_CONFIG = REPO_ROOT / "config/product-strategy-terra-v1.json"
FALLBACK_SOL_CONFIG = REPO_ROOT / "config/llm-strategy-v1.json"
DEFAULT_POLICY = {"left_warning_x": 200, "other_lane_margin_px": 80}
DEFAULT_OPEN_SOURCE_CONFIG: dict[str, Any] = {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "protocol": "responses",
    "vision_mode": "text_only",
    "reasoning_effort": "low",
    "api_key_environment_variable": "PVZ_LLM_API_KEY",
    "image_transfer_authorized": True,
    "timeout_seconds": 30,
    "max_output_tokens": 320,
}
SUPPORTED_PROTOCOLS = ("responses", "chat_completions")
SUPPORTED_VISION_MODES = ("text_only", "low_jpeg", "high_png")
SUPPORTED_REASONING_EFFORTS = ("low", "medium", "auto")


def validate_base_url(url: str) -> str:
    """Validate an OpenAI-compatible base URL (HTTPS or local loopback HTTP, no credentials/query)."""
    cleaned = (url or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(cleaned)
    host = (parsed.hostname or "").lower()
    is_local = host in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme not in ("https", "http") or (parsed.scheme == "http" and not is_local):
        raise ValueError("Base URL 必须使用 https://（仅本机 localhost/127.0.0.1 允许 http://）")
    if not host or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Base URL 格式无效（不得包含内嵌凭据、查询参数或锚点）")
    return cleaned


def encode_rgb_png_data_url(rgb_bytes: bytes, size: tuple[int, int] = (800, 600)) -> str:
    """Encode an 800x600 RGB frame purely in memory as a PNG data URL (0 disk writes)."""
    buf = io.BytesIO()
    with Image.frombytes("RGB", size, rgb_bytes) as im:
        im.save(buf, format="PNG", compress_level=1)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def encode_rgb_jpeg_data_url(
    rgb_bytes: bytes,
    size: tuple[int, int] = (800, 600),
    target_size: tuple[int, int] = (512, 384),
    quality: int = 75,
) -> str:
    """Encode an RGB frame purely in memory as a compact downscaled JPEG data URL (0 disk writes)."""
    buf = io.BytesIO()
    with Image.frombytes("RGB", size, rgb_bytes) as im:
        small = im.resize(target_size, Image.Resampling.BILINEAR)
        small.save(buf, format="JPEG", quality=quality, optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class _KeepAlivePool:
    """Thread-safe HTTP/HTTPS Keep-Alive connection pool (avoids repeated TCP+TLS handshakes)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conns: dict[tuple[str, str, int, str], http.client.HTTPConnection] = {}
        self._ssl_ctx = ssl.create_default_context()

    def _open_conn(self, scheme: str, host: str, port: int, proxy_url: str, timeout: float) -> http.client.HTTPConnection:
        if proxy_url:
            pp = urllib.parse.urlsplit(proxy_url)
            if pp.scheme in ("http", "https") and pp.hostname:
                p_port = pp.port or (443 if pp.scheme == "https" else 80)
                if scheme == "https":
                    conn = http.client.HTTPSConnection(pp.hostname, p_port, timeout=timeout, context=self._ssl_ctx)
                    conn.set_tunnel(host, port)
                    return conn
                return http.client.HTTPConnection(pp.hostname, p_port, timeout=timeout)
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout, context=self._ssl_ctx)
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def post_json(self, url: str, body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        proxies = urllib.request.getproxies()
        proxy_url = "" if host in ("127.0.0.1", "localhost", "::1") else (proxies.get(scheme) or proxies.get("all") or "")
        # Fallback to urllib if proxy is non-HTTP (e.g. socks5) or contains auth
        if proxy_url:
            pp = urllib.parse.urlsplit(proxy_url)
            if pp.scheme not in ("http", "https") or pp.username or pp.password:
                return _urllib_post_json(url, body, key, timeout)

        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "Content-Length": str(len(payload)),
        }
        req_target = url if (proxy_url and scheme == "http") else path
        pool_key = (scheme, host, port, proxy_url)

        for attempt in range(2):
            with self._lock:
                conn = self._conns.pop(pool_key, None)
            reused = conn is not None
            try:
                if conn is None:
                    conn = self._open_conn(scheme, host, port, proxy_url, timeout)
                else:
                    conn.timeout = timeout
                    if conn.sock is not None:
                        conn.sock.settimeout(timeout)
                conn.request("POST", req_target, body=payload, headers=headers)
                resp = conn.getresponse()
                raw = resp.read(2 * 1024 * 1024 + 1)
                status = resp.status
                close_hdr = (resp.getheader("Connection") or "").lower()
                if status < 400 and close_hdr != "close":
                    with self._lock:
                        old = self._conns.get(pool_key)
                        self._conns[pool_key] = conn
                    if old is not None and old is not conn:
                        try:
                            old.close()
                        except Exception:
                            pass
                else:
                    try:
                        conn.close()
                    except Exception:
                        pass
                if status >= 400:
                    raise ProviderError("provider_http_error", status)
                if len(raw) > 2 * 1024 * 1024:
                    raise ProviderError("provider_response_too_large")
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeError):
                    raise ProviderError("provider_invalid_json") from None
            except ProviderError:
                raise
            except (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionResetError, BrokenPipeError):
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                if reused and attempt == 0:
                    continue
                raise ProviderError("provider_transport_error") from None
            except Exception as exc:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
                raise ProviderError("provider_timeout" if timed_out else "provider_transport_error") from None
        raise ProviderError("provider_transport_error")

    def close_all(self) -> None:
        with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
        for c in conns:
            try:
                c.close()
            except Exception:
                pass


_SHARED_KEEPALIVE_POOL = _KeepAlivePool()


def _urllib_post_json(url: str, body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ProviderError("provider_http_error", exc.code) from None
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
        raise ProviderError("provider_timeout" if timed_out else "provider_transport_error") from None
    if len(raw) > 2 * 1024 * 1024:
        raise ProviderError("provider_response_too_large")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("provider_invalid_json") from None


def _default_transport(url: str, body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
    return _SHARED_KEEPALIVE_POOL.post_json(url, body, key, timeout)


def _default_models_transport(url: str, key: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ProviderError("provider_http_error", exc.code) from None
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
        raise ProviderError("provider_timeout" if timed_out else "provider_transport_error") from None
    if len(raw) > 2 * 1024 * 1024:
        raise ProviderError("provider_response_too_large")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("provider_invalid_json") from None


def parse_model_ids(payload: Any) -> list[str]:
    """Extract sorted unique model IDs from OpenAI-compatible GET /models JSON."""
    items = None
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload["data"]
        elif isinstance(payload.get("models"), list):
            items = payload["models"]
    elif isinstance(payload, list):
        items = payload
    if not isinstance(items, list):
        raise ProviderError("provider_response_shape_invalid")
    seen: set[str] = set()
    models: list[str] = []
    for item in items:
        mid = item.get("id") or item.get("name") if isinstance(item, dict) else (item if isinstance(item, str) else None)
        if isinstance(mid, str):
            mid = mid.strip()
            if mid and mid not in seen:
                seen.add(mid)
                models.append(mid)
    models.sort()
    return models


class UserConfigurableProvider:
    """OpenAI-compatible client (responses or chat_completions) using in-memory key or local env var."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        key_getter: Callable[[], str],
        transport: Callable[[str, dict[str, Any], str, float], dict[str, Any]] = _default_transport,
    ):
        self.config = dict(config)
        self.base_url = validate_base_url(self.config["base_url"])
        model = str(self.config.get("model", "")).strip()
        if not model:
            raise ValueError("模型 ID 不能为空")
        self.model = model
        protocol = str(self.config.get("protocol", "responses")).strip()
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"不支持的协议：{protocol}")
        self.protocol = protocol
        self._key_getter = key_getter
        self._transport = transport
        self._reasoning_unsupported = False

    def complete(
        self,
        prompt: str,
        *,
        image_data_url: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        key = self._key_getter()
        if not key:
            raise ProviderError("missing_local_api_key")
        effort = str(self.config.get("reasoning_effort", "low")).strip().lower()
        use_reasoning = (not self._reasoning_unsupported) and effort in ("low", "medium")
        default_tokens = 320 if use_reasoning else 180
        tokens = int(max_tokens or min(int(self.config.get("max_output_tokens", default_tokens)), default_tokens))
        timeout = float(self.config.get("timeout_seconds", 45))
        vision_mode = str(self.config.get("vision_mode", "text_only")).strip()
        img_detail = "high" if vision_mode == "high_png" else "low"

        def _build_body(include_reasoning: bool) -> tuple[str, dict[str, Any]]:
            if self.protocol == "responses":
                content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
                if image_data_url:
                    content.append({"type": "input_image", "image_url": image_data_url, "detail": img_detail})
                body: dict[str, Any] = {
                    "model": self.model,
                    "input": [{"role": "user", "content": content}],
                    "max_output_tokens": tokens,
                }
                if include_reasoning and effort in ("low", "medium"):
                    body["reasoning"] = {"effort": effort}
                if json_schema is not None:
                    body["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": "pvz_readonly_strategy",
                            "strict": True,
                            "schema": json_schema,
                        }
                    }
                return f"{self.base_url}/responses", body

            content = [{"type": "text", "text": prompt}]
            if image_data_url:
                content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": img_detail}})
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": tokens,
            }
            if include_reasoning and effort in ("low", "medium"):
                body["reasoning_effort"] = effort
            if json_schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pvz_readonly_strategy",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            return f"{self.base_url}/chat/completions", body

        endpoint, body = _build_body(use_reasoning)
        try:
            raw = self._transport(endpoint, body, key, timeout)
        except ProviderError as exc:
            # Some gateways/models reject reasoning/reasoning_effort with HTTP 400; retry once cleanly without it.
            if use_reasoning and exc.code == "provider_http_error" and exc.http_status == 400:
                self._reasoning_unsupported = True
                endpoint, body = _build_body(False)
                raw = self._transport(endpoint, body, key, timeout)
            else:
                raise
        text = extract_text(raw, self.protocol)
        return {
            "text": text,
            "protocol": self.protocol,
            "model_returned": raw.get("model") or self.model,
        }


@dataclass(frozen=True)
class VerifiedLlmAdvice:
    source_state_id: str
    reviewed_state_id: str
    model: str
    phase: str
    goal: str
    focus_row: int | None
    reason: str
    uncertainties: tuple[str, ...]
    latency_s: float
    verified_at: float
    epoch: int
    possibly_stale: bool = False
    stale_reason: str = ""


class AsyncLlmCoach:
    """Single-flight non-blocking async LLM coach; default disabled (0 network calls)."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        provider_factory: Callable[[dict[str, Any]], Any] | None = None,
        transport: Callable[[str, dict[str, Any], str, float], dict[str, Any]] = _default_transport,
        models_transport: Callable[[str, str, float], dict[str, Any]] = _default_models_transport,
        policy: dict[str, int] | None = None,
        enabled: bool = False,
        min_interval_s: float = 18.0,
        vision_mode: str = "text_only",
        reasoning_effort: str = "low",
    ):
        if config_path is not None:
            self.config_path = Path(config_path)
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        elif DEFAULT_TERRA_CONFIG.is_file():
            self.config_path = DEFAULT_TERRA_CONFIG
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        elif FALLBACK_SOL_CONFIG.is_file():
            self.config_path = FALLBACK_SOL_CONFIG
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        else:
            self.config_path = None
            self.config = dict(DEFAULT_OPEN_SOURCE_CONFIG)
        self.config["vision_mode"] = vision_mode if vision_mode in SUPPORTED_VISION_MODES else "text_only"
        self.config["reasoning_effort"] = reasoning_effort if reasoning_effort in SUPPORTED_REASONING_EFFORTS else "low"
        self.clock = clock
        self.policy = dict(policy or DEFAULT_POLICY)
        self._transport = transport
        self._models_transport = models_transport
        self._custom_provider_factory = provider_factory
        self._provider = None
        self._override_api_key: str = ""  # In-memory only; never persisted or logged
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-async-llm-coach")
        self._enabled = bool(enabled)
        self._min_interval_s = max(10.0, float(min_interval_s))
        self._closed = False
        self._epoch = 0
        self._inflight = False
        self._inflight_started_at: float | None = None
        self._next_allowed_at = 0.0
        self._pending_hypothesis: dict[str, Any] | None = None
        self._verified_advice: VerifiedLlmAdvice | None = None
        self._verified_bundle: dict[str, Any] | None = None
        self._status = "disabled" if not self._enabled else "idle"
        self._last_error: str | None = None
        self._requests_sent = 0
        self._verified_count = 0
        self._rejected_count = 0

    def _resolve_api_key(self) -> str:
        with self._lock:
            if self._override_api_key:
                return self._override_api_key
            env_name = self.config.get("api_key_environment_variable", "PVZ_LLM_API_KEY")
        try:
            return load_local_api_key(env_name)
        except Exception:
            return os.environ.get(env_name, "").strip()

    def has_api_key(self) -> bool:
        return bool(self._resolve_api_key())

    def api_key_source(self) -> str:
        with self._lock:
            if self._override_api_key:
                return "manual_memory"
        return "local_env" if self.has_api_key() else "missing"

    def configure_connection(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        protocol: str | None = None,
        vision_mode: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[bool, str]:
        """Apply user-supplied base_url / in-memory api_key / model / protocol / speed options without writing to disk."""
        try:
            clean_url = validate_base_url(base_url)
        except ValueError as exc:
            return False, str(exc)

        clean_model = (model if model is not None else str(self.config.get("model", ""))).strip()
        if not clean_model:
            return False, "模型 ID 不能为空"

        clean_proto = (protocol if protocol is not None else str(self.config.get("protocol", "responses"))).strip()
        if clean_proto not in SUPPORTED_PROTOCOLS:
            return False, f"协议仅支持：{' / '.join(SUPPORTED_PROTOCOLS)}"

        clean_vision = (vision_mode if vision_mode is not None else str(self.config.get("vision_mode", "text_only"))).strip()
        if clean_vision not in SUPPORTED_VISION_MODES:
            return False, f"请求模式仅支持：{' / '.join(SUPPORTED_VISION_MODES)}"

        clean_effort = (reasoning_effort if reasoning_effort is not None else str(self.config.get("reasoning_effort", "low"))).strip()
        if clean_effort not in SUPPORTED_REASONING_EFFORTS:
            return False, f"推理强度仅支持：{' / '.join(SUPPORTED_REASONING_EFFORTS)}"

        with self._lock:
            self.config["base_url"] = clean_url
            self.config["model"] = clean_model
            self.config["protocol"] = clean_proto
            self.config["vision_mode"] = clean_vision
            self.config["reasoning_effort"] = clean_effort
            if api_key is not None:
                self._override_api_key = api_key.strip()
            self._provider = None
            self._epoch += 1
            self._pending_hypothesis = None
            self._verified_advice = None
            self._verified_bundle = None
            self._last_error = None
            if self._enabled and not self._inflight:
                self._status = "idle"
            src = self.api_key_source()
        src_txt = (
            "自定义密钥（仅内存，不落盘）" if src == "manual_memory"
            else "本机环境变量 PVZ_LLM_API_KEY" if src == "local_env"
            else "未设置密钥"
        )
        return True, f"已应用：{clean_url} | {clean_model} ({clean_proto}/{clean_vision}/{clean_effort}) | {src_txt}"

    def clear_manual_api_key(self) -> str:
        with self._lock:
            self._override_api_key = ""
            self._provider = None
        return "已清空内存中的手动密钥（如已配本机环境变量则自动回退环境变量）"

    def fetch_models_async(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        callback: Callable[[bool, list[str], str], None],
    ) -> None:
        """Fetch available model IDs from GET {base_url}/models asynchronously without writing key to disk."""
        try:
            clean_url = validate_base_url(base_url)
        except ValueError as exc:
            callback(False, [], str(exc))
            return

        manual_key = (api_key or "").strip()

        def _run():
            t0 = self.clock()
            key = manual_key or self._resolve_api_key()
            if not key:
                callback(False, [], "缺少 API Key，无法拉取模型列表")
                return
            endpoint = f"{clean_url}/models"
            try:
                raw = self._models_transport(endpoint, key, 12.0)
                models = parse_model_ids(raw)
                dt = max(0.0, self.clock() - t0)
                if not models:
                    callback(False, [], f"已连接（{dt:.1f}s），但接口未返回可用模型列表")
                    return
                callback(True, models, f"已自动拉取 {len(models)} 个可用模型（{dt:.1f}s）")
            except ProviderError as exc:
                dt = max(0.0, self.clock() - t0)
                callback(False, [], f"拉取模型列表失败（{dt:.1f}s）：{exc}")
            except Exception as exc:
                dt = max(0.0, self.clock() - t0)
                callback(False, [], f"拉取模型列表失败（{dt:.1f}s）：{type(exc).__name__}")

        threading.Thread(target=_run, name="pvz-llm-fetch-models", daemon=True).start()

    def test_connection_async(self, callback: Callable[[bool, str], None]) -> None:
        """Send a single lightweight text-only ping ('只回复OK', 0 screenshots) in background."""
        def _run():
            t0 = self.clock()
            try:
                provider = self._get_provider()
                res = provider.complete("只回复OK", image_data_url=None, json_schema=None, max_tokens=16)
                dt = max(0.0, self.clock() - t0)
                reply = str(res.get("text", "")).strip()[:24]
                model_ret = res.get("model_returned") or self.config.get("model", "")
                callback(True, f"连通成功（{dt:.1f}s）· 模型 {model_ret} 返回：{reply}")
            except ProviderError as exc:
                dt = max(0.0, self.clock() - t0)
                callback(False, f"连通失败（{dt:.1f}s）：{exc}")
            except Exception as exc:
                dt = max(0.0, self.clock() - t0)
                callback(False, f"连通失败（{dt:.1f}s）：{type(exc).__name__}")

        threading.Thread(target=_run, name="pvz-llm-ping", daemon=True).start()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._epoch += 1
                self._pending_hypothesis = None
                self._verified_advice = None
                self._verified_bundle = None
                self._status = "disabled"
                self._last_error = None
            elif not self._inflight:
                self._status = "idle"
                self._last_error = None

    def set_interval(self, seconds: float) -> None:
        with self._lock:
            self._min_interval_s = max(10.0, min(60.0, float(seconds)))

    def invalidate(self, reason: str = "continuity_lost") -> None:
        with self._lock:
            self._epoch += 1
            self._pending_hypothesis = None
            self._verified_advice = None
            self._verified_bundle = None
            if self._enabled and not self._inflight:
                self._status = f"cleared:{reason}"

    def _get_provider(self):
        with self._lock:
            if self._provider is None:
                cfg_copy = copy.deepcopy(self.config)
                if self._custom_provider_factory is not None:
                    self._provider = self._custom_provider_factory(cfg_copy)
                else:
                    self._provider = UserConfigurableProvider(
                        cfg_copy, key_getter=self._resolve_api_key, transport=self._transport
                    )
            return self._provider

    def maybe_dispatch(
        self,
        *,
        state: dict[str, Any],
        rgb_bytes: bytes,
        phase_info: dict[str, Any],
        channel_info: dict[str, Any],
    ) -> bool:
        """Dispatch at most one background request if enabled, idle, key present, and interval elapsed."""
        snapshot = copy.deepcopy(state)
        phase_copy = copy.deepcopy(phase_info)
        ch_copy = copy.deepcopy(channel_info)
        with self._lock:
            if self._closed or not self._enabled or self._inflight:
                return False
            now = self.clock()
            if now < self._next_allowed_at:
                return False
            vision_mode = str(self.config.get("vision_mode", "text_only"))
            if vision_mode != "text_only" and not self.config.get("image_transfer_authorized"):
                self._status = "blocked_no_image_auth"
                return False
            if self._provider is None and self._custom_provider_factory is None and not self.has_api_key():
                self._status = "missing_api_key"
                return False
            epoch = self._epoch
            # Only copy/hold RGB bytes in memory when an image mode is active
            frame_payload = rgb_bytes if vision_mode != "text_only" else b""
            try:
                self._executor.submit(self._worker, snapshot, frame_payload, phase_copy, ch_copy, epoch, now, vision_mode)
            except RuntimeError:
                return False
            self._inflight = True
            self._inflight_started_at = now
            self._next_allowed_at = now + self._min_interval_s
            self._status = "waiting"
            self._last_error = None
            self._requests_sent += 1
            return True

    def _build_prompt(self, state: dict[str, Any], phase_info: dict[str, Any], ch: dict[str, Any]) -> str:
        phase = phase_info.get("current_phase", "阶段待确认")
        focus = phase_info.get("current_focus", "持续观测")
        sun_str = str(ch["sun_estimate"]) if ch.get("sun_estimate") is not None else "未知"
        per_lane = ch.get("per_lane") or {}
        lane_str = "、".join(f"{r}路×{len(xs)}(x={int(round(min(xs)))})" for r, xs in sorted(per_lane.items())) or "无"
        ready = ch.get("ready_plants") or []
        ready_str = f"，可买：{','.join(ready[:6])}" if ready else ""
        return (
            f"PVZ 3-7只读顾问，给1条简短中文策略，不给坐标不点击。\n"
            f"阶段：{phase}（重点：{focus}；低威胁尽量对称，高威胁优先守线）。\n"
            f"局势：阳光={sun_str}，亮卡={ch.get('cards_bright')}{ready_str}；"
            f"僵尸={ch.get('zombie_heads')}({lane_str})，最左x={ch.get('leftmost_head_x')}；"
            f"已占={ch.get('cells_occupied')}/54，地面阳光={ch.get('sun_cues')}。\n"
            f"goal选collect_resources/defend_lane/develop_economy/hold；仅defend_lane填focus_row(1..6)否则null。\n"
            f"reason<=32字(1句具体建议)，uncertainties尽量空数组[](仅重大缺口写1条<=28字)。"
            f"原样回填 source_state_id={state['state_id']}"
        )

    def _worker(
        self,
        source_state: dict[str, Any],
        rgb_bytes: bytes,
        phase_info: dict[str, Any],
        channel_info: dict[str, Any],
        epoch: int,
        started_at: float,
        vision_mode: str = "text_only",
    ) -> None:
        try:
            provider = self._get_provider()
            if vision_mode == "high_png" and rgb_bytes:
                data_url: str | None = encode_rgb_png_data_url(rgb_bytes)
            elif vision_mode == "low_jpeg" and rgb_bytes:
                data_url = encode_rgb_jpeg_data_url(rgb_bytes)
            else:
                data_url = None
            prompt = self._build_prompt(source_state, phase_info, channel_info)
            res = provider.complete(prompt, image_data_url=data_url, json_schema=STRATEGY_SCHEMA)
            strategy = validate_strategy(strict_json(res["text"]), source_state["state_id"])
            latency_s = max(0.0, self.clock() - started_at)
            model_name = res.get("model_returned") or self.config.get("model", "")
            with self._lock:
                self._inflight = False
                self._inflight_started_at = None
                if self._closed or not self._enabled or epoch != self._epoch:
                    self._rejected_count += 1
                    self._status = "rejected_stale"
                    return
                self._pending_hypothesis = {
                    "source": source_state,
                    "strategy": strategy,
                    "model": model_name,
                    "phase": phase_info.get("current_phase", "阶段待确认"),
                    "latency_s": latency_s,
                    "epoch": epoch,
                }
                self._status = "pending_recheck"
        except Exception as exc:
            with self._lock:
                self._inflight = False
                self._inflight_started_at = None
                self._last_error = str(exc) if isinstance(exc, ProviderError) else type(exc).__name__
                self._status = "error" if self._enabled else "disabled"

    def recheck_against_current(self, current_state: dict[str, Any] | None) -> VerifiedLlmAdvice | None:
        """Recheck any pending or retained LLM hypothesis against the current <=2.0s published snapshot."""
        with self._lock:
            now = self.clock()
            if not self._enabled:
                self._verified_advice = None
                self._verified_bundle = None
                return None
            if current_state is None or not fresh(current_state, now):
                self._verified_advice = None
                self._verified_bundle = None
                return None

            newly_verified = False
            hyp = self._pending_hypothesis
            if hyp is not None:
                self._pending_hypothesis = None
                if hyp["epoch"] != self._epoch:
                    self._rejected_count += 1
                    self._status = "rejected_stale"
                else:
                    src_age = now - hyp["source"]["frame_time_seconds"]
                    age_ok = 0 <= src_age <= 30.0
                    ok, _reason = self._validate_hypothesis(hyp["source"], current_state, hyp["strategy"])
                    stale = not (age_ok and ok)
                    stale_reason = "" if not stale else ("expired" if not age_ok else _reason)
                    st = hyp["strategy"]
                    self._verified_advice = VerifiedLlmAdvice(
                        source_state_id=hyp["source"]["state_id"],
                        reviewed_state_id=current_state["state_id"],
                        model=hyp["model"],
                        phase=hyp["phase"],
                        goal=st["goal"],
                        focus_row=st["focus_row"],
                        reason=st["reason"],
                        uncertainties=tuple(st["uncertainties"]),
                        latency_s=hyp["latency_s"],
                        verified_at=now,
                        epoch=hyp["epoch"],
                        possibly_stale=stale,
                        stale_reason=stale_reason,
                    )
                    self._verified_bundle = {"source": hyp["source"], "strategy": st}
                    self._verified_count += 1
                    self._status = "verified_stale" if stale else "verified"
                    newly_verified = True

            if self._verified_advice is not None and not newly_verified:
                if (
                    self._verified_advice.epoch != self._epoch
                    or (now - self._verified_advice.verified_at) > 25.0
                    or self._verified_bundle is None
                ):
                    self._verified_advice = None
                    self._verified_bundle = None
                else:
                    ok_ret, reason_ret = self._validate_hypothesis(
                        self._verified_bundle["source"], current_state, self._verified_bundle["strategy"]
                    )
                    if not ok_ret and not self._verified_advice.possibly_stale:
                        adv = self._verified_advice
                        self._verified_advice = VerifiedLlmAdvice(
                            source_state_id=adv.source_state_id,
                            reviewed_state_id=adv.reviewed_state_id,
                            model=adv.model,
                            phase=adv.phase,
                            goal=adv.goal,
                            focus_row=adv.focus_row,
                            reason=adv.reason,
                            uncertainties=adv.uncertainties,
                            latency_s=adv.latency_s,
                            verified_at=adv.verified_at,
                            epoch=adv.epoch,
                            possibly_stale=True,
                            stale_reason=reason_ret,
                        )
                        self._status = "verified_stale"
            return self._verified_advice

    def _validate_hypothesis(
        self, source: dict[str, Any], current: dict[str, Any], strategy: dict[str, Any]
    ) -> tuple[bool, str]:
        goal = strategy["goal"]
        try:
            if goal == "defend_lane":
                defense_evidence(source, current, strategy["focus_row"], self.policy)
            elif goal == "collect_resources":
                resource_evidence(source, current, self.policy)
            else:
                # Macro planning advice (develop_economy / hold): tolerate routine far-right spawns
                # so slower models remain useful, but withdraw immediately on urgent left-warning or wave surge.
                observed_basis(current)
                src_heads = source.get("zombies", {}).get("candidates", [])
                cur_heads = current.get("zombies", {}).get("candidates", [])
                left_warn = int(self.policy.get("left_warning_x", 200))
                if any(float(c["head_xy"][0]) <= left_warn for c in cur_heads if isinstance(c, dict) and "head_xy" in c):
                    return False, "urgent_threat_appeared"
                if len(cur_heads) - len(src_heads) >= 3:
                    return False, "threat_wave_surged"
            return True, "ok"
        except ReviewBlocked as exc:
            return False, exc.code
        except Exception:
            return False, "context_unknown"

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            waiting_s = (
                round(max(0.0, now - self._inflight_started_at), 1)
                if self._inflight and self._inflight_started_at is not None else 0.0
            )
            advice = self._verified_advice
            if advice is not None and (advice.epoch != self._epoch or (now - advice.verified_at) > 25.0):
                advice = None
            return {
                "enabled": self._enabled,
                "base_url": self.config.get("base_url", ""),
                "model": self.config.get("model", ""),
                "protocol": self.config.get("protocol", "responses"),
                "vision_mode": self.config.get("vision_mode", "text_only"),
                "reasoning_effort": self.config.get("reasoning_effort", "low"),
                "has_api_key": self.has_api_key(),
                "api_key_source": self.api_key_source(),
                "interval_s": int(self._min_interval_s),
                "status": self._status,
                "inflight": self._inflight,
                "waiting_s": waiting_s,
                "last_error": self._last_error,
                "requests_sent": self._requests_sent,
                "verified_count": self._verified_count,
                "rejected_count": self._rejected_count,
                "advice": advice,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._enabled = False
            self._override_api_key = ""
            self._epoch += 1
            self._pending_hypothesis = None
            self._verified_advice = None
            self._verified_bundle = None
        self._executor.shutdown(wait=False, cancel_futures=True)
