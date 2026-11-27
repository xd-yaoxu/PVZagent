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
import re
import ssl
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

from ..goal_review import ReviewBlocked, defense_evidence, validate_policy
from ..llm_advice import strict_json
from ..llm_provider import ProviderError, extract_text, load_local_api_key
from ..resource_review import resource_evidence
from ..resource_threat_v2 import validate_threat_context
from ..strategy_review import STRATEGY_SCHEMA, fresh, observed_basis, validate_strategy
from .zombie_intel import (
    WATER_ROWS, extended_species_prompt_hint, get_zombie_profile, is_localized_vehicle,
    is_plant_legal_for_row, supplementary_snorkels,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TERRA_CONFIG = REPO_ROOT / "config/product-strategy-terra-v1.json"
FALLBACK_SOL_CONFIG = REPO_ROOT / "config/llm-strategy-v1.json"
DEFAULT_POLICY = {"left_warning_x": 200, "other_lane_margin_px": 80}

_TERRAIN_PLANTS = {
    "土豆雷": "potato_mine", "地刺": "spikeweed", "缠绕海草": "tangle_kelp", "海草": "tangle_kelp", "睡莲": "lily_pad",
}
_TERRAIN_PLANTS.update({plant: plant for plant in tuple(_TERRAIN_PLANTS.values())})
_PLANT_MENTION = re.compile("|".join(_TERRAIN_PLANTS))
_ROW_NUMBER = r"[1-6１２３４５６一二三四五六]"
_ROW_MENTION = re.compile(
    rf"(?<![\d])(?:第\s*)?{_ROW_NUMBER}(?:\s*(?:[/／、及和与或]|以及)\s*{_ROW_NUMBER})*\s*(?:水|陆)?[路行]"
)
_ROW_DIGITS = str.maketrans("１２３４５６一二三四五六", "123456123456")
_NON_PLACEMENT = re.compile(
    r"(?:勿|莫|别|严禁|禁止|不要|不能|不可|不得|不应|不宜|不必|无需|不用|不放|不种|不补|不建议|不推荐|"
    r"无须|没有|已有|已放|已种|已补|已部署|已确认|原有|现有|缺少|缺乏|无)(?:建议|推荐)?"
    r"[\s:：可用已亮选放种补再加埋植摆下铺垫上下在往本该此水陆路地一株颗个棵张的或、/]*$"
)
_PLANT_STATUS = re.compile(
    r"^\s*(?:已|正在|还在|仍在|在冷却|就绪|冷却中|亮卡|未就绪|在场|稳守|无需|足够|被吃|"
    r"勿|不[可要能应宜得](?:放|种|补|用)|仅限)"
)
_TERRAIN_ERRORS = {"illegal_terrain_water_land_only", "illegal_terrain_land_water_only"}


def validate_terrain_advice(strategy: dict[str, Any]) -> tuple[bool, str]:
    """Check plant mentions against their placement lane in the compact reason."""
    focus = strategy.get("focus_row") if strategy.get("goal") == "defend_lane" else None
    for clause in re.split(r"[，,。；;！!？?\n]|(?:但是|而是|改为|改放(?=.)|否则)", strategy.get("reason") or ""):
        rows = list(_ROW_MENTION.finditer(clause))
        last_plant_end = 0
        previous_negative = False
        for mention in _PLANT_MENTION.finditer(clause):
            before = clause[last_plant_end:mention.start()]
            after = clause[mention.end():]
            last_plant_end = mention.end()
            previous_negative = bool(
                _NON_PLACEMENT.search(_ROW_MENTION.sub("", before))
                or (previous_negative and re.fullmatch(r"\s*(?:和|与|或|[/／、])\s*", before))
                or (_PLANT_STATUS.match(after) and not re.search(r"[放种补埋铺垫摆加下植]\s*$", before))
            )
            if previous_negative:
                continue
            prior_row = next((m for m in reversed(rows) if m.end() <= mention.start()), None)
            next_row = next((m for m in rows if m.start() >= mention.end()), None)
            row_mention = prior_row
            if next_row is not None:
                between = clause[mention.end():next_row.start()]
                if re.fullmatch(r"\s*(?:可|先|再)?(?:放|种|补|埋|摆|部署)?(?:在|到|于|给)\s*", between) or (
                    prior_row is None and re.fullmatch(r"\s*(?:防守|防|守住?|处理|应对|拦截)?\s*", between)
                ):
                    row_mention = next_row
            target_rows = (
                [int(n) for n in re.findall(r"[1-6]", row_mention[0].translate(_ROW_DIGITS))]
                if row_mention else [focus]
            )
            for row in target_rows:
                if row is not None and not is_plant_legal_for_row(_TERRAIN_PLANTS[mention[0]], row):
                    return False, (
                        "illegal_terrain_water_land_only" if row in WATER_ROWS else "illegal_terrain_land_water_only"
                    )
    return True, "ok"


def _localized_lane_threats(state: dict[str, Any]) -> dict[int, list[float]]:
    if state.get("scene", {}).get("value") != "playing":
        raise ReviewBlocked("nonplaying_scene")
    validate_threat_context(state)
    zombies, vehicles = state["zombies"], state["vehicles"]
    if zombies["uncertain_edge_presence"] or zombies["uncertain_water_candidates"] or vehicles["uncertain_candidates"]:
        raise ReviewBlocked("unlocalized_threat_context")
    if any(not is_localized_vehicle(vehicle) for vehicle in vehicles["candidates"]):
        raise ReviewBlocked("vehicle_position_unknown")

    lanes: dict[int, list[float]] = {}
    for candidates, point_key in ((zombies["candidates"], "head_xy"), (vehicles["candidates"], "front_xy")):
        for candidate in candidates:
            lanes.setdefault(candidate["row"], []).append(float(candidate[point_key][0]))
    for row, x in supplementary_snorkels(state.get("zombie_intel") or {}):
        lanes.setdefault(row, []).append(x)
    return lanes


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
SUPPORTED_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "auto")
_REASONING_TOKENS = {"none": 140, "minimal": 140, "low": 240, "medium": 320, "auto": 160, None: 160}
_REASONING_FALLBACK = {"none": "minimal", "minimal": "low", "medium": "low", "low": None}
_SCHEMA_FALLBACK = {"strict_json": "json_object", "json_object": "plain"}
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_COMPAT_RETRIES = 5

_COMPACT_FIELDS = ("goal", "focus_row", "reason")
LLM_COMPACT_STRATEGY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_COMPACT_FIELDS),
    "properties": {field: copy.deepcopy(STRATEGY_SCHEMA["properties"][field]) for field in _COMPACT_FIELDS},
}


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


def _decode_json(raw: bytes) -> dict[str, Any]:
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ProviderError("provider_response_too_large")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("provider_invalid_json") from None
    if not isinstance(data, dict):
        raise ProviderError("provider_response_shape_invalid")
    return data


def _http_error(status: int, response) -> ProviderError:
    error = ProviderError("provider_http_error", status)
    if status == 400:
        try:
            raw = response.read(16 * 1024).decode("utf-8", errors="replace")
        except (OSError, http.client.HTTPException):
            return error
        # Retain capability hints only. Upstream errors can echo request content.
        error.compatibility_hints = frozenset(re.findall(
            r"\b(?:store|json_schema|json_object|response_format|text\.format|strict|stream|streaming)\b",
            raw.lower(),
        ))
    return error


class _StreamResponse(dict):
    def __init__(self, payload: dict[str, Any], ttft_ms: float | None):
        super().__init__(payload)
        self.ttft_ms = ttft_ms


def _read_sse(response, *, responses: bool, started_at: float) -> _StreamResponse:
    chunks: list[str] = []
    data: list[bytes] = []
    event_type = ""
    ttft_ms = None
    total_bytes = 0
    model = None
    finish_reason = None
    done = False

    while True:
        line = response.readline(_MAX_RESPONSE_BYTES - total_bytes + 1)
        eof = not line
        total_bytes += len(line)
        if total_bytes > _MAX_RESPONSE_BYTES:
            raise ProviderError("provider_response_too_large")
        line = line.rstrip(b"\r\n")
        if line:
            field, _, value = line.partition(b":")
            if field == b"data":
                data.append(value.removeprefix(b" "))
            elif field == b"event":
                event_type = value.strip().decode("ascii", errors="replace")
            continue
        if not data:
            if eof:
                break
            event_type = ""
            continue

        raw = b"\n".join(data)
        data.clear()
        if raw == b"[DONE]" and not responses:
            done = True
            break
        event = _decode_json(raw)
        kind = event.get("type") or event_type
        event_type = ""
        if kind in ("error", "response.failed", "response.incomplete") or event.get("error"):
            raise ProviderError("response_not_completed")
        if responses:
            if kind in ("response.refusal.delta", "response.refusal.done"):
                raise ProviderError("model_refusal")
            if kind == "response.completed":
                completed = event.get("response") or {}
                if completed.get("output"):
                    return _StreamResponse(completed, ttft_ms)
                if completed.get("status") not in (None, "completed"):
                    raise ProviderError("response_not_completed")
                model = completed.get("model") or model
                text = "".join(chunks)
                return _StreamResponse({
                    "model": model,
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                }, ttft_ms)
            if kind == "response.created":
                model = (event.get("response") or {}).get("model")
            delta = event.get("delta") if kind == "response.output_text.delta" else None
        else:
            model = event.get("model") or model
            choices = event.get("choices") or []
            choice = choices[0] if choices else {}
            finish_reason = choice.get("finish_reason") or finish_reason
            message = choice.get("delta") or {}
            if message.get("refusal"):
                raise ProviderError("model_refusal")
            delta = message.get("content")
        if isinstance(delta, str) and delta:
            if ttft_ms is None:
                ttft_ms = max(0.0, (time.perf_counter() - started_at) * 1000)
            chunks.append(delta)

    if responses or (not done and finish_reason is None):
        raise ProviderError("response_not_completed")
    return _StreamResponse({
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"content": "".join(chunks)}}],
    }, ttft_ms)


def _read_completion(response, stream_enabled: bool, started_at: float, url: str) -> dict[str, Any]:
    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    if stream_enabled and content_type == "text/event-stream":
        return _read_sse(response, responses=url.endswith("/responses"), started_at=started_at)
    return _decode_json(response.read(_MAX_RESPONSE_BYTES + 1))


class _KeepAlivePool:
    """Thread-safe HTTP/HTTPS Keep-Alive connection pool (avoids repeated TCP+TLS handshakes)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conns: dict[tuple[str, str, int, str], http.client.HTTPConnection] = {}
        self._warming: set[tuple[str, str, int, str]] = set()
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

    @staticmethod
    def _route(url: str) -> tuple[str, str, int, str]:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        proxies = urllib.request.getproxies()
        proxy_url = "" if host in ("127.0.0.1", "localhost", "::1") else (proxies.get(scheme) or proxies.get("all") or "")
        return scheme, host, port, proxy_url

    @staticmethod
    def _can_pool(proxy_url: str) -> bool:
        if not proxy_url:
            return True
        pp = urllib.parse.urlsplit(proxy_url)
        return pp.scheme in ("http", "https") and not (pp.username or pp.password)

    def warmup_connection_async(self, url: str, timeout: float) -> None:
        def connect() -> None:
            pool_key = None
            conn = None
            try:
                route = self._route(url)
                if not self._can_pool(route[3]):
                    return
                with self._lock:
                    if route in self._conns or route in self._warming:
                        return
                    self._warming.add(route)
                    pool_key = route
                conn = self._open_conn(*route, timeout)
                conn.connect()
                with self._lock:
                    if route not in self._conns:
                        self._conns[route] = conn
                        conn = None
            except Exception:
                pass
            finally:
                if pool_key is not None:
                    with self._lock:
                        self._warming.discard(pool_key)
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass

        try:
            threading.Thread(target=connect, name="pvz-llm-warmup", daemon=True).start()
        except RuntimeError:
            pass

    def post_json(self, url: str, body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
        started_at = time.perf_counter()
        pool_key = self._route(url)
        scheme, host, port, proxy_url = pool_key
        if not self._can_pool(proxy_url):
            return _urllib_post_json(url, body, key, timeout)
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "Content-Length": str(len(payload)),
        }
        req_target = url if (proxy_url and scheme == "http") else path
        for attempt in range(2):
            with self._lock:
                conn = self._conns.pop(pool_key, None)
            reused = conn is not None
            resp = None
            try:
                if conn is None:
                    conn = self._open_conn(scheme, host, port, proxy_url, timeout)
                else:
                    conn.timeout = timeout
                    if conn.sock is not None:
                        conn.sock.settimeout(timeout)
                conn.request("POST", req_target, body=payload, headers=headers)
                resp = conn.getresponse()
                if resp.status >= 400:
                    raise _http_error(resp.status, resp)
                result = _read_completion(resp, body.get("stream") is True, started_at, url)
                if resp.isclosed() and not resp.will_close:
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
                return result
            except ProviderError:
                if conn is not None:
                    conn.close()
                raise
            except (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionResetError, BrokenPipeError):
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
                if reused and resp is None and attempt == 0:
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
            finally:
                if resp is not None:
                    resp.close()
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
    started_at = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _read_completion(resp, body.get("stream") is True, started_at, url)
    except urllib.error.HTTPError as exc:
        with exc:
            raise _http_error(exc.code, exc) from None
    except ProviderError:
        raise
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)
        raise ProviderError("provider_timeout" if timed_out else "provider_transport_error") from None


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
        schema_mode: str | None = None,
        stream_enabled: bool | None = None,
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
        self._compat_lock = threading.Lock()
        effort = str(self.config.get("reasoning_effort", "low")).strip().lower()
        self._effective_reasoning_effort = effort if effort in SUPPORTED_REASONING_EFFORTS else "low"
        self._store_unsupported = False
        mode = schema_mode if schema_mode is not None else self.config.get("schema_mode", "strict_json")
        if mode not in _SCHEMA_FALLBACK:
            raise ValueError(f"不支持的 JSON 格式：{mode}")
        self._schema_fallback_mode = mode
        self.stream_enabled = bool(self.config.get("stream_enabled", False) if stream_enabled is None else stream_enabled)
        self._stream_unsupported = False

    def warmup_connection_async(self) -> None:
        if self._transport is _default_transport:
            _SHARED_KEEPALIVE_POOL.warmup_connection_async(self.base_url, 5.0)

    def _retry_after_400(
        self, exc: ProviderError, effort: str | None, store: bool, schema: str | None, stream: bool,
    ) -> bool:
        hints = getattr(exc, "compatibility_hints", ())
        schema_rejected = any(hint in hints for hint in ("json_schema", "json_object", "response_format", "text.format", "strict"))
        with self._compat_lock:
            if store and "store" in hints:
                self._store_unsupported = True
            elif schema in _SCHEMA_FALLBACK and schema_rejected:
                if self._schema_fallback_mode == schema:
                    self._schema_fallback_mode = _SCHEMA_FALLBACK[schema]
            elif stream and ("stream" in hints or "streaming" in hints):
                self._stream_unsupported = True
            elif effort in _REASONING_FALLBACK:
                if self._effective_reasoning_effort == effort:
                    self._effective_reasoning_effort = _REASONING_FALLBACK[effort]
            elif store:
                self._store_unsupported = True
            elif schema in _SCHEMA_FALLBACK:
                if self._schema_fallback_mode == schema:
                    self._schema_fallback_mode = _SCHEMA_FALLBACK[schema]
            elif stream:
                self._stream_unsupported = True
            else:
                return False
        return True

    def complete(
        self,
        prompt: str,
        *,
        instructions: str | None = None,
        image_data_url: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        key = self._key_getter()
        if not key:
            raise ProviderError("missing_local_api_key")
        timeout = float(self.config.get("timeout_seconds", 45))
        vision_mode = str(self.config.get("vision_mode", "text_only")).strip()
        img_detail = "high" if vision_mode == "high_png" else "low"
        responses = self.protocol == "responses"
        endpoint = f"{self.base_url}/{'responses' if responses else 'chat/completions'}"
        if responses:
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            if image_data_url:
                content.append({"type": "input_image", "image_url": image_data_url, "detail": img_detail})
        else:
            content = [{"type": "text", "text": prompt}]
            if image_data_url:
                content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": img_detail}})
        user_message = {"role": "user", "content": content}

        for attempt in range(_MAX_COMPAT_RETRIES + 1):
            with self._compat_lock:
                effort = self._effective_reasoning_effort
                store = responses and not self._store_unsupported
                schema = self._schema_fallback_mode if json_schema is not None else None
                stream = self.stream_enabled and not self._stream_unsupported
            default_tokens = _REASONING_TOKENS[effort]
            tokens = max_tokens if max_tokens is not None else min(
                int(self.config.get("max_output_tokens", default_tokens)), default_tokens
            )
            system_text = instructions
            if schema in ("json_object", "plain") and (not instructions or json_schema != LLM_COMPACT_STRATEGY_SCHEMA):
                json_hint = "Return only a JSON object matching this schema: " + json.dumps(json_schema, ensure_ascii=False)
                system_text = f"{instructions}\n{json_hint}" if instructions else json_hint
            body: dict[str, Any] = {"model": self.model}
            if responses:
                body.update(input=[user_message], max_output_tokens=tokens)
                if system_text:
                    body["instructions"] = system_text
                if store:
                    body["store"] = False
                if effort in _REASONING_FALLBACK:
                    body["reasoning"] = {"effort": effort}
            else:
                messages = [{"role": "system", "content": system_text}] if system_text else []
                body.update(messages=[*messages, user_message], max_tokens=tokens)
                if effort in _REASONING_FALLBACK:
                    body["reasoning_effort"] = effort
            if schema in _SCHEMA_FALLBACK:
                output_format: dict[str, Any] = {"type": "json_object"}
                if schema == "strict_json":
                    definition = {"name": "pvz_readonly_strategy", "strict": True, "schema": json_schema}
                    output_format = {"type": "json_schema"}
                    output_format.update(definition if responses else {"json_schema": definition})
                if responses:
                    body["text"] = {"format": output_format}
                else:
                    body["response_format"] = output_format
            if stream:
                body["stream"] = True
            try:
                raw = self._transport(endpoint, body, key, timeout)
            except ProviderError as exc:
                if exc.http_status != 400 or not self._retry_after_400(exc, effort, store, schema, stream):
                    raise
                if attempt == _MAX_COMPAT_RETRIES:
                    raise
                continue
            if stream and not isinstance(raw, _StreamResponse):
                with self._compat_lock:
                    self._stream_unsupported = True
            return {
                "text": extract_text(raw, self.protocol),
                "protocol": self.protocol,
                "model_returned": raw.get("model") or self.model,
                "effective_reasoning_effort": effort,
                "configured_reasoning_effort": self.config.get("reasoning_effort", "low"),
                "ttft_ms": raw.ttft_ms if isinstance(raw, _StreamResponse) else None,
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


@dataclass(frozen=True)
class _DispatchSignature:
    vehicle_rows: frozenset[int]
    warning_rows: frozenset[int]
    defense_rows: frozenset[int]

    def changed_from(self, previous: _DispatchSignature) -> bool:
        return bool(
            self.vehicle_rows - previous.vehicle_rows
            or self.warning_rows - previous.warning_rows
            or self.defense_rows != previous.defense_rows
        )


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
        stream_enabled: bool | None = None,
        event_cooldown_s: float = 3.5,
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
        if stream_enabled is not None:
            self.config["stream_enabled"] = bool(stream_enabled)
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
        self._min_interval_s = max(3.0, min(60.0, float(min_interval_s)))
        self._event_cooldown_s = max(3.0, float(event_cooldown_s))
        self._closed = False
        self._epoch = 0
        self._inflight = False
        self._inflight_started_at: float | None = None
        self._next_allowed_at = 0.0
        self._last_dispatch_at: float | None = None
        self._last_dispatch_sig: _DispatchSignature | None = None
        self._pending_hypothesis: dict[str, Any] | None = None
        self._verified_advice: VerifiedLlmAdvice | None = None
        self._verified_bundle: dict[str, Any] | None = None
        self._status = "disabled" if not self._enabled else "idle"
        self._last_error: str | None = None
        self._last_ttft_ms: float | None = None
        self._requests_sent = 0
        self._verified_count = 0
        self._rejected_count = 0

    def _resolve_api_key(self) -> str:
        with self._lock:
            if self._override_api_key:
                return self._override_api_key
            env_name = self.config.get("api_key_environment_variable", "PVZ_LLM_API_KEY")
        return self._read_env_api_key(env_name)

    @staticmethod
    def _read_env_api_key(env_name: str) -> str:
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
        stream_enabled: bool | None = None,
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
            new_config = {
                "base_url": clean_url, "model": clean_model, "protocol": clean_proto,
                "vision_mode": clean_vision, "reasoning_effort": clean_effort,
                "stream_enabled": bool(self.config.get("stream_enabled", False) if stream_enabled is None else stream_enabled),
            }
            new_key = self._override_api_key if api_key is None else api_key.strip()
            changed = new_key != self._override_api_key or any(
                self.config.get(name, False) != value for name, value in new_config.items()
            )
            if changed:
                self.config.update(new_config)
                self._override_api_key = new_key
                self._provider = None
                self._epoch += 1
                self._pending_hypothesis = None
                self._verified_advice = None
                self._verified_bundle = None
                self._last_error = None
                self._last_ttft_ms = None
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
            self.invalidate("api_key_cleared")
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
            protocol = self.config.get("protocol", "responses")

            def elapsed() -> str:
                ms = round(max(0.0, self.clock() - t0) * 1000)
                seconds = f"{ms / 1000:.3f}".rstrip("0").rstrip(".")
                return f"{ms} ms ({seconds}s)"

            try:
                provider = self._get_provider()
                res = provider.complete("只回复OK", image_data_url=None, json_schema=None)
                protocol = res.get("protocol") or protocol
                reply = str(res.get("text", "")).strip()[:24]
                model_ret = res.get("model_returned") or self.config.get("model", "")
                effort = res.get("effective_reasoning_effort")
                configured = res.get("configured_reasoning_effort", effort)
                fallback = f" · 生效思考档 {effort or '模型默认（省略参数）'}" if effort != configured else ""
                callback(True, f"连接成功 · {elapsed()} · {protocol}{fallback} · {model_ret} 返回 {reply}")
            except ProviderError as exc:
                callback(False, f"连接失败 · {elapsed()} · {protocol} · {exc}")
            except Exception as exc:
                callback(False, f"连接失败 · {elapsed()} · {protocol} · {type(exc).__name__}")

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
            self._min_interval_s = max(3.0, min(60.0, float(seconds)))
            if self._last_dispatch_at is not None:
                self._next_allowed_at = self._last_dispatch_at + self._min_interval_s

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
                    manual_key = self._override_api_key
                    env_name = cfg_copy.get("api_key_environment_variable", "PVZ_LLM_API_KEY")
                    self._provider = UserConfigurableProvider(
                        cfg_copy, key_getter=lambda: manual_key or self._read_env_api_key(env_name),
                        transport=self._transport,
                    )
            return self._provider

    def warmup_connection_async(self) -> None:
        with self._lock:
            if self._closed or not self._enabled:
                return
            try:
                warmup = getattr(self._get_provider(), "warmup_connection_async", None)
                if warmup is not None:
                    warmup()
            except Exception:
                pass

    def _dispatch_signature(self, state: dict[str, Any], channels: dict[str, Any]) -> _DispatchSignature:
        vehicles = tuple(
            vehicle for vehicle in state.get("vehicles", {}).get("candidates", ()) if is_localized_vehicle(vehicle)
        )
        vehicle_rows = frozenset(vehicle["row"] for vehicle in vehicles)
        per_lane = channels.get("per_lane")
        if per_lane is None:
            per_lane = {}
            for zombie in state.get("zombies", {}).get("candidates", ()):
                per_lane.setdefault(zombie["row"], []).append(zombie["head_xy"][0])
        warning_x = self.policy.get("left_warning_x", DEFAULT_POLICY["left_warning_x"])
        warning_rows = {row for row, xs in per_lane.items() if any(x <= warning_x for x in xs)}
        warning_rows.update(vehicle["row"] for vehicle in vehicles if vehicle["front_xy"][0] <= warning_x)
        defense = channels.get("lane_defense") or state.get("lane_defense") or {}
        threatened_rows = set(per_lane) | vehicle_rows
        return _DispatchSignature(vehicle_rows, frozenset(warning_rows),
                                  frozenset(defense.get("needs_defense_rows", threatened_rows)))

    def maybe_dispatch(
        self,
        *,
        state: dict[str, Any],
        rgb_bytes: bytes,
        phase_info: dict[str, Any],
        channel_info: dict[str, Any],
    ) -> bool:
        """Dispatch one background request on a material change or at the steady interval."""
        with self._lock:
            if self._closed or not self._enabled or self._inflight:
                return False
            now = self.clock()
            signature = self._dispatch_signature(state, channel_info)
            if now < self._next_allowed_at:
                if (
                    self._last_dispatch_at is None
                    or self._last_dispatch_sig is None
                    or now - self._last_dispatch_at < self._event_cooldown_s
                    or not signature.changed_from(self._last_dispatch_sig)
                ):
                    return False
            vision_mode = str(self.config.get("vision_mode", "text_only"))
            if vision_mode != "text_only" and not self.config.get("image_transfer_authorized"):
                self._status = "blocked_no_image_auth"
                return False
            uses_local_key = self._custom_provider_factory is None and (
                self._provider is None or isinstance(self._provider, UserConfigurableProvider)
            )
            if uses_local_key and not self.has_api_key():
                self._status = "missing_api_key"
                return False
            epoch = self._epoch
            # Only copy/hold RGB bytes in memory when an image mode is active
            frame_payload = rgb_bytes if vision_mode != "text_only" else b""
            try:
                self._executor.submit(
                    self._worker, copy.deepcopy(state), frame_payload,
                    copy.deepcopy(phase_info), copy.deepcopy(channel_info), epoch, now, vision_mode,
                )
            except RuntimeError:
                return False
            self._inflight = True
            self._inflight_started_at = now
            self._next_allowed_at = now + self._min_interval_s
            self._last_dispatch_at = now
            self._last_dispatch_sig = signature
            self._status = "waiting"
            self._last_error = None
            self._requests_sent += 1
            return True

    @staticmethod
    def _build_static_instructions(vision_mode: str) -> str:
        visual_hint = ""
        if vision_mode != "text_only":
            visual_hint = f"看图核实护甲磨损，并检查可见的其他经典僵尸：{extended_species_prompt_hint()}。\n"
        return (
            "PVZ 3-7只读顾问，给1条简短中文策略，不给坐标不点击。"
            "低威胁尽量对称，高威胁优先守线。\n"
            "稳守路严禁追加射手、坚果或一次性消耗植物。优先处理缺口，其次吃紧，未知先观察。"
            "若所有受威胁路均已稳守，建议收阳、补向日葵或睡莲发育，或留阳应对下一波。\n"
            "按潜水姿态和卡片选对策。投手和香蒲只在卡片栏或画面确认后纳入对策。\n"
            "地形铁律：1/2/5/6路为陆地，严禁推荐缠绕海草或睡莲；3/4路为水路，严禁推荐土豆雷或地刺，"
            "陆生植物在水路须先垫睡莲（樱桃也可放相邻2/5陆路溅射），缠绕海草仅限3/4水路。"
            "优先用空睡莲底座；新垫睡莲须有空水格、亮卡和足够总阳光；樱桃跨路须有可溅射的空陆格。\n"
            f"{visual_hint}"
            '仅输出JSON对象 {"goal":"hold","focus_row":null,"reason":"一句中文策略"}。'
            "goal选collect_resources/defend_lane/develop_economy/hold；仅defend_lane填focus_row(1..6)否则null。"
            "reason≤48字，用一句话给出具体对策，适用时说明针对的僵尸和推荐植物。"
        )

    def _build_dynamic_user_prompt(self, state: dict[str, Any], ch: dict[str, Any]) -> str:
        sun_str = str(ch["sun_estimate"]) if ch.get("sun_estimate") is not None else "未知"
        per_lane = ch.get("per_lane") or {}
        intel = ch.get("zombie_intel") or {}
        counter_plan = ch.get("counter_plan") or {}
        defense = ch.get("lane_defense") or state.get("lane_defense") or {}
        plants = ch.get("plant_intel") or state.get("plant_intel") or {}
        summaries = plants.get("row_summary", {})
        cells = (state.get("fields") or {}).get("cells") or ()

        def lane_label(row):
            return f"{row}路({'水' if row in WATER_ROWS else '陆'})"

        def pad_label(row):
            if row not in WATER_ROWS:
                return ""
            count = summaries.get(row, {}).get("empty_lily_pads")
            if count is None:
                count = sum(c.get("row") == row and c.get("occupancy") == "base_only" for c in cells)
            return f" 空睡莲x{count}" if count else " 无空睡莲"

        lane_str = "、".join(
            f"{lane_label(r)}{pad_label(r)}×{len(xs)}(x={int(round(min(xs)))})"
            for r, xs in sorted(per_lane.items())
        ) or "无"
        defense_lines = []
        for row, lane in sorted(defense.get("by_row", {}).items()):
            if lane["verdict"] == "clear":
                continue
            board = summaries.get(row, {})
            threats = intel.get("by_lane", {}).get(row, {})
            species = "+".join(
                f"{get_zombie_profile(kind).name_cn}x{count}" for kind, count in threats.get("species_counts", {}).items()
            )
            deployment = lane["plants_compact_zh"] or "无已确认部署"
            defense_lines.append(
                f"{lane_label(row)}{pad_label(row)} {deployment}，{board.get('total_dps', lane['effective_dps']):g} DPS"
                f"/阻挡{lane['blocker_hp']:g} HP｜{species or '确认威胁'}约{lane['threat_hp']:g}血｜{lane['verdict_zh']}"
            )
        defense_text = "每路现有部署与充分性评估\n" + "\n".join(defense_lines) + "\n" if defense_lines else ""
        summary = intel.get("summary_short")
        threat_summary = (
            re.sub(r"([1-6])路(?=[:：])", lambda m: lane_label(int(m[1])) + pad_label(int(m[1])), summary)
            if summary else lane_str
        )
        counter_line = counter_plan.get("llm_counter_line") or ""
        counter_text = f"对策参考：{counter_line}。\n" if counter_line else ""
        alternatives = counter_plan.get("alternatives") or []
        if alternatives:
            stance = counter_plan.get("target_armor_stage") or counter_plan.get("target_stance", "待确认")
            count = counter_plan.get("snorkel_count", 1)
            lane_count = counter_plan.get("lane_count", count)
            row = counter_plan.get("focus_row")
            lane_status = pad_label(row).strip()
            counter_text += f"潜水状态：{stance}，潜水×{count}，该路威胁×{lane_count}，{lane_status}。\n"
            if "备选对策：" not in counter_line:
                counter_text += f"备选对策：{'；'.join(alternatives[:3])}。\n"
        ready = ch.get("ready_plants") or []
        ready_str = f"，亮且够买：{','.join(ready[:6])}" if ready else ""
        vehicle_count = ch.get("confirmed_vehicles", 0)
        vehicle_text = f"，确认载具={vehicle_count}" if vehicle_count > 0 else ""
        return (
            f"局势：阳光={sun_str}，亮卡={ch.get('cards_bright')}{ready_str}；"
            f"威胁：{threat_summary}，僵尸头={ch.get('zombie_heads')}{vehicle_text}，"
            f"最左x={ch.get('leftmost_threat_x', ch.get('leftmost_head_x'))}；"
            f"已占={ch.get('cells_occupied')}/54，地面阳光={ch.get('sun_cues')}。\n"
            f"{defense_text}{counter_text}"
        )

    def _build_prompt_parts(
        self, state: dict[str, Any], ch_info: dict[str, Any], *, vision_mode: str | None = None,
    ) -> tuple[str, str]:
        mode = vision_mode if vision_mode is not None else self.config.get("vision_mode", "text_only")
        return self._build_static_instructions(mode), self._build_dynamic_user_prompt(state, ch_info)

    def _build_prompt(
        self, state: dict[str, Any], phase_info: dict[str, Any], ch: dict[str, Any] | None = None,
    ) -> str:
        """Keep the legacy combined prompt contract for injected providers."""
        if ch is None:
            ch, phase_info = phase_info, {}
        instructions, user_prompt = self._build_prompt_parts(state, ch)
        phase = phase_info.get("current_phase", "阶段待确认")
        focus = phase_info.get("current_focus", "持续观测")
        return (
            f"{instructions}\n阶段：{phase}（重点：{focus}）。\n{user_prompt}"
            "旧版输出还需 uncertainties 数组（无重大缺口时为空数组）。"
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
            with self._lock:
                if self._closed or not self._enabled or epoch != self._epoch:
                    self._inflight = False
                    self._inflight_started_at = None
                    self._rejected_count += 1
                    return
                provider = self._get_provider()
            if vision_mode == "high_png" and rgb_bytes:
                data_url: str | None = encode_rgb_png_data_url(rgb_bytes)
            elif vision_mode == "low_jpeg" and rgb_bytes:
                data_url = encode_rgb_jpeg_data_url(rgb_bytes)
            else:
                data_url = None
            if isinstance(provider, UserConfigurableProvider):
                instructions, prompt = self._build_prompt_parts(source_state, channel_info, vision_mode=vision_mode)
                res = provider.complete(
                    prompt, instructions=instructions, image_data_url=data_url, json_schema=LLM_COMPACT_STRATEGY_SCHEMA,
                )
            else:
                prompt = self._build_prompt(source_state, phase_info, channel_info)
                res = provider.complete(prompt, image_data_url=data_url, json_schema=STRATEGY_SCHEMA)
            parsed = strict_json(res["text"])
            source_id = source_state["state_id"]
            if isinstance(parsed, dict):
                returned_id = parsed.get("source_state_id")
                if "source_state_id" not in parsed or (isinstance(returned_id, str) and not returned_id.strip()):
                    parsed["source_state_id"] = source_id
                if parsed.get("uncertainties") is None:
                    parsed["uncertainties"] = []
            strategy = validate_strategy(parsed, source_id)
            terrain_ok, _ = validate_terrain_advice(strategy)
            latency_s = max(0.0, self.clock() - started_at)
            model_name = res.get("model_returned") or self.config.get("model", "")
            with self._lock:
                self._inflight = False
                self._inflight_started_at = None
                if self._closed or not self._enabled or epoch != self._epoch:
                    self._rejected_count += 1
                    self._status = "rejected_stale"
                    return
                if not terrain_ok:
                    self._rejected_count += 1
                    self._status = "rejected_illegal_terrain"
                    return
                self._pending_hypothesis = {
                    "source": source_state,
                    "strategy": strategy,
                    "model": model_name,
                    "phase": phase_info.get("current_phase", "阶段待确认"),
                    "latency_s": latency_s,
                    "epoch": epoch,
                }
                self._last_ttft_ms = res.get("ttft_ms")
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
                    if _reason in _TERRAIN_ERRORS:
                        self._rejected_count += 1
                        self._status = "rejected_illegal_terrain"
                        return self._verified_advice
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
                    if reason_ret in _TERRAIN_ERRORS:
                        self._verified_advice = None
                        self._verified_bundle = None
                        self._rejected_count += 1
                        self._status = "rejected_illegal_terrain"
                        return None
                    higher_deficit = (reason_ret == "higher_deficit_lane_present"
                                      and self._verified_advice.stale_reason == "lane_already_sufficient")
                    if not ok_ret and (not self._verified_advice.possibly_stale or higher_deficit):
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
        terrain_ok, reason = validate_terrain_advice(strategy)
        if not terrain_ok:
            return False, reason
        goal = strategy["goal"]
        try:
            by_row = (current.get("lane_defense") or {}).get("by_row", {})
            deficits = {row for row, lane in by_row.items() if lane.get("verdict") == "deficit"}
            if goal == "defend_lane" and by_row.get(strategy["focus_row"], {}).get("verdict") == "sufficient":
                return False, "higher_deficit_lane_present" if deficits else "lane_already_sufficient"
            if goal in ("collect_resources", "develop_economy", "hold", "hold_formation"):
                prior = (source.get("lane_defense") or {}).get("by_row", {})
                if any(prior.get(row, {}).get("verdict") != "deficit" for row in deficits):
                    return False, "lane_defense_deficit_appeared"
            source_has_vehicle = any(is_localized_vehicle(v) for v in source.get("vehicles", {}).get("candidates", ()))
            current_has_vehicle = any(is_localized_vehicle(v) for v in current.get("vehicles", {}).get("candidates", ()))
            if goal == "defend_lane":
                focus_row = strategy["focus_row"]
                has_vehicle = source_has_vehicle or current_has_vehicle
                has_supplement = any(
                    row == focus_row
                    for state in (source, current)
                    for row, _ in supplementary_snorkels(state.get("zombie_intel") or {})
                )
                if not (has_vehicle or has_supplement):
                    defense_evidence(source, current, focus_row, self.policy)
                else:
                    policy = validate_policy(self.policy)
                    left_warn = policy["left_warning_x"]
                    margin = policy["other_lane_margin_px"]
                    before = _localized_lane_threats(source)
                    after = _localized_lane_threats(current)
                    if not before.get(focus_row) or not after.get(focus_row):
                        return False, "focus_lane_unconfirmed"
                    if not has_vehicle and len(before[focus_row]) != len(after[focus_row]):
                        return False, "focus_count_changed"
                    src_focus, cur_focus = min(before[focus_row]), min(after[focus_row])
                    if (cur_focus <= left_warn) != (src_focus <= left_warn):
                        return False, "focus_warning_boundary_crossed"

                    def warnings(lanes, focus_x):
                        return {
                            row for row, xs in lanes.items()
                            if row != focus_row and (min(xs) <= left_warn or min(xs) + margin < focus_x)
                        }

                    if warnings(after, cur_focus) - warnings(before, src_focus):
                        return False, "new_other_lane_warning"
            elif goal == "collect_resources":
                resource_evidence(source, current, self.policy)
            else:
                # Macro planning advice (develop_economy / hold): tolerate routine far-right spawns
                # so slower models remain useful, but withdraw immediately on urgent left-warning or wave surge.
                if current_has_vehicle and not source_has_vehicle:
                    return False, "vehicle_threat_appeared"
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
                "stream_enabled": self.config.get("stream_enabled", False),
                "has_api_key": self.has_api_key(),
                "api_key_source": self.api_key_source(),
                "interval_s": int(self._min_interval_s),
                "status": self._status,
                "inflight": self._inflight,
                "waiting_s": waiting_s,
                "ttft_ms": self._last_ttft_ms,
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
