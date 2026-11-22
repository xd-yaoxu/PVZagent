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
import io
import json
import os
from pathlib import Path
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
    "api_key_environment_variable": "PVZ_LLM_API_KEY",
    "image_transfer_authorized": True,
    "timeout_seconds": 30,
    "max_output_tokens": 220,
}
SUPPORTED_PROTOCOLS = ("responses", "chat_completions")


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


def _default_transport(url: str, body: dict[str, Any], key: str, timeout: float) -> dict[str, Any]:
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
        tokens = int(max_tokens or self.config.get("max_output_tokens", 512))
        timeout = float(self.config.get("timeout_seconds", 45))

        if self.protocol == "responses":
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            if image_data_url:
                content.append({"type": "input_image", "image_url": image_data_url, "detail": "high"})
            body: dict[str, Any] = {
                "model": self.model,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": tokens,
            }
            effort = self.config.get("reasoning_effort")
            if effort in ("low", "medium", "high"):
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
            endpoint = f"{self.base_url}/responses"
        else:
            content = [{"type": "text", "text": prompt}]
            if image_data_url:
                content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}})
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": tokens,
            }
            if json_schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pvz_readonly_strategy",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            endpoint = f"{self.base_url}/chat/completions"

        raw = self._transport(endpoint, body, key, timeout)
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
    ) -> tuple[bool, str]:
        """Apply user-supplied base_url / in-memory api_key / model / protocol without writing to disk."""
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

        with self._lock:
            self.config["base_url"] = clean_url
            self.config["model"] = clean_model
            self.config["protocol"] = clean_proto
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
        return True, f"已应用：{clean_url} | 模型 {clean_model} ({clean_proto}) | 密钥来源：{src_txt}"

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
            if not self.config.get("image_transfer_authorized"):
                self._status = "blocked_no_image_auth"
                return False
            if self._provider is None and self._custom_provider_factory is None and not self.has_api_key():
                self._status = "missing_api_key"
                return False
            epoch = self._epoch
            try:
                self._executor.submit(self._worker, snapshot, rgb_bytes, phase_copy, ch_copy, epoch, now)
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
        return (
            f"PVZ原版3-7实时只读顾问，只给一条中文战略建议，不给像素坐标，不执行点击。\n"
            f"当前阶段：{phase}（重点：{focus}；低威胁时尽量对称整齐，有威胁优先守线）。\n"
            f"当前观测摘要：阳光={sun_str}，亮卡={ch.get('cards_bright')}，确认僵尸头={ch.get('zombie_heads')}，"
            f"最左x={ch.get('leftmost_head_x')}，已占格={ch.get('cells_occupied')}/54，地面阳光候选={ch.get('sun_cues')}。\n"
            f"goal 仅选 collect_resources/defend_lane/develop_economy/hold；仅 defend_lane 填 focus_row(1..6)，其余 null。\n"
            f"reason<=48字（1-2句具体可执行建议），uncertainties<=2条每条<=48字。未知不猜。原样回填 source_state_id={state['state_id']}"
        )

    def _worker(
        self,
        source_state: dict[str, Any],
        rgb_bytes: bytes,
        phase_info: dict[str, Any],
        channel_info: dict[str, Any],
        epoch: int,
        started_at: float,
    ) -> None:
        try:
            provider = self._get_provider()
            data_url = encode_rgb_png_data_url(rgb_bytes)
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
                elif not 0 <= now - hyp["source"]["frame_time_seconds"] <= 30.0:
                    self._rejected_count += 1
                    self._status = "rejected_expired"
                else:
                    ok, _reason = self._validate_hypothesis(hyp["source"], current_state, hyp["strategy"])
                    if ok:
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
                        )
                        self._verified_bundle = {"source": hyp["source"], "strategy": st}
                        self._verified_count += 1
                        self._status = "verified"
                        newly_verified = True
                    else:
                        self._rejected_count += 1
                        self._verified_advice = None
                        self._verified_bundle = None
                        self._status = f"rejected:{_reason}"

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
                    if not ok_ret:
                        self._verified_advice = None
                        self._verified_bundle = None
                        self._rejected_count += 1
                        self._status = f"rejected:{reason_ret}"
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
                if observed_basis(source)["headed_zones"] != observed_basis(current)["headed_zones"]:
                    return False, "threats_changed"
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
