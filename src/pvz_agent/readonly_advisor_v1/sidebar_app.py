"""Single right-hand collapsible read-only advice sidebar UI (Tkinter, zero second windows)."""
from __future__ import annotations

import hashlib
import queue
import tkinter as tk
from tkinter import ttk
from typing import Any

from .advisor_runtime import AdvisorController
from .async_llm_coach import SUPPORTED_PROTOCOLS, SUPPORTED_REASONING_EFFORTS, validate_base_url
from .local_settings import LocalSettingsStore

EXPANDED_W = 420
EXPANDED_H = 720
COLLAPSED_W = 58
COLLAPSED_H = 220
RIGHT_MARGIN = 12
TOP_MARGIN = 36

_VISION_LABEL_TO_CODE = {
    "纯文本极速(不传图)": "text_only",
    "轻量看图(512px·low)": "low_jpeg",
    "高清看图(原画·high)": "high_png",
}
_VISION_CODE_TO_LABEL = {v: k for k, v in _VISION_LABEL_TO_CODE.items()}
_VISION_SHORT = {"text_only": "纯文本", "low_jpeg": "轻量图", "high_png": "高清图"}

_EFFORT_LABEL_TO_CODE = {
    "none(关闭思考)": "none",
    "minimal(最少思考)": "minimal",
    "low(极速)": "low",
    "medium(标准)": "medium",
    "auto(模型默认)": "auto",
}
_EFFORT_CODE_TO_LABEL = {v: k for k, v in _EFFORT_LABEL_TO_CODE.items()}


class ReadonlyAdvisorSidebarApp:
    """Single right-hand collapsible sidebar with Advice (default), Timeline, and Settings tabs."""

    def __init__(
        self,
        root: tk.Tk,
        controller: AdvisorController,
        *,
        settings_store: LocalSettingsStore | None = None,
    ):
        self.root = root
        self.controller = controller
        self.settings_store = settings_store if settings_store is not None else LocalSettingsStore()
        self._collapsed = False
        self._closed = False
        self._active_tab = "advice"
        self._selecting = False
        self._select_remaining = 0
        self._poll_handle = None
        self._select_handle = None
        self._fetch_debounce_handle = None
        self._fetch_seq = 0
        self._last_fetched_fingerprint: str | None = None
        self._ui_queue: queue.Queue = queue.Queue()
        self._last_history_snapshot: list[str] = []
        self._last_timeline_key: str = ""

        saved = self.settings_store.load()
        llm_init = controller.llm_coach.status_snapshot()
        init_url = saved.get("base_url") or llm_init.get("base_url", "")
        init_key = saved.get("api_key", "")
        init_model = saved.get("model") or llm_init.get("model", "")
        init_proto = saved.get("protocol") or llm_init.get("protocol", "responses")
        init_vision = saved.get("vision_mode") or llm_init.get("vision_mode", "text_only")
        init_effort = saved.get("reasoning_effort") or llm_init.get("reasoning_effort", "low")
        init_stream = saved.get("stream_enabled", llm_init.get("stream_enabled", False))
        init_interval = float(saved.get("min_interval_s", llm_init.get("interval_s", 18.0)))

        init_llm_enabled = bool(saved.get("llm_enabled", True))
        self._remember_var = tk.BooleanVar(master=root, value=bool(saved.get("remember_enabled", True)))
        self._topmost_var = tk.BooleanVar(master=root, value=bool(saved.get("topmost", True)))
        self._llm_enabled_var = tk.BooleanVar(master=root, value=init_llm_enabled)
        self._llm_interval_var = tk.StringVar(master=root, value=f"{init_interval:g}")
        self._base_url_var = tk.StringVar(master=root, value=init_url)
        self._api_key_var = tk.StringVar(master=root, value=init_key)
        self._show_key_var = tk.BooleanVar(master=root, value=False)
        self._model_var = tk.StringVar(master=root, value=init_model)
        self._protocol_var = tk.StringVar(master=root, value=init_proto)
        self._vision_mode_var = tk.StringVar(master=root, value=_VISION_CODE_TO_LABEL.get(init_vision, "纯文本极速(不传图)"))
        self._reasoning_effort_var = tk.StringVar(master=root, value=_EFFORT_CODE_TO_LABEL.get(init_effort, "low(极速)"))
        self._stream_var = tk.BooleanVar(master=root, value=init_stream)

        self._initial_conn_feedback = "尚未修改默认连接配置"
        if saved:
            controller.llm_coach.set_interval(float(init_interval))
            if init_url and init_model:
                controller.llm_coach.configure_connection(
                    base_url=init_url,
                    api_key=init_key if init_key else None,
                    model=init_model,
                    protocol=init_proto,
                    vision_mode=init_vision,
                    reasoning_effort=init_effort,
                    stream_enabled=init_stream,
                )
            if init_url and init_key:
                try:
                    self._last_fetched_fingerprint = self._connection_fingerprint(validate_base_url(init_url), init_key)
                except ValueError:
                    pass
            if saved.get("api_key_decrypt_failed"):
                self._initial_conn_feedback = "已恢复 Base URL，但本机旧加密密钥解密失败，请重新填写 API Key"
            elif init_key and init_url:
                self._initial_conn_feedback = "已自动恢复上次记住的 Base URL、API Key 与模型（未自动发起外网请求）"
            elif init_url:
                self._initial_conn_feedback = "已自动恢复上次记住的 Base URL 与模型"

        self._build_ui()
        self.controller.llm_coach.set_enabled(init_llm_enabled)
        self._apply_topmost(persist=False)
        self._position_window(EXPANDED_W, EXPANDED_H)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._schedule_poll()

    def _position_window(self, w: int, h: int) -> None:
        try:
            sw = self.root.winfo_screenwidth()
            x = max(0, sw - w - RIGHT_MARGIN)
            self.root.geometry(f"{w}x{h}+{x}+{TOP_MARGIN}")
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.root.title("PVZagent 只读实时建议助手 v1.2.0")
        self.root.minsize(COLLAPSED_W, COLLAPSED_H)
        try:
            self.root.attributes("-topmost", True)
            self.root.lift()
        except Exception:
            pass

        # Collapsed narrow strip (hidden by default)
        self._collapsed_frame = ttk.Frame(self.root, padding=6)
        ttk.Button(self._collapsed_frame, text="展开\n侧栏", width=5, command=self.toggle_collapse).pack(fill="x", pady=(0, 8))
        self._collapsed_badge = ttk.Label(self._collapsed_frame, text="待机", justify="center", anchor="center")
        self._collapsed_badge.pack(fill="x", pady=(0, 8))
        ttk.Label(self._collapsed_frame, text="只读\n0输入", justify="center", anchor="center").pack(side="bottom")

        # Main expanded container
        self._main_frame = ttk.Frame(self.root, padding=10)
        self._main_frame.pack(fill="both", expand=True)

        # 1. Top tab + collapse bar
        nav = ttk.Frame(self._main_frame)
        nav.pack(fill="x", pady=(0, 6))
        self._btn_advice = ttk.Button(nav, text="建议", width=6, command=lambda: self.show_tab("advice"))
        self._btn_timeline = ttk.Button(nav, text="时间轴", width=7, command=lambda: self.show_tab("timeline"))
        self._btn_settings = ttk.Button(nav, text="设置", width=6, command=lambda: self.show_tab("settings"))
        self._btn_collapse = ttk.Button(nav, text="折叠", width=6, command=self.toggle_collapse)
        self._btn_advice.pack(side="left")
        self._btn_timeline.pack(side="left", padx=(4, 0))
        self._btn_settings.pack(side="left", padx=(4, 0))
        self._btn_collapse.pack(side="right")
        ttk.Label(nav, text="只读·0输入", anchor="e").pack(side="right", padx=(0, 6))

        # 2. Control & real-time status strip
        ctrl = ttk.Frame(self._main_frame)
        ctrl.pack(fill="x", pady=(0, 6))
        self._btn_select = ttk.Button(ctrl, text="选窗(5秒)", command=self.begin_select_window)
        self._btn_start = ttk.Button(ctrl, text="开始监测", command=self.start_monitoring)
        self._btn_pause = ttk.Button(ctrl, text="暂停监测", command=self.pause_monitoring)
        self._btn_select.pack(side="left")
        self._btn_start.pack(side="left", padx=(6, 0))
        self._btn_pause.pack(side="left", padx=(6, 0))

        # Fixed-height status box so status text wrapping never shifts content below
        status_box = ttk.Frame(self._main_frame, height=58)
        status_box.pack(fill="x", pady=(0, 4))
        status_box.pack_propagate(False)
        self._status_lbl = ttk.Label(
            status_box,
            text="未选窗 · 点击「选窗(5秒)」后切到原版 PVZ 窗口",
            wraplength=EXPANDED_W - 24,
            justify="left",
            anchor="nw",
        )
        self._status_lbl.pack(fill="x", pady=(0, 2))

        self._llm_status_lbl = ttk.Label(
            status_box,
            text="模型通道：纯本地极速模式（0 外网请求，可在「设置」填写并开启 LLM）",
            wraplength=EXPANDED_W - 24,
            justify="left",
            anchor="nw",
        )
        self._llm_status_lbl.pack(fill="x")
        ttk.Separator(self._main_frame, orient="horizontal").pack(fill="x", pady=(0, 6))

        # 3. Same-sidebar content host
        self._host = ttk.Frame(self._main_frame)
        self._host.pack(fill="both", expand=True)

        # --- Page A: Advice (default) — fixed-height top cards so Advice History NEVER jumps! ---
        self._page_advice = ttk.Frame(self._host)

        phase_box = ttk.Frame(self._page_advice, height=34)
        phase_box.pack(fill="x", pady=(0, 6))
        phase_box.pack_propagate(False)
        self._phase_lbl = ttk.Label(
            phase_box,
            text="当前阶段：阶段待确认（00:00）｜ 重点：持续刷新观测",
            wraplength=EXPANDED_W - 24,
            justify="left",
            anchor="nw",
        )
        self._phase_lbl.pack(fill="both", expand=True)

        ttk.Label(self._page_advice, text="当前建议（固定高度，不引起下方历史跳动）", anchor="w").pack(fill="x")
        self._advice_card = tk.Frame(
            self._page_advice,
            height=154,
            background="#f2f2f2",
            highlightbackground="#c8d4e0",
            highlightthickness=1,
            padx=8,
            pady=6,
        )
        self._advice_card.pack(fill="x", pady=(3, 8))
        self._advice_card.pack_propagate(False)

        self._current_title_lbl = tk.Label(
            self._advice_card,
            text="暂无当前建议",
            font=("Microsoft YaHei UI", 13, "bold"),
            background="#f2f2f2",
            foreground="#1a1a1a",
            wraplength=EXPANDED_W - 44,
            justify="left",
            anchor="w",
        )
        self._current_title_lbl.pack(fill="x")

        self._current_meta_lbl = tk.Label(
            self._advice_card,
            text="当前无有效建议（暂停、切窗或等待新帧）",
            font=("Microsoft YaHei UI", 8),
            background="#f2f2f2",
            foreground="#556677",
            wraplength=EXPANDED_W - 44,
            justify="left",
            anchor="w",
        )
        self._current_meta_lbl.pack(fill="x", pady=(1, 4))

        self._current_reason_lbl = tk.Label(
            self._advice_card,
            text="",
            font=("Microsoft YaHei UI", 9),
            background="#f2f2f2",
            foreground="#222222",
            wraplength=EXPANDED_W - 44,
            justify="left",
            anchor="nw",
        )
        self._current_reason_lbl.pack(fill="x", pady=(0, 3))

        self._llm_advice_lbl = tk.Label(
            self._advice_card,
            text="",
            font=("Microsoft YaHei UI", 9),
            background="#f2f2f2",
            foreground="#0b4f8a",
            wraplength=EXPANDED_W - 44,
            justify="left",
            anchor="nw",
        )
        self._llm_advice_lbl.pack(fill="x")

        ttk.Label(self._page_advice, text="分层真实证据评估（卡片／威胁／位置）", anchor="w").pack(fill="x")
        self._evidence_box = tk.Frame(
            self._page_advice,
            height=112,
            background="#fafafa",
            highlightbackground="#dddddd",
            highlightthickness=1,
            padx=6,
            pady=1,
        )
        self._evidence_box.pack(fill="x", pady=(3, 8))
        self._evidence_box.pack_propagate(False)
        self._evidence_lbl = tk.Label(
            self._evidence_box,
            text="当前无有效对局帧证据（未连接、已暂停或已过期）",
            font=("Microsoft YaHei UI", 9),
            background="#fafafa",
            foreground="#333333",
            wraplength=EXPANDED_W - 40,
            justify="left",
            anchor="nw",
        )
        self._evidence_lbl.pack(fill="both", expand=True)

        ttk.Label(self._page_advice, text="建议历史（旧在上、新在下，重复已合并）", anchor="w").pack(fill="x")
        self._hist_box = ttk.Frame(self._page_advice)
        self._hist_box.pack(fill="both", expand=True, pady=(3, 0))
        self._history_text = tk.Text(self._hist_box, height=10, width=44, wrap="word", state="disabled")
        hist_sb = ttk.Scrollbar(self._hist_box, orient="vertical", command=self._history_text.yview)
        self._history_text.configure(yscrollcommand=hist_sb.set)
        self._history_text.pack(side="left", fill="both", expand=True)
        hist_sb.pack(side="right", fill="y")

        # --- Page B: Timeline ---
        self._page_timeline = ttk.Frame(self._host)
        self._timeline_hdr = ttk.Label(
            self._page_timeline,
            text="本局时间轴（相对时钟 00:00）",
            anchor="w",
            wraplength=EXPANDED_W - 24,
        )
        self._timeline_hdr.pack(fill="x", pady=(0, 6))
        tl_box = ttk.Frame(self._page_timeline)
        tl_box.pack(fill="both", expand=True)
        self._timeline_text = tk.Text(tl_box, height=24, width=44, wrap="word", state="disabled")
        tl_sb = ttk.Scrollbar(tl_box, orient="vertical", command=self._timeline_text.yview)
        self._timeline_text.configure(yscrollcommand=tl_sb.set)
        self._timeline_text.pack(side="left", fill="both", expand=True)
        tl_sb.pack(side="right", fill="y")

        # --- Page C: Settings ---
        self._page_settings = ttk.Frame(self._host)
        ttk.Checkbutton(
            self._page_settings,
            text="侧栏窗口置顶显示（不抢占游戏前景焦点）",
            variable=self._topmost_var,
            command=self._apply_topmost,
        ).pack(anchor="w", pady=(2, 6))

        ttk.Checkbutton(
            self._page_settings,
            text="启用后台异步 LLM 战术建议（默认开启，取消勾选切回纯本地 0 外网）",
            variable=self._llm_enabled_var,
            command=self._apply_llm_toggle,
        ).pack(anchor="w", pady=(0, 6))

        conn_box = ttk.LabelFrame(self._page_settings, text="API 连接设置（仅本机 DPAPI 加密，绝不写入项目）", padding=8)
        conn_box.pack(fill="x", pady=(0, 8))

        ttk.Label(conn_box, text="Base URL（例如 https://api.openai.com/v1）：", anchor="w").pack(fill="x")
        self._base_url_entry = ttk.Entry(conn_box, textvariable=self._base_url_var)
        self._base_url_entry.pack(fill="x", pady=(2, 6))

        key_hdr = ttk.Frame(conn_box)
        key_hdr.pack(fill="x")
        ttk.Label(key_hdr, text="API Key（填完与 URL 后自动拉取模型列表）：").pack(side="left")
        ttk.Checkbutton(
            key_hdr, text="显示", variable=self._show_key_var, command=self._toggle_key_mask
        ).pack(side="right")
        self._api_key_entry = ttk.Entry(conn_box, textvariable=self._api_key_var, show="*")
        self._api_key_entry.pack(fill="x", pady=(2, 6))

        for widget in (self._base_url_entry, self._api_key_entry):
            widget.bind("<FocusOut>", lambda _e: self._maybe_trigger_auto_fetch(immediate=True))
            widget.bind("<Return>", lambda _e: self._maybe_trigger_auto_fetch(immediate=True))
        self._base_url_var.trace_add("write", lambda *_a: self._maybe_trigger_auto_fetch(immediate=False))
        self._api_key_var.trace_add("write", lambda *_a: self._maybe_trigger_auto_fetch(immediate=False))

        mp_row = ttk.Frame(conn_box)
        mp_row.pack(fill="x", pady=(0, 6))
        ttk.Label(mp_row, text="模型：").pack(side="left")
        init_model = self._model_var.get().strip()
        self._model_combo = ttk.Combobox(
            mp_row,
            textvariable=self._model_var,
            values=(init_model,) if init_model else (),
            width=18,
            state="normal",
        )
        self._model_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_or_proto_changed())
        self._model_combo.bind("<FocusOut>", lambda _e: self._on_model_or_proto_changed())
        ttk.Label(mp_row, text="协议：").pack(side="left")
        self._proto_combo = ttk.Combobox(
            mp_row, textvariable=self._protocol_var, values=SUPPORTED_PROTOCOLS, width=19, state="readonly"
        )
        self._proto_combo.pack(side="left")
        self._proto_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_or_proto_changed())

        int_row = ttk.Frame(conn_box)
        int_row.pack(fill="x", pady=(0, 4))
        ttk.Label(int_row, text="间隔(秒)：").pack(side="left")
        combo = ttk.Combobox(
            int_row, textvariable=self._llm_interval_var,
            values=("3", "4", "6", "10", "12", "15", "18", "25", "30"), width=4, state="readonly"
        )
        combo.pack(side="left", padx=(0, 6))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_llm_interval())
        ttk.Label(int_row, text="推理强度：").pack(side="left")
        self._effort_combo = ttk.Combobox(
            int_row,
            textvariable=self._reasoning_effort_var,
            values=tuple(_EFFORT_CODE_TO_LABEL[code] for code in SUPPORTED_REASONING_EFFORTS),
            width=12,
            state="readonly",
        )
        self._effort_combo.pack(side="left", fill="x", expand=True)
        self._effort_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_or_proto_changed())

        vis_row = ttk.Frame(conn_box)
        vis_row.pack(fill="x", pady=(0, 4))
        ttk.Label(vis_row, text="请求模式：").pack(side="left")
        self._vision_combo = ttk.Combobox(
            vis_row,
            textvariable=self._vision_mode_var,
            values=tuple(_VISION_LABEL_TO_CODE.keys()),
            width=22,
            state="readonly",
        )
        self._vision_combo.pack(side="left", fill="x", expand=True)
        self._vision_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_or_proto_changed())
        ttk.Checkbutton(
            vis_row, text="流式接收", variable=self._stream_var, command=self._on_model_or_proto_changed,
        ).pack(side="left", padx=(6, 0))

        ttk.Checkbutton(
            conn_box,
            text="记住 Base URL / API Key / 模型（本机 DPAPI 加密，不写项目）",
            variable=self._remember_var,
            command=self._on_remember_toggle,
        ).pack(anchor="w", pady=(0, 4))

        btn_row = ttk.Frame(conn_box)
        btn_row.pack(fill="x", pady=(2, 4))
        ttk.Button(btn_row, text="应用设置", command=self._apply_connection_settings).pack(side="left")
        ttk.Button(btn_row, text="拉取模型", command=self._manual_fetch_models).pack(side="left", padx=(5, 0))
        ttk.Button(btn_row, text="测试连通", command=self._test_connection).pack(side="left", padx=(5, 0))
        ttk.Button(btn_row, text="清空密钥", command=self._clear_manual_key).pack(side="left", padx=(5, 0))

        self._conn_feedback_lbl = ttk.Label(
            conn_box, text=self._initial_conn_feedback, wraplength=EXPANDED_W - 44, justify="left", anchor="w"
        )
        self._conn_feedback_lbl.pack(fill="x", pady=(2, 2))

        self._api_key_lbl = ttk.Label(conn_box, text="", wraplength=EXPANDED_W - 44, justify="left", anchor="w")
        self._api_key_lbl.pack(fill="x")

        ttk.Button(
            self._page_settings,
            text="手动切新局时间轴（重置 00:00 与 run_id）",
            command=self._manual_new_run,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(
            self._page_settings,
            text=(
                "安全与边界声明：\n"
                "• 纯只读伴随侧栏：0 游戏点击/按键输入，STOP 锁定保留。\n"
                "• 纯内存感知：0 像素图片写入磁盘，退出自动清零缓冲。\n"
                "• API Key 默认经 Windows DPAPI 加密保存在本机用户目录（%LOCALAPPDATA%\\PVZagent），绝不明文落盘或写入项目；取消勾选「记住」或点击「清空密钥」可立即删除。\n"
                "• 独立 2.0s 看门狗：切出游戏、暂停、过期或不确定时立即撤下当前建议并保留历史；旧代结果不得复活。"
            ),
            wraplength=EXPANDED_W - 24,
            justify="left",
            anchor="nw",
        ).pack(fill="both", expand=True)

        self.show_tab("advice")

    def show_tab(self, name: str) -> None:
        self._active_tab = name
        for page in (self._page_advice, self._page_timeline, self._page_settings):
            page.pack_forget()
        target = {
            "advice": self._page_advice,
            "timeline": self._page_timeline,
            "settings": self._page_settings,
        }[name]
        target.pack(fill="both", expand=True)

    def toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._main_frame.pack_forget()
            self._collapsed_frame.pack(fill="both", expand=True)
            self._position_window(COLLAPSED_W, COLLAPSED_H)
        else:
            self._collapsed_frame.pack_forget()
            self._main_frame.pack(fill="both", expand=True)
            self._position_window(EXPANDED_W, EXPANDED_H)

    def _selected_vision_mode(self) -> str:
        return _VISION_LABEL_TO_CODE.get(self._vision_mode_var.get().strip(), "text_only")

    def _selected_reasoning_effort(self) -> str:
        return _EFFORT_LABEL_TO_CODE.get(self._reasoning_effort_var.get().strip(), "low")

    def _persist_settings(self) -> tuple[bool, str]:
        try:
            interval_s = float(self._llm_interval_var.get())
        except Exception:
            interval_s = 18.0
        return self.settings_store.save(
            base_url=self._base_url_var.get(),
            api_key=self._api_key_var.get(),
            model=self._model_var.get(),
            protocol=self._protocol_var.get(),
            min_interval_s=interval_s,
            vision_mode=self._selected_vision_mode(),
            reasoning_effort=self._selected_reasoning_effort(),
            stream_enabled=bool(self._stream_var.get()),
            llm_enabled=bool(self._llm_enabled_var.get()),
            topmost=bool(self._topmost_var.get()),
            remember_enabled=bool(self._remember_var.get()),
        )

    def _on_remember_toggle(self) -> None:
        if not self._remember_var.get():
            self.settings_store.clear_all()
            self._conn_feedback_lbl.configure(text="已取消记忆并清除本机已保存配置（当前仅保留在本次内存）")
            return
        ok, msg = self._persist_settings()
        self._conn_feedback_lbl.configure(text=msg if ok else f"记忆失败：{msg}")

    def _on_model_or_proto_changed(self) -> None:
        if self._closed:
            return
        ok, msg = self.controller.llm_coach.configure_connection(
            base_url=self._base_url_var.get(),
            api_key=self._api_key_var.get() if self._api_key_var.get().strip() else None,
            model=self._model_var.get(),
            protocol=self._protocol_var.get(),
            vision_mode=self._selected_vision_mode(),
            reasoning_effort=self._selected_reasoning_effort(),
            stream_enabled=bool(self._stream_var.get()),
        )
        if ok:
            if self._remember_var.get():
                self._persist_settings()
            self._conn_feedback_lbl.configure(text=msg)

    def _cancel_pending_fetch(self) -> None:
        self._fetch_seq += 1
        if self._fetch_debounce_handle is not None:
            try:
                self.root.after_cancel(self._fetch_debounce_handle)
            except Exception:
                pass
            self._fetch_debounce_handle = None

    def _apply_topmost(self, *, persist: bool = True) -> None:
        try:
            self.root.attributes("-topmost", bool(self._topmost_var.get()))
        except Exception:
            pass
        if persist and hasattr(self, "_conn_feedback_lbl") and self._remember_var.get():
            self._persist_settings()

    def _apply_llm_toggle(self) -> None:
        self.controller.llm_coach.set_enabled(bool(self._llm_enabled_var.get()))
        if self._remember_var.get():
            self._persist_settings()

    def _apply_llm_interval(self) -> None:
        try:
            self.controller.llm_coach.set_interval(float(self._llm_interval_var.get()))
        except Exception:
            pass
        if self._remember_var.get():
            self._persist_settings()

    def _toggle_key_mask(self) -> None:
        self._api_key_entry.configure(show="" if self._show_key_var.get() else "*")

    def _apply_connection_settings(self) -> tuple[bool, str]:
        self._cancel_pending_fetch()
        self._apply_llm_interval()
        ok, msg = self.controller.llm_coach.configure_connection(
            base_url=self._base_url_var.get(),
            api_key=self._api_key_var.get(),
            model=self._model_var.get(),
            protocol=self._protocol_var.get(),
            vision_mode=self._selected_vision_mode(),
            reasoning_effort=self._selected_reasoning_effort(),
            stream_enabled=bool(self._stream_var.get()),
        )
        if ok and self._remember_var.get():
            saved_ok, saved_msg = self._persist_settings()
            msg = f"{msg}（{saved_msg}）" if saved_ok else f"{msg}（提示：{saved_msg}）"
        self._conn_feedback_lbl.configure(text=msg)
        return ok, msg

    def _clear_manual_key(self) -> None:
        self._cancel_pending_fetch()
        self._api_key_var.set("")
        self._last_fetched_fingerprint = None
        self.settings_store.clear_api_key()
        msg = self.controller.llm_coach.clear_manual_api_key()
        self._conn_feedback_lbl.configure(text=f"{msg}（本机已加密保存的密钥同步清除）")

    def _connection_fingerprint(self, clean_url: str, key: str) -> str:
        return hashlib.sha256(f"{clean_url}\0{key}".encode("utf-8")).hexdigest()

    def _maybe_trigger_auto_fetch(self, *, immediate: bool = False) -> None:
        if self._closed:
            return
        if self._fetch_debounce_handle is not None:
            try:
                self.root.after_cancel(self._fetch_debounce_handle)
            except Exception:
                pass
            self._fetch_debounce_handle = None
        if immediate:
            self._run_auto_fetch_if_ready(force=False)
        else:
            self._fetch_debounce_handle = self.root.after(650, lambda: self._run_auto_fetch_if_ready(force=False))

    def _manual_fetch_models(self) -> None:
        self._run_auto_fetch_if_ready(force=True)

    def _run_auto_fetch_if_ready(self, *, force: bool = False) -> bool:
        self._fetch_debounce_handle = None
        if self._closed:
            return False
        raw_url = self._base_url_var.get().strip()
        raw_key = self._api_key_var.get().strip()
        try:
            clean_url = validate_base_url(raw_url)
        except ValueError as exc:
            if force:
                self._conn_feedback_lbl.configure(text=str(exc))
            return False

        # Auto-fetch triggers once both a valid base_url AND a non-empty key (manual or env) are ready
        effective_key = raw_key
        if not effective_key:
            if not force:
                return False
            if not self.controller.llm_coach.has_api_key():
                self._conn_feedback_lbl.configure(text="请先填写 API Key（或配置 PVZ_LLM_API_KEY）后再拉取模型列表")
                return False
            effective_key = "__ENV__"

        fp = self._connection_fingerprint(clean_url, effective_key)
        if not force and fp == self._last_fetched_fingerprint:
            return False

        self._last_fetched_fingerprint = fp
        self._fetch_seq += 1
        seq = self._fetch_seq
        self._conn_feedback_lbl.configure(text="正在自动拉取可用模型列表...")

        def _on_models(ok: bool, models: list[str], text: str) -> None:
            if self._closed:
                return

            def _apply():
                if self._closed or seq != self._fetch_seq:
                    return
                # Verify UI URL and API Key still match the request that completed (never resurrect a cleared/changed key)
                cur_url = self._base_url_var.get().strip().rstrip("/")
                cur_key = self._api_key_var.get().strip()
                if cur_url != clean_url or cur_key != raw_key:
                    return
                display_text = text
                if ok and models:
                    self._model_combo.configure(values=tuple(models))
                    cur = self._model_var.get().strip()
                    if not cur or cur not in models:
                        preferred = next((m for m in ("gpt-6.1-terra", "gpt-6.1-sol") if m in models), models[0])
                        self._model_var.set(preferred)
                    self.controller.llm_coach.configure_connection(
                        base_url=clean_url,
                        api_key=raw_key if raw_key else None,
                        model=self._model_var.get().strip(),
                        protocol=self._protocol_var.get(),
                        vision_mode=self._selected_vision_mode(),
                        reasoning_effort=self._selected_reasoning_effort(),
                        stream_enabled=bool(self._stream_var.get()),
                    )
                    if self._remember_var.get():
                        saved_ok, _ = self._persist_settings()
                        if saved_ok:
                            display_text = f"{display_text}（已自动本机加密记忆）"
                else:
                    self._last_fetched_fingerprint = None
                self._conn_feedback_lbl.configure(text=display_text)

            self._ui_queue.put(_apply)

        self.controller.llm_coach.fetch_models_async(
            base_url=clean_url,
            api_key=raw_key if raw_key else None,
            callback=_on_models,
        )
        return True

    def _test_connection(self) -> None:
        ok, msg = self._apply_connection_settings()
        if not ok:
            return
        self._conn_feedback_lbl.configure(text="正在发送文本连通测试（不传截图）...")

        def _on_done(_ok: bool, text: str) -> None:
            if not self._closed:
                self._ui_queue.put(lambda: None if self._closed else self._conn_feedback_lbl.configure(text=text))

        self.controller.llm_coach.test_connection_async(_on_done)

    def _manual_new_run(self) -> None:
        self.controller.phase_tracker.start_new_run(self.controller.clock(), reason="用户手动切新局")
        self._last_timeline_key = ""

    def begin_select_window(self) -> None:
        if self._selecting:
            return
        self.controller.pause_monitoring()
        self._selecting = True
        self._select_remaining = 5
        self._tick_select()

    def _tick_select(self) -> None:
        if self._closed or not self._selecting:
            return
        if self._select_remaining > 0:
            self._status_lbl.configure(
                text=f"请在 {self._select_remaining} 秒内切到原版 PVZ-Windowed.exe 窗口（勿选 Magpie 放大窗）..."
            )
            self._select_remaining -= 1
            self._select_handle = self.root.after(1000, self._tick_select)
            return
        self._selecting = False
        self._select_handle = None
        ok, msg = self.controller.try_select_foreground_now()
        self._status_lbl.configure(text=msg)
        if ok:
            self.show_tab("advice")

    def start_monitoring(self) -> None:
        ok, msg = self.controller.start_monitoring()
        self._status_lbl.configure(text=msg)
        if ok:
            self.show_tab("advice")

    def pause_monitoring(self) -> None:
        self.controller.pause_monitoring()

    def _schedule_poll(self) -> None:
        if not self._closed:
            self._poll_handle = self.root.after(120, self._poll_ui)

    def _drain_ui_queue(self) -> None:
        while not self._closed:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                pass

    def _poll_ui(self) -> None:
        self._poll_handle = None
        if self._closed:
            return
        try:
            self._drain_ui_queue()
            self.render_snapshot(self.controller.ui_snapshot())
        finally:
            self._schedule_poll()

    def render_snapshot(self, snap: dict[str, Any]) -> None:
        if not self._selecting:
            self._status_lbl.configure(text=snap["status_text"])

        has_lease = snap["has_lease"]
        monitoring = snap["monitoring"]
        self._btn_start.state(["!disabled" if (has_lease and not monitoring) else "disabled"])
        self._btn_pause.state(["!disabled" if monitoring else "disabled"])

        badge = "监测中" if snap["mode_state"] == "monitoring" else "等待" if monitoring else "待机"
        self._collapsed_badge.configure(text=badge)

        phase = snap["phase"]
        self._phase_lbl.configure(
            text=f"当前阶段：{phase['current_phase']}（{phase['current_status']} · {phase['elapsed_text']}）｜ 重点：{phase['current_focus']}"
        )

        has_active = bool(snap.get("has_active_advice"))
        card_bg = "#e8f4ff" if has_active else "#f2f2f2"
        card_border = "#4a90d9" if has_active else "#c8d4e0"
        self._advice_card.configure(background=card_bg, highlightbackground=card_border)
        for lbl in (self._current_title_lbl, self._current_meta_lbl, self._current_reason_lbl, self._llm_advice_lbl):
            lbl.configure(background=card_bg)

        self._current_title_lbl.configure(text=snap["current_advice_title"])
        self._current_meta_lbl.configure(text=snap.get("current_advice_meta", ""))
        self._current_reason_lbl.configure(text=snap["current_reason_text"])
        self._evidence_lbl.configure(text=snap["evidence_text"])

        llm = snap["llm"]
        if not llm["enabled"]:
            self._llm_status_lbl.configure(text="模型通道：纯本地极速模式（0 外网请求，可在「设置」填写并开启 LLM）")
            self._llm_advice_lbl.configure(text="")
        else:
            if llm["inflight"]:
                st_str = f"等待 {llm['model']} 返回（已用 {llm['waiting_s']:.1f}s）"
            elif llm["status"] == "missing_api_key":
                st_str = "缺少 API Key（请在「设置」填写或配置 PVZ_LLM_API_KEY）"
            elif llm["status"] == "verified":
                st_str = f"{llm['model']} 已返回并展示（共 {llm['verified_count']} 条）"
            elif llm["status"] == "verified_stale":
                st_str = f"{llm['model']} 已返回并展示（局势变动，已标可能过时 · 共 {llm['verified_count']} 条）"
            elif llm["status"].startswith("rejected"):
                st_str = f"{llm['model']} 上一条建议因局势变化/过期已撤回（{llm['status']}）"
            elif llm["status"] == "error":
                st_str = f"{llm['model']} 请求异常（{llm['last_error']}），等待下一轮重试"
            else:
                v_tag = _VISION_SHORT.get(llm.get("vision_mode", "text_only"), "纯文本")
                e_tag = llm.get("reasoning_effort", "low")
                st_str = f"{llm['model']} 就绪（{v_tag}·{e_tag}·间隔 {llm['interval_s']}s）"
            self._llm_status_lbl.configure(text=f"模型通道：{st_str}")

            adv = llm["advice"]
            if adv is not None:
                unc = "；".join(adv.uncertainties)
                self._llm_advice_lbl.configure(
                    text=f"[LLM·{adv.model} · {adv.latency_s:.1f}s] {adv.reason}" + (f"\n不确定项：{unc}" if unc else "")
                )
            else:
                self._llm_advice_lbl.configure(text="")

        key_src = llm.get("api_key_source", "missing")
        key_desc = (
            (
                "自定义密钥（已加载 · 本机 DPAPI 加密记忆）"
                if self._remember_var.get()
                else "自定义密钥（已加载到本次内存，未持久化）"
            )
            if key_src == "manual_memory"
            else "本机环境变量 PVZ_LLM_API_KEY（已检测）" if key_src == "local_env"
            else "未设置（请上方填写 API Key 或配置 PVZ_LLM_API_KEY）"
        )
        self._api_key_lbl.configure(text=f"当前生效密钥：{key_desc}")

        # History (old on top, new on bottom)
        hist = snap["rich_history"]
        if hist != self._last_history_snapshot:
            self._history_text.configure(state="normal")
            self._history_text.delete("1.0", "end")
            for line in hist:
                self._history_text.insert("end", line + "\n\n")
            self._history_text.see("end")
            self._history_text.configure(state="disabled")
            self._last_history_snapshot = list(hist)

        # Timeline tab (always refresh elapsed clock in header; rebuild body only when segments change)
        self._timeline_hdr.configure(text=f"本局 ID：{phase['run_id']} ｜ 相对时长：{phase['elapsed_text']}")
        tl_key = repr((phase["run_id"], len(phase["segments"]), phase["segments"][-1] if phase["segments"] else None))
        if tl_key != self._last_timeline_key:
            self._timeline_text.configure(state="normal")
            self._timeline_text.delete("1.0", "end")
            for seg in phase["segments"]:
                evts = "\n  事件：" + "；".join(seg["events"]) if seg["events"] else ""
                block = (
                    f"[{seg['start']} - {seg['end']}] {seg['phase']}（{seg['status']}）\n"
                    f"  重点：{seg['focus']} | 建议条数：{seg['advice_count']}\n"
                    f"  依据：{seg['basis']}{evts}\n\n"
                )
                self._timeline_text.insert("end", block)
            self._timeline_text.see("end")
            self._timeline_text.configure(state="disabled")
            self._last_timeline_key = tl_key

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._selecting = False
        for handle in (self._poll_handle, self._select_handle, self._fetch_debounce_handle):
            if handle is not None:
                try:
                    self.root.after_cancel(handle)
                except Exception:
                    pass
        try:
            self.controller.close()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass
