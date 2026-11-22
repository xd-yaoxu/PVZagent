"""Main-thread Tk projection for the optional, in-memory resource sidebar.

The view attaches to host-owned controls. The opt-in host imports Tk and runs
the existing read-only job only when explicitly called.
"""
from threading import current_thread, main_thread


_GOALS = {
    'collect_resources': '关注资源收集',
    'develop_economy': '关注经济发展',
    'hold': '维持当前阵型',
}


def _advice_text(advice):
    if advice.goal == 'defend_lane' and type(advice.focus_row) is int and 1 <= advice.focus_row <= 6:
        return f'关注第 {advice.focus_row} 路防守'
    return _GOALS.get(advice.goal, '建议内容未知')


def _overlap(old, new):
    """Retained entries when a bounded history deque drops its oldest rows."""
    for size in range(min(len(old), len(new)), 0, -1):
        if old[-size:] == new[:size]:
            return size
    return 0


class TkResourceSidebarView:
    """Attach to a host-owned scheduler, current Label and dedicated history Text.

    Use only on the Tk main thread. Call before_show_advice before mapping the
    advice page and close before destroying its controls or replacing its source.
    A source is single-run; create a new view to attach another source.
    """

    refresh_ms = 100

    def __init__(self, scheduler, current_label, history_text, source):
        self._require_main_thread()
        self._scheduler = scheduler
        self._label = current_label
        self._text = history_text
        self._source = source
        self._closed = False
        self._pending = None
        self._history = None
        self._current_text = None
        self._refresh()
        self._schedule()

    @staticmethod
    def _require_main_thread():
        if current_thread() is not main_thread():
            raise RuntimeError('Sidebar widgets must be used on the main thread')

    def _schedule(self):
        if not self._closed:
            ticket = object()
            handle = self._scheduler.after(self.refresh_ms, lambda: self._tick(ticket))
            self._pending = (ticket, handle)

    def _tick(self, ticket):
        self._require_main_thread()
        if self._closed or self._pending is None or self._pending[0] is not ticket:
            return
        self._pending = None
        try:
            self._refresh()
        finally:
            self._schedule()

    def before_show_advice(self):
        """Refresh on return from Settings, before making Advice visible."""
        self._require_main_thread()
        if not self._closed:
            self._refresh()

    def close(self):
        """Detach and clear the controls before the host destroys or reuses them."""
        self._require_main_thread()
        if self._closed:
            return
        self._closed = True
        pending, self._pending = self._pending, None
        self._source = None
        if pending is not None:
            self._scheduler.after_cancel(pending[1])
        self._set_current(None)
        self._set_history(())

    def _refresh(self):
        try:
            snapshot = self._source.read()
            if snapshot.history != self._history:
                self._set_history(snapshot.history)
            current = self._source.read().current
        except Exception:
            self._set_current(None)
            raise
        self._set_current(current)

    def _set_current(self, advice):
        value = '暂无当前建议' if advice is None else _advice_text(advice)
        if value != self._current_text:
            self._label.configure(text=value)
            self._current_text = value

    def _set_history(self, history):
        if history == self._history:
            return
        text = self._text
        old = self._history or ()
        anchor = None
        follow = not old
        if old:
            text.count('1.0', 'end', 'update', 'ypixels')
            follow = text.yview()[1] >= 0.999
            if not follow:
                top = text.index('@0,0')
                for index in range(len(old)):
                    mark = f'entry_{index}'
                    if text.compare(mark, '<=', top):
                        offset = (text.count(mark, top, 'chars') or (0,))[0]
                        anchor = index, offset
                    else:
                        break

        text.configure(state='normal')
        try:
            for index in range(len(old)):
                text.mark_unset(f'entry_{index}')
            text.delete('1.0', 'end')
            for index, advice in enumerate(history):
                mark = f'entry_{index}'
                text.mark_set(mark, 'end-1c')
                text.mark_gravity(mark, 'left')
                text.insert('end', _advice_text(advice) + '\n\n')
        finally:
            text.configure(state='disabled')

        if history:
            text.count('1.0', 'end', 'update', 'ypixels')
            if follow:
                text.see('end')
            elif anchor is not None:
                kept = _overlap(old, history)
                index = anchor[0] - (len(old) - kept)
                if index >= 0:
                    text.yview(f'entry_{index}+{anchor[1]} chars')
                else:
                    text.yview('1.0')
        self._history = history


def _build_window(root, tk, source, close_display):
    root.title('PVZagent · 只读建议')
    root.geometry('420x620-16+60')
    root.minsize(300, 320)
    root.configure(background='#101010')

    tabs = tk.Frame(root, background='#101010')
    tabs.pack(fill='x', padx=14, pady=(14, 6))
    advice = tk.Frame(root, background='#101010')
    settings = tk.Frame(root, background='#101010')
    current = tk.Label(advice, anchor='w', background='#101010', foreground='#fafafa')
    current.pack(fill='x', pady=(0, 10))
    tk.Label(advice, text='历史记录', anchor='w', background='#101010',
             foreground='#e2e2e2').pack(fill='x', pady=(0, 5))
    history = tk.Text(advice, state='disabled', wrap='word', width=1, height=1,
                      background='#191919', foreground='#ededed', relief='flat')
    history.pack(fill='both', expand=True)
    tk.Label(settings, text='侧栏为独立窗口。关闭侧栏只结束显示，本轮只读任务仍会运行至结束。',
             background='#101010', foreground='#ededed', wraplength=260,
             justify='left', anchor='nw').pack(fill='x')
    tk.Button(settings, text='关闭侧栏', command=close_display).pack(anchor='w', pady=(12, 0))

    view = TkResourceSidebarView(root, current, history, source)

    def show_advice():
        view.before_show_advice()
        settings.pack_forget()
        advice.pack(fill='both', expand=True, padx=14, pady=(4, 14))

    def show_settings():
        advice.pack_forget()
        settings.pack(fill='both', expand=True, padx=14, pady=(4, 14))

    tk.Button(tabs, text='建议', command=show_advice).pack(side='left', padx=(0, 8))
    tk.Button(tabs, text='设置', command=show_settings).pack(side='left')
    show_advice()
    return view


def run_with_resource_sidebar(run_readonly, **runtime_args):
    """Host one existing read-only job; closing the window only detaches its view."""
    TkResourceSidebarView._require_main_thread()
    from concurrent.futures import Future
    from threading import Thread
    import tkinter as tk
    from .resource_sidebar import ReadonlyResourceSidebar

    source = ReadonlyResourceSidebar()
    outcome = Future()
    ui_errors = []
    view = None
    pending = None
    closed = False
    started = False

    def run():
        try:
            outcome.set_result(run_readonly(sidebar=source, **runtime_args))
        except BaseException as exc:
            outcome.set_exception(exc)

    worker = Thread(target=run, name='pvz-readonly-sidebar-run', daemon=False)
    class ResourceSidebarWindow(tk.Tk):
        def readprofile(self, base_name, class_name):
            # The opt-in window does not source user or working-directory profiles.
            return None

    root = ResourceSidebarWindow()

    def close_display():
        nonlocal closed, pending
        if closed:
            return
        closed = True
        try:
            if pending is not None:
                root.after_cancel(pending)
                pending = None
        finally:
            try:
                if view is not None:
                    view.close()
            finally:
                root.destroy()

    def check_worker():
        nonlocal pending
        pending = None
        if closed:
            return
        if worker.is_alive():
            pending = root.after(100, check_worker)
        else:
            close_display()

    def tk_callback_error(_kind, exc, traceback):
        ui_errors.append(exc.with_traceback(traceback))
        close_display()

    try:
        root.withdraw()
        view = _build_window(root, tk, source, close_display)
        root.protocol('WM_DELETE_WINDOW', close_display)
        root.report_callback_exception = tk_callback_error
        pending = root.after(100, check_worker)
        root.deiconify()
        worker.start()
        started = True
        root.mainloop()
    except BaseException as exc:
        ui_errors.append(exc)
    finally:
        try:
            close_display()
        except BaseException as exc:
            ui_errors.append(exc)

    if started:
        # No Tk callback waits for the existing bounded read-only run.
        worker.join()
        worker_error = outcome.exception()
        if worker_error is not None:
            ui_errors.append(worker_error)
    if len(ui_errors) == 1:
        raise ui_errors[0]
    if ui_errors:
        raise BaseExceptionGroup('Resource sidebar and read-only run failures', ui_errors)
    return outcome.result()

