import os, sys, time, math, threading
from queue import Queue, Empty
from datetime import datetime, timezone

import numpy as np

os.environ["KIVY_NO_FILELOG"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "0"

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.slider import Slider
from kivy.graphics import Color, Rectangle, Line, Triangle, Ellipse
from kivy.core.text import Label as CoreLabel
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.utils import platform

from engine import (
    SYMBOL, TF_LIST, DEFAULT_TF, HIST_BARS,
    BULLISH, BEARISH,
    HAS_WS, interval_seconds, fetch_klines, fetch_latest, compute_overlays,
    SignalTracker, KlineWS, empty_data
)

try:
    from plyer import notification as plyer_notify
    HAS_PLYER = True
except ImportError:
    HAS_PLYER = False

SIG_OBF_COLOR = (1.0, 0.84, 0.0)
SIG_BOS_COLOR = (0.0, 0.82, 1.0)
SIG_FVG_COLOR = (1.0, 0.0, 1.0)
SIG_RADIUS = dp(7)


class CandleChart(Widget):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.data = None
        self.overlays = {}
        self.tf_sec = 900
        self.vt0 = 0.0; self.vt1 = 1.0
        self.vp0 = 0.0; self.vp1 = 1.0
        self._touches = {}
        self._pinch_prev = None; self._drag_prev = None
        self.bind(size=self._redraw, pos=self._redraw)

    def set_data(self, data, overlays, tf_sec):
        self.data = data; self.overlays = overlays; self.tf_sec = tf_sec
        self._redraw()

    def auto_range(self, show_last=200):
        if self.data is None or len(self.data["time"]) == 0: return
        t, h, l = self.data["time"], self.data["high"], self.data["low"]
        n = len(t); s = max(0, n - show_last)
        self.vt0 = float(t[s]); self.vt1 = float(t[-1]) + self.tf_sec * 5
        self.vp0 = float(np.min(l[s:])); self.vp1 = float(np.max(h[s:]))
        pad = (self.vp1 - self.vp0) * 0.05
        self.vp0 -= pad; self.vp1 += pad
        self._redraw()

    def t2x(self, t):
        d = self.vt1 - self.vt0
        return self.x + (t - self.vt0) / d * self.width if abs(d) > 1e-9 else self.x

    def p2y(self, p):
        d = self.vp1 - self.vp0
        return self.y + (p - self.vp0) / d * self.height if abs(d) > 1e-9 else self.y

    def x2t(self, x): return self.vt0 + (x - self.x) / max(1, self.width) * (self.vt1 - self.vt0)
    def y2p(self, y): return self.vp0 + (y - self.y) / max(1, self.height) * (self.vp1 - self.vp0)

    def _grid_step(self):
        pr = self.vp1 - self.vp0
        if pr <= 0: return pr
        raw = pr / 8
        mag = 10 ** int(math.floor(math.log10(max(1e-10, raw))))
        for s in [1, 2, 5, 10, 20, 50, 100]:
            if mag * s >= raw: return mag * s
        return raw

    def _redraw(self, *_):
        self.canvas.clear()
        with self.canvas:
            Color(0.06, 0.06, 0.09, 1)
            Rectangle(pos=self.pos, size=self.size)
            if self.data is None or len(self.data["time"]) == 0:
                lbl = CoreLabel(text="Loading...", font_size=sp(18)); lbl.refresh()
                Color(1, 1, 1, 0.5)
                Rectangle(texture=lbl.texture,
                          pos=(self.center_x - lbl.texture.width/2,
                               self.center_y - lbl.texture.height/2),
                          size=lbl.texture.size)
                return
            self._draw_grid()
            self._draw_fvgs()
            self._draw_obs()
            self._draw_candles()
            self._draw_struct()
            self._draw_sw()
            self._draw_signals()
            self._draw_axes()

    def _draw_grid(self):
        Color(0.15, 0.15, 0.2, 0.4)
        step = self._grid_step()
        if step <= 0: return
        p = math.ceil(self.vp0 / step) * step; cnt = 0
        while p < self.vp1 and cnt < 50:
            y = self.p2y(p)
            Line(points=[self.x, y, self.x + self.width, y], width=0.5)
            p += step; cnt += 1

    def _draw_candles(self):
        t, o, h, l, c = (self.data[k] for k in ("time","open","high","low","close"))
        n = len(t)
        if n == 0: return
        i0 = max(0, int(np.searchsorted(t, self.vt0, side="left")) - 1)
        i1 = min(n, int(np.searchsorted(t, self.vt1, side="right")) + 1)
        bw = self.tf_sec * 0.65
        for i in range(i0, i1):
            xc = self.t2x(float(t[i]))
            yh = self.p2y(float(h[i])); yl = self.p2y(float(l[i]))
            yo = self.p2y(float(o[i])); yc = self.p2y(float(c[i]))
            xl = self.t2x(float(t[i]) - bw/2); xr = self.t2x(float(t[i]) + bw/2)
            w = max(1, xr - xl)
            Color(0.55, 0.55, 0.55, 1)
            Line(points=[xc, yl, xc, yh], width=1)
            if c[i] >= o[i]: Color(0.03, 1.0, 0.33, 1)
            else: Color(0.93, 0.28, 0.03, 1)
            bb = min(yo, yc); bh = max(1, abs(yc - yo))
            Rectangle(pos=(xl, bb), size=(w, bh))

    def _draw_struct(self):
        for ev in self.overlays.get("swing_struct", []):
            if ev.get("dir") == "UP": Color(0.03, 0.6, 0.5, 0.8)
            else: Color(0.95, 0.21, 0.27, 0.8)
            x0 = self.t2x(ev["t0"]); x1 = self.t2x(ev["t1"]); y = self.p2y(ev["level"])
            Line(points=[x0, y, x1, y], width=1.5)
            lbl = CoreLabel(text=ev["tag"], font_size=sp(11)); lbl.refresh()
            xm = self.t2x(ev["tmid"])
            Color(1, 1, 1, 0.85)
            Rectangle(texture=lbl.texture,
                      pos=(xm - lbl.texture.width/2, y + dp(2)),
                      size=lbl.texture.size)

    def _draw_obs(self):
        for ob in self.overlays.get("internal_obs", []):
            x0 = self.t2x(float(ob.barTime)); x1 = self.x + self.width
            yl = self.p2y(float(min(ob.barLow, ob.barHigh)))
            yh = self.p2y(float(max(ob.barLow, ob.barHigh)))
            if yh <= yl: continue
            if ob.bias == BULLISH: Color(0.19, 0.47, 0.96, 0.3)
            else: Color(0.97, 0.49, 0.5, 0.3)
            Rectangle(pos=(x0, yl), size=(max(1, x1 - x0), yh - yl))

    def _draw_fvgs(self):
        for g in self.overlays.get("fvgs", []):
            x0 = self.t2x(float(g.leftTime)); x1 = self.t2x(float(g.rightTime))
            yl = self.p2y(float(min(g.top, g.bottom)))
            yh = self.p2y(float(max(g.top, g.bottom)))
            if yh <= yl: continue
            if g.bias == BULLISH: Color(0, 1, 0.41, 0.2)
            else: Color(1, 0, 0.03, 0.2)
            Rectangle(pos=(x0, yl), size=(max(1, x1 - x0), yh - yl))

    def _draw_sw(self):
        sw = self.overlays.get("strongweak")
        if not sw: return
        xr = self.x + self.width
        Color(0.95, 0.21, 0.27, 0.8)
        yt = self.p2y(sw["top"])
        Line(points=[self.t2x(sw["lastTopTime"]), yt, xr, yt], width=1.5)
        lbl = CoreLabel(text=sw["top_text"], font_size=sp(10)); lbl.refresh()
        Color(1, 1, 1, 0.8)
        Rectangle(texture=lbl.texture,
                  pos=(xr - lbl.texture.width - dp(4), yt + dp(2)),
                  size=lbl.texture.size)
        Color(0.03, 0.6, 0.5, 0.8)
        yb = self.p2y(sw["bottom"])
        Line(points=[self.t2x(sw["lastBottomTime"]), yb, xr, yb], width=1.5)
        lbl = CoreLabel(text=sw["bottom_text"], font_size=sp(10)); lbl.refresh()
        Color(1, 1, 1, 0.8)
        Rectangle(texture=lbl.texture,
                  pos=(xr - lbl.texture.width - dp(4), yb + dp(2)),
                  size=lbl.texture.size)

    def _draw_signal_marker(self, x, y, bias, color_rgb, label_text):
        r = float(SIG_RADIUS)
        sx, sy = self.t2x(x), self.p2y(y)

        Color(*color_rgb, 0.22)
        Ellipse(pos=(sx - r*1.3, sy - r*1.3), size=(r*2.6, r*2.6))

        Color(*color_rgb, 0.55)
        Ellipse(pos=(sx - r, sy - r), size=(r*2, r*2))

        Color(0, 0, 0, 0.5)
        Ellipse(pos=(sx - r*0.6, sy - r*0.6), size=(r*1.2, r*1.2))

        Color(*color_rgb, 0.95)
        hr = r * 0.55
        if bias == BULLISH:
            Triangle(points=[sx, sy + hr, sx - hr*0.7, sy - hr*0.4, sx + hr*0.7, sy - hr*0.4])
        else:
            Triangle(points=[sx, sy - hr, sx - hr*0.7, sy + hr*0.4, sx + hr*0.7, sy + hr*0.4])

        lbl = CoreLabel(text=label_text, font_size=sp(8), bold=True); lbl.refresh()
        Color(1, 1, 1, 0.9)
        Rectangle(texture=lbl.texture,
                  pos=(sx - lbl.texture.width/2, sy + r + dp(1)),
                  size=lbl.texture.size)

    def _draw_signals(self):
        tf_shift = self.tf_sec * 0.12
        for s in self.overlays.get("ob_flip_signals", []):
            self._draw_signal_marker(s["x"], s["y"], s["bias"], SIG_OBF_COLOR, "OBF")
        for s in self.overlays.get("bos_range_signals", []):
            self._draw_signal_marker(s["x"] - tf_shift, s["y"], s["bias"], SIG_BOS_COLOR, "BOS")
        for s in self.overlays.get("fvg_ob_signals", []):
            self._draw_signal_marker(s["x"] + tf_shift, s["y"], s["bias"], SIG_FVG_COLOR, "FVG")

    def _draw_axes(self):
        step = self._grid_step()
        if step <= 0: return
        p = math.ceil(self.vp0 / step) * step; cnt = 0
        while p < self.vp1 and cnt < 50:
            y = self.p2y(p)
            lbl = CoreLabel(text=f"{p:.0f}", font_size=sp(9)); lbl.refresh()
            Color(1, 1, 1, 0.5)
            Rectangle(texture=lbl.texture,
                      pos=(self.x + self.width - lbl.texture.width - dp(2),
                           y - lbl.texture.height/2),
                      size=lbl.texture.size)
            p += step; cnt += 1

    # ── touch handling ──
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos): return False
        if hasattr(touch, "is_mouse_scrolling") and touch.is_mouse_scrolling:
            f = 0.93 if touch.button == "scrollup" else 1.07
            ct, cp = self.x2t(touch.pos[0]), self.y2p(touch.pos[1])
            self.vt0 = ct - (ct - self.vt0) * f; self.vt1 = ct + (self.vt1 - ct) * f
            self.vp0 = cp - (cp - self.vp0) * f; self.vp1 = cp + (self.vp1 - cp) * f
            self._redraw(); return True
        touch.grab(self)
        self._touches[touch.uid] = touch.pos
        if len(self._touches) == 2:
            pts = list(self._touches.values())
            self._pinch_prev = math.dist(pts[0], pts[1])
        elif len(self._touches) == 1:
            self._drag_prev = touch.pos
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self: return False
        self._touches[touch.uid] = touch.pos
        if len(self._touches) == 1:
            prev = self._drag_prev
            if prev is None: self._drag_prev = touch.pos; return True
            dx, dy = touch.pos[0] - prev[0], touch.pos[1] - prev[1]
            tr, pr = self.vt1 - self.vt0, self.vp1 - self.vp0
            self.vt0 -= dx / max(1, self.width) * tr; self.vt1 -= dx / max(1, self.width) * tr
            self.vp0 -= dy / max(1, self.height) * pr; self.vp1 -= dy / max(1, self.height) * pr
            self._drag_prev = touch.pos; self._redraw()
        elif len(self._touches) == 2:
            pts = list(self._touches.values())
            d = math.dist(pts[0], pts[1])
            cx, cy = (pts[0][0]+pts[1][0])*0.5, (pts[0][1]+pts[1][1])*0.5
            if self._pinch_prev and self._pinch_prev > 5 and d > 5:
                raw = self._pinch_prev / d; s = 1.0 + (raw - 1.0) * 0.6
                ct, cp = self.x2t(cx), self.y2p(cy)
                self.vt0 = ct - (ct - self.vt0)*s; self.vt1 = ct + (self.vt1 - ct)*s
                self.vp0 = cp - (cp - self.vp0)*s; self.vp1 = cp + (self.vp1 - cp)*s
                self._redraw()
            self._pinch_prev = d
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self: return False
        touch.ungrab(self)
        self._touches.pop(touch.uid, None)
        if not self._touches:
            self._pinch_prev = None; self._drag_prev = None
        elif len(self._touches) == 1:
            self._pinch_prev = None
            rem = list(self._touches.values())
            self._drag_prev = rem[0] if rem else None
        return True


class AllSignalApp(App):
    def build(self):
        self.title = "All Signal"
        Window.clearcolor = (0.06, 0.06, 0.09, 1)
        if platform == "android":
            try:
                from android.permissions import request_permissions, Permission
                request_permissions([Permission.INTERNET, Permission.WAKE_LOCK,
                                     Permission.POST_NOTIFICATIONS,
                                     Permission.FOREGROUND_SERVICE])
            except Exception: pass

        self._is_landscape = Window.width > Window.height
        self.root_box = BoxLayout(orientation="vertical", padding=dp(4), spacing=dp(4))
        self._build_top_bar()
        self._build_replay_bar()
        self.chart = CandleChart()
        self.root_box.add_widget(self.chart)

        self.q = Queue()
        self.ws_tf = None; self.ws_4h = None
        self.data = None; self.data_4h = None
        self.data_full = None; self.data_4h_full = None
        self.tracker = SignalTracker()
        self.cur_tf = DEFAULT_TF
        self._last_ov = None; self._last_chart_t = 0
        self.replay_enabled = False; self.replay_playing = False
        self.replay_index = 0; self._replay_clock = None
        self._rest_busy = False

        threading.Thread(target=self._load, args=(DEFAULT_TF,), daemon=True).start()
        Clock.schedule_interval(self._poll, 0.1)
        Clock.schedule_interval(self._rest_poll_tick, 3.0)
        Window.bind(on_resize=self._on_window_resize)
        if platform == "android":
            Clock.schedule_once(self._start_bg_service, 3)
        return self.root_box

    def _build_top_bar(self):
        is_land = Window.width > Window.height
        bar_h = dp(36) if is_land else dp(44)
        self.top_bar = BoxLayout(size_hint_y=None, height=bar_h, spacing=dp(4))
        self.tf_spin = Spinner(text=DEFAULT_TF, values=TF_LIST,
                               size_hint_x=0.18, font_size=sp(13))
        self.tf_spin.bind(text=self._on_tf)
        self.top_bar.add_widget(self.tf_spin)
        self.status = Label(text="Loading...", halign="right", valign="middle",
                            color=(1,1,1,0.6), font_size=sp(11),
                            size_hint_x=1, shorten=True, shorten_from="left")
        self.status.bind(size=self.status.setter("text_size"))
        self.top_bar.add_widget(self.status)
        fit_btn = Button(text="Fit", size_hint_x=0.12, font_size=sp(13))
        fit_btn.bind(on_press=lambda *_: self.chart.auto_range())
        self.top_bar.add_widget(fit_btn)
        self.root_box.add_widget(self.top_bar)

    def _build_replay_bar(self):
        is_land = Window.width > Window.height
        bar_h = dp(32) if is_land else dp(38)
        self.replay_bar = BoxLayout(size_hint_y=None, height=bar_h, spacing=dp(2))
        self.replay_toggle = ToggleButton(text="Replay", size_hint_x=0.14, font_size=sp(11))
        self.replay_toggle.bind(state=self._on_replay_toggle)
        self.replay_bar.add_widget(self.replay_toggle)
        self.replay_play_btn = Button(text="Play", size_hint_x=0.1, font_size=sp(11), disabled=True)
        self.replay_play_btn.bind(on_press=self._on_replay_play)
        self.replay_bar.add_widget(self.replay_play_btn)
        self.replay_back_btn = Button(text="<<", size_hint_x=0.07, font_size=sp(12), disabled=True)
        self.replay_back_btn.bind(on_press=lambda *_: self._replay_step(-1))
        self.replay_bar.add_widget(self.replay_back_btn)
        self.replay_fwd_btn = Button(text=">>", size_hint_x=0.07, font_size=sp(12), disabled=True)
        self.replay_fwd_btn.bind(on_press=lambda *_: self._replay_step(+1))
        self.replay_bar.add_widget(self.replay_fwd_btn)
        self.replay_speed_spin = Spinner(text="10",
                                         values=[str(x) for x in [1,2,5,10,20,50,100]],
                                         size_hint_x=0.1, font_size=sp(11))
        self.replay_speed_spin.disabled = True
        self.replay_bar.add_widget(self.replay_speed_spin)
        self.replay_slider = Slider(min=0, max=0, value=0, step=1, disabled=True, size_hint_x=0.35)
        self.replay_slider.bind(value=self._on_replay_slider)
        self.replay_bar.add_widget(self.replay_slider)
        self.replay_info = Label(text="", halign="right", valign="middle",
                                 color=(1,1,1,0.6), font_size=sp(10),
                                 size_hint_x=0.17, shorten=True)
        self.replay_info.bind(size=self.replay_info.setter("text_size"))
        self.replay_bar.add_widget(self.replay_info)
        self.root_box.add_widget(self.replay_bar)

    def _on_window_resize(self, window, w, h):
        is_land = w > h
        if is_land != self._is_landscape:
            self._is_landscape = is_land
            self.top_bar.height = dp(36) if is_land else dp(44)
            self.replay_bar.height = dp(32) if is_land else dp(38)

    def _start_bg_service(self, _dt):
        try:
            from jnius import autoclass
            activity = autoclass("org.kivy.android.PythonActivity").mActivity
            ctx = activity.getApplicationContext()
            sname = str(ctx.getPackageName()) + ".ServiceMonitor"
            autoclass(sname).start(activity, "")
        except Exception: pass

    def _on_tf(self, _spin, text):
        if self.replay_enabled:
            self.replay_toggle.state = "normal"
            self._on_replay_toggle(self.replay_toggle, "normal")
        self.cur_tf = text; self.status.text = "Loading..."
        self._stop_ws()
        threading.Thread(target=self._load, args=(text,), daemon=True).start()

    def _load(self, tf):
        try:
            d = fetch_klines(SYMBOL, tf, HIST_BARS)
            d4 = fetch_klines(SYMBOL, "4h", max(1000, HIST_BARS // 3))
            Clock.schedule_once(lambda dt: self._loaded(tf, d, d4))
        except Exception as e:
            Clock.schedule_once(lambda dt: setattr(self.status, "text", f"Error: {e}"))

    def _loaded(self, tf, data, data_4h):
        self.data = data; self.data_4h = data_4h
        self.data_full = {k: v.copy() for k, v in data.items()}
        self.data_4h_full = {k: v.copy() for k, v in data_4h.items()}
        self.cur_tf = tf
        ov = compute_overlays(data, data_4h); self._last_ov = ov
        alerts = self.tracker.check(ov)
        for at, am in alerts: self._notify(at, am)
        self.chart.set_data(data, ov, interval_seconds(tf))
        self.chart.auto_range()
        n = len(data["time"])
        price = f"${data['close'][-1]:.2f}" if n else ""
        self.status.text = f"{SYMBOL} | {tf} | {n} bars {price}"
        self._stop_ws()
        if HAS_WS:
            self.ws_tf = KlineWS(SYMBOL, tf, self.q, "tf"); self.ws_tf.start()
            self.ws_4h = KlineWS(SYMBOL, "4h", self.q, "4h"); self.ws_4h.start()

    def _stop_ws(self):
        for ws in (self.ws_tf, self.ws_4h):
            if ws:
                try: ws.stop()
                except Exception: pass
        self.ws_tf = self.ws_4h = None

    def _notify(self, title, msg):
        if HAS_PLYER:
            try:
                plyer_notify.notify(title=f"Signal: {title}", message=msg,
                                    app_name="All Signal", timeout=10)
            except Exception: pass

    # ── replay ──

    def _on_replay_toggle(self, inst, state):
        enabled = (state == "down")
        self.replay_enabled = enabled; self.replay_playing = False
        if self._replay_clock: self._replay_clock.cancel(); self._replay_clock = None
        self.replay_play_btn.text = "Play"
        if enabled:
            self._stop_ws()
            try:
                while True: self.q.get_nowait()
            except Empty: pass
            self.tf_spin.disabled = True
            for w in (self.replay_play_btn, self.replay_back_btn, self.replay_fwd_btn,
                      self.replay_speed_spin, self.replay_slider):
                w.disabled = False
            n = len(self.data_full["time"]) if self.data_full else 0
            if n <= 0:
                self.replay_slider.max = 0; self.replay_index = 0; return
            self.replay_slider.max = n - 1; self.replay_index = n - 1
            self.replay_slider.value = self.replay_index
            self._apply_replay_view()
        else:
            self.tf_spin.disabled = False
            for w in (self.replay_play_btn, self.replay_back_btn, self.replay_fwd_btn,
                      self.replay_speed_spin, self.replay_slider):
                w.disabled = True
            self.replay_info.text = ""
            self._stop_ws()
            threading.Thread(target=self._load, args=(self.cur_tf,), daemon=True).start()

    def _on_replay_play(self, *_):
        if not self.replay_enabled: return
        self.replay_playing = not self.replay_playing
        if self.replay_playing:
            self.replay_play_btn.text = "Pause"; self._start_replay_timer()
        else:
            self.replay_play_btn.text = "Play"
            if self._replay_clock: self._replay_clock.cancel(); self._replay_clock = None

    def _start_replay_timer(self):
        if self._replay_clock: self._replay_clock.cancel()
        try: spd = max(1, int(self.replay_speed_spin.text))
        except ValueError: spd = 10
        self._replay_clock = Clock.schedule_interval(self._replay_tick, max(0.01, 1.0/spd))

    def _replay_tick(self, _dt):
        if not self.replay_enabled or not self.replay_playing:
            if self._replay_clock: self._replay_clock.cancel(); self._replay_clock = None
            return
        n = len(self.data_full["time"]) if self.data_full else 0
        if n <= 0 or self.replay_index >= n - 1:
            self.replay_playing = False; self.replay_play_btn.text = "Play"
            if self._replay_clock: self._replay_clock.cancel(); self._replay_clock = None
            return
        self._replay_step(+1)

    def _replay_step(self, delta):
        if not self.replay_enabled or self.data_full is None: return
        n = len(self.data_full["time"])
        if n <= 0: return
        new_i = int(np.clip(self.replay_index + delta, 0, n - 1))
        if new_i == self.replay_index: return
        self.replay_index = new_i; self.replay_slider.value = new_i
        self._apply_replay_view()

    def _on_replay_slider(self, inst, value):
        if not self.replay_enabled: return
        idx = int(value)
        if idx == self.replay_index: return
        self.replay_index = idx; self._apply_replay_view()

    def _apply_replay_view(self):
        if self.data_full is None: return
        n = len(self.data_full["time"])
        if n <= 0: return
        i = int(np.clip(self.replay_index, 0, n-1))
        view = {k: v[:i+1].copy() for k, v in self.data_full.items()}
        if len(view["time"]) == 0: return
        t_last = float(view["time"][-1])
        if self.data_4h_full is not None and len(self.data_4h_full["time"]) > 0:
            mask = self.data_4h_full["time"] <= t_last
            v4h = {k: v[mask].copy() for k, v in self.data_4h_full.items()}
        else:
            v4h = self.data_4h_full
        tf_sec = interval_seconds(self.cur_tf)
        ov = compute_overlays(view, v4h)
        self.chart.set_data(view, ov, tf_sec)
        if self.replay_playing:
            show = min(200, i+1); s = max(0, i+1-show)
            self.chart.vt0 = float(view["time"][s])
            self.chart.vt1 = float(view["time"][-1]) + tf_sec*5
            self.chart.vp0 = float(np.min(view["low"][s:]))
            self.chart.vp1 = float(np.max(view["high"][s:]))
            pad = (self.chart.vp1 - self.chart.vp0) * 0.05
            self.chart.vp0 -= pad; self.chart.vp1 += pad
            self.chart._redraw()
        try:
            dt = datetime.fromtimestamp(t_last, tz=timezone.utc)
            ts_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception: ts_str = ""
        self.replay_info.text = f"{i+1}/{n} | {ts_str}"
        self.status.text = f"{SYMBOL} | {self.cur_tf} | REPLAY | {i+1} bars"

    # ── live polling ──

    def _poll(self, _dt):
        if self.data is None or self.replay_enabled: return
        tf_upd = False; tf_closed = False; h4_closed = False
        try:
            while True:
                msg = self.q.get_nowait()
                if msg["type"] != "kline": continue
                tag = msg["tag"]; tt = float(msg["t"])
                vals = [float(msg[k]) for k in ("o","h","l","c","v")]
                closed = bool(msg["closed"])
                tgt = self.data_4h if tag == "4h" else self.data
                if tgt is None or len(tgt["time"]) == 0: continue
                last_t = float(tgt["time"][-1])
                if abs(tt - last_t) < 1e-9:
                    for i, k in enumerate(("open","high","low","close","volume")):
                        tgt[k][-1] = vals[i]
                elif tt > last_t:
                    tgt["time"] = np.append(tgt["time"], tt)
                    for i, k in enumerate(("open","high","low","close","volume")):
                        tgt[k] = np.append(tgt[k], vals[i])
                    mx = 4500 if tag == "4h" else HIST_BARS
                    if len(tgt["time"]) > mx:
                        for k in tgt: tgt[k] = tgt[k][-mx:]
                if tag == "4h": h4_closed = h4_closed or closed
                else: tf_upd = True; tf_closed = tf_closed or closed
        except Empty: pass

        if tf_closed or h4_closed:
            ov = compute_overlays(self.data, self.data_4h); self._last_ov = ov
            alerts = self.tracker.check(ov)
            for at, am in alerts: self._notify(at, am)
            self.chart.set_data(self.data, ov, interval_seconds(self.cur_tf))
            n = len(self.data["time"])
            price = f"${self.data['close'][-1]:.2f}" if n else ""
            self.status.text = f"{SYMBOL} | {self.cur_tf} | {n} bars {price}"
        elif tf_upd:
            now = time.time()
            if now - self._last_chart_t > 0.25:
                if self._last_ov: self.chart.set_data(self.data, self._last_ov, interval_seconds(self.cur_tf))
                else: self.chart._redraw()
                n = len(self.data["time"])
                price = f"${self.data['close'][-1]:.2f}" if n else ""
                self.status.text = f"{SYMBOL} | {self.cur_tf} | {price}"
                self._last_chart_t = now

    def _rest_poll_tick(self, _dt):
        if self.data is None or self.replay_enabled or self._rest_busy: return
        self._rest_busy = True
        threading.Thread(target=self._rest_poll_worker, daemon=True).start()

    def _rest_poll_worker(self):
        try:
            tf = self.cur_tf
            rows = fetch_latest(SYMBOL, tf, 3)
            rows_4h = fetch_latest(SYMBOL, "4h", 3) if tf != "4h" else []
            if rows: Clock.schedule_once(lambda dt: self._apply_rest(rows, rows_4h, tf))
            else: self._rest_busy = False
        except Exception: self._rest_busy = False

    def _apply_rest(self, rows, rows_4h, tf):
        self._rest_busy = False
        if self.replay_enabled or self.data is None or tf != self.cur_tf: return
        changed = False; new_candle = False
        for row in rows:
            tt = row["t"]; vals = [row["o"], row["h"], row["l"], row["c"], row["v"]]
            tgt = self.data
            if tgt is None or len(tgt["time"]) == 0: continue
            last_t = float(tgt["time"][-1])
            if abs(tt - last_t) < 1e-9:
                for i, k in enumerate(("open","high","low","close","volume")):
                    if abs(tgt[k][-1] - vals[i]) > 1e-12: tgt[k][-1] = vals[i]; changed = True
            elif tt > last_t:
                tgt["time"] = np.append(tgt["time"], tt)
                for i, k in enumerate(("open","high","low","close","volume")):
                    tgt[k] = np.append(tgt[k], vals[i])
                if len(tgt["time"]) > HIST_BARS:
                    for k in tgt: tgt[k] = tgt[k][-HIST_BARS:]
                changed = True; new_candle = True
        for row in rows_4h:
            tt = row["t"]; vals = [row["o"], row["h"], row["l"], row["c"], row["v"]]
            tgt = self.data_4h
            if tgt is None or len(tgt["time"]) == 0: continue
            last_t = float(tgt["time"][-1])
            if abs(tt - last_t) < 1e-9:
                for i, k in enumerate(("open","high","low","close","volume")):
                    tgt[k][-1] = vals[i]
            elif tt > last_t:
                tgt["time"] = np.append(tgt["time"], tt)
                for i, k in enumerate(("open","high","low","close","volume")):
                    tgt[k] = np.append(tgt[k], vals[i])
                if len(tgt["time"]) > 4500:
                    for k in tgt: tgt[k] = tgt[k][-4500:]
                new_candle = True
        if not changed and not new_candle: return
        if new_candle:
            ov = compute_overlays(self.data, self.data_4h); self._last_ov = ov
            alerts = self.tracker.check(ov)
            for at, am in alerts: self._notify(at, am)
            self.chart.set_data(self.data, ov, interval_seconds(self.cur_tf))
        elif self._last_ov:
            self.chart.set_data(self.data, self._last_ov, interval_seconds(self.cur_tf))
        else:
            self.chart._redraw()
        n = len(self.data["time"])
        price = f"${self.data['close'][-1]:.2f}" if n else ""
        self.status.text = f"{SYMBOL} | {self.cur_tf} | {n} bars {price}"

    def on_stop(self): self._stop_ws()


if __name__ == "__main__":
    AllSignalApp().run()
