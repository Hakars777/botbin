from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import requests
import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


# =============================
# SETTINGS
# =============================
SYMBOL = "BTCUSDT"
TF_LIST = ["15m", "4h", "1d"]
DEFAULT_TF = "15m"

BINANCE_REST = "https://fapi.binance.com"

BACKTEST_DAYS = 183  # ~6 months
INITIAL_BALANCE = 1000.0
RISK_PCT = 0.02

# Structure lengths
SWINGS_LEN = 50
INTERNAL_LEN = 5

# Volatility parsing
ATR_LEN = 200
HIGH_VOL_MULT = 2.0

# FVG (4h, auto threshold, extend not needed for backtest logic)
FVG_TF = "4h"

BULLISH_LEG = 1
BEARISH_LEG = 0

BULLISH = +1
BEARISH = -1

BOS = "BOS"
CHOCH = "CHoCH"


# =============================
# Helpers
# =============================
def interval_seconds(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    raise ValueError(tf)


def ms_to_s(ms: int) -> float:
    return ms / 1000.0


def s_to_ms(s: float) -> int:
    return int(round(s * 1000.0))


def get_server_time_ms() -> int:
    r = requests.get(f"{BINANCE_REST}/fapi/v1/time", timeout=10)
    r.raise_for_status()
    return int(r.json()["serverTime"])


def align_down(ms: int, step_ms: int) -> int:
    return (ms // step_ms) * step_ms


def true_range(h, l, prev_c):
    return np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = true_range(high, low, prev_c)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1.0 / float(n)
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]
    return atr


def parsed_hilo(high: np.ndarray, low: np.ndarray, atr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hv = (high - low) >= (HIGH_VOL_MULT * atr)
    pH = np.where(hv, low, high)
    pL = np.where(hv, high, low)
    return pH, pL


def leg_at(i: int, size: int, high: np.ndarray, low: np.ndarray, prev_leg: int) -> int:
    if i < size:
        return prev_leg
    cand_h = high[i - size]
    cand_l = low[i - size]
    win_h = np.max(high[i - size + 1:i + 1])
    win_l = np.min(low[i - size + 1:i + 1])
    if cand_h > win_h:
        return BEARISH_LEG
    if cand_l < win_l:
        return BULLISH_LEG
    return prev_leg


def crossover(prev: float, cur: float, level: float) -> bool:
    return prev <= level and cur > level


def crossunder(prev: float, cur: float, level: float) -> bool:
    return prev >= level and cur < level


def ranges_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    lo = max(min(a_lo, a_hi), min(b_lo, b_hi))
    hi = min(max(a_lo, a_hi), max(b_lo, b_hi))
    return hi >= lo


# =============================
# REST: klines paging by time-range
# =============================
def fetch_klines_range(symbol: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    step_ms = interval_seconds(tf) * 1000
    limit = 1500
    out = []
    cur = start_ms

    while cur <= end_ms:
        params = {
            "symbol": symbol,
            "interval": tf,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit
        }
        r = requests.get(f"{BINANCE_REST}/fapi/v1/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)
        last_open_ms = int(data[-1][0])
        nxt = last_open_ms + step_ms
        if nxt <= cur:
            break
        cur = nxt

        if len(data) < limit:
            break

        time.sleep(0.06)

    if not out:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "taker_buy_base"])

    dedup: Dict[int, list] = {}
    for k in out:
        dedup[int(k[0])] = k
    keys = sorted(dedup.keys())
    rows = [dedup[t] for t in keys]

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["time"] = df["open_time"].astype(np.int64).apply(ms_to_s).astype(float)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[c] = df[c].astype(float)

    return df[["time", "open", "high", "low", "close", "volume", "taker_buy_base"]].reset_index(drop=True)


# =============================
# Data structures
# =============================
@dataclass
class Pivot:
    currentLevel: float | None = None
    lastLevel: float | None = None
    crossed: bool = False
    barTime: float | None = None
    barIndex: int | None = None


@dataclass
class Trend:
    bias: int = 0  # +1 bull, -1 bear


@dataclass
class Trailing:
    top: float | None = None
    bottom: float | None = None
    lastTopTime: float | None = None
    lastBottomTime: float | None = None


@dataclass
class InternalOB:
    ob_id: int
    lo: float
    hi: float
    origin_time: float  # time of OB candle
    bias: int
    created_idx: int     # bar index when OB appears (breakout bar)
    destroyed_idx: int | None = None


@dataclass
class FVGActive:
    lo: float
    hi: float
    bias: int
    left_time: float


@dataclass
class SwingEvent:
    tag: str
    direction: int  # +1 UP, -1 DOWN
    level: float
    pivot_idx: int
    breakout_idx: int
    t0: float
    t1: float


@dataclass
class BOSZone:
    bias: int
    zone_lo: float
    zone_hi: float
    pivot_idx: int
    breakout_idx: int
    cvd_ok: bool
    used: bool = False


@dataclass
class Trade:
    rule: int
    direction: int
    entry_idx: int
    exit_idx: int
    entry_time: float
    exit_time: float
    entry: float
    exit: float
    stop: float
    tp: float
    qty: float
    pnl: float
    rr: float
    result: str  # "TP"/"SL"


# =============================
# pyqtgraph axis + candles
# =============================
class DateAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                dt = datetime.fromtimestamp(v, tz=timezone.utc)
                if spacing >= 86400:
                    out.append(dt.strftime("%Y-%m-%d"))
                elif spacing >= 3600:
                    out.append(dt.strftime("%m-%d %H:%M"))
                else:
                    out.append(dt.strftime("%H:%M"))
            except Exception:
                out.append("")
        return out


class CandleItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.times = np.array([], dtype=float)
        self.open = np.array([], dtype=float)
        self.high = np.array([], dtype=float)
        self.low = np.array([], dtype=float)
        self.close = np.array([], dtype=float)
        self.w = 1.0
        self.picture = None

    def set_data(self, times, open_, high_, low_, close_, w):
        self.times = np.asarray(times, dtype=float)
        self.open = np.asarray(open_, dtype=float)
        self.high = np.asarray(high_, dtype=float)
        self.low = np.asarray(low_, dtype=float)
        self.close = np.asarray(close_, dtype=float)
        self.w = float(w)
        self._regen()
        self.update()

    def _regen(self):
        pic = QtGui.QPicture()
        p = QtGui.QPainter(pic)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)

        up_pen = QtGui.QPen(QtGui.QColor(0, 255, 85))
        dn_pen = QtGui.QPen(QtGui.QColor(237, 72, 7))
        up_pen.setWidthF(1.2)
        dn_pen.setWidthF(1.2)
        wick_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
        wick_pen.setWidthF(1.0)

        for x, o, hi, lo, cl in zip(self.times, self.open, self.high, self.low, self.close):
            p.setPen(wick_pen)
            p.drawLine(QtCore.QPointF(x, lo), QtCore.QPointF(x, hi))

            up = cl >= o
            p.setPen(up_pen if up else dn_pen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 85) if up else QtGui.QColor(237, 72, 7)))

            top = cl if up else o
            bot = o if up else cl
            height = top - bot
            if abs(height) < 1e-9:
                height = 1e-6
            rect = QtCore.QRectF(x - self.w / 2.0, bot, self.w, height)
            p.drawRect(rect)

        p.end()
        self.picture = pic

    def paint(self, painter, opt, w):
        if self.picture:
            painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if self.times.size == 0:
            return QtCore.QRectF()
        x0 = float(self.times.min() - self.w)
        x1 = float(self.times.max() + self.w)
        y0 = float(self.low.min())
        y1 = float(self.high.max())
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)


# =============================
# Backtest Engine
# =============================
class BacktestEngine:
    def __init__(self, symbol: str, tf: str):
        self.symbol = symbol
        self.tf = tf

    def run(self) -> dict:
        tf_sec = interval_seconds(self.tf)
        tf_ms = tf_sec * 1000

        server_ms = get_server_time_ms()
        end_open_ms = align_down(server_ms, tf_ms)  # open time of current candle
        end_ms = end_open_ms - 1                    # exclude current developing candle

        start_dt = datetime.fromtimestamp(end_open_ms / 1000.0, tz=timezone.utc) - timedelta(days=BACKTEST_DAYS)
        start_ms = align_down(int(start_dt.timestamp() * 1000), tf_ms)

        df = fetch_klines_range(self.symbol, self.tf, start_ms, end_ms)
        if df.empty or len(df) < max(600, SWINGS_LEN * 6):
            return {"ok": False, "error": "Недостаточно данных для бэктеста.", "tf": self.tf}

        # 4h for FVG (extra history)
        htf_ms = interval_seconds(FVG_TF) * 1000
        start4_ms = start_ms - 14 * 86400 * 1000
        start4_ms = align_down(start4_ms, htf_ms)
        df4h = fetch_klines_range(self.symbol, "4h", start4_ms, end_ms)
        if df4h.empty or len(df4h) < 5:
            return {"ok": False, "error": "Недостаточно 4h данных для FVG.", "tf": self.tf}

        # arrays
        t = df["time"].to_numpy(dtype=float)
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)
        tb = df["taker_buy_base"].to_numpy(dtype=float)

        n = len(df)

        # "real" CVD using Binance taker-buy base volume from kline:
        # delta = taker_buy - taker_sell = tb - (v - tb) = 2*tb - v
        cvd_delta = 2.0 * tb - v
        cvd = np.cumsum(cvd_delta)

        # volatility parsed high/low
        atr = atr_wilder(h, l, c, ATR_LEN)
        pH, pL = parsed_hilo(h, l, atr)

        # legs
        leg_swing_series = np.zeros(n, dtype=int)
        leg_internal_series = np.zeros(n, dtype=int)
        cur_s = 0
        cur_i = 0
        for i in range(n):
            cur_s = leg_at(i, SWINGS_LEN, h, l, cur_s)
            cur_i = leg_at(i, INTERNAL_LEN, h, l, cur_i)
            leg_swing_series[i] = cur_s
            leg_internal_series[i] = cur_i

        # pivots/trend/trailing
        swingHigh = Pivot()
        swingLow = Pivot()
        internalHigh = Pivot()
        internalLow = Pivot()
        swingTrend = Trend(0)
        internalTrend = Trend(0)
        trailing = Trailing()

        # FVG engine precompute 4h arrays
        t4 = df4h["time"].to_numpy(dtype=float)
        o4 = df4h["open"].to_numpy(dtype=float)
        h4 = df4h["high"].to_numpy(dtype=float)
        l4 = df4h["low"].to_numpy(dtype=float)
        c4 = df4h["close"].to_numpy(dtype=float)

        active_fvgs: List[FVGActive] = []
        cum_abs = 0.0
        prev_j = -1
        cur_htf_high = None
        cur_htf_low = None

        def fvg_update(i_bar: int):
            nonlocal cum_abs, prev_j, cur_htf_high, cur_htf_low
            ti = float(t[i_bar])
            j = int(np.searchsorted(t4, ti, side="right") - 1)
            if j < 0:
                # still mitigate
                _fvg_mitigate(i_bar)
                return

            new_tf = (j != prev_j)

            if new_tf or cur_htf_high is None:
                cur_htf_high = float(h[i_bar])
                cur_htf_low = float(l[i_bar])
            else:
                hi = float(h[i_bar])
                lo = float(l[i_bar])
                if hi > cur_htf_high:
                    cur_htf_high = hi
                if lo < cur_htf_low:
                    cur_htf_low = lo

            if new_tf and j >= 2:
                lastClose = float(c4[j - 1])
                lastOpen = float(o4[j - 1])
                lastTime = float(t4[j - 1])

                currentHigh = float(cur_htf_high)
                currentLow = float(cur_htf_low)

                last2High = float(h4[j - 2])
                last2Low = float(l4[j - 2])

                denom = (lastOpen * 100.0)
                barDeltaPercent = ((lastClose - lastOpen) / denom) if abs(denom) > 1e-12 else 0.0
                cum_abs += abs(barDeltaPercent)
                denom_idx = float(i_bar) if i_bar > 0 else 1.0
                threshold = (cum_abs / denom_idx) * 2.0

                bullish = (currentLow > last2High) and (lastClose > last2High) and (barDeltaPercent > threshold)
                bearish = (currentHigh < last2Low) and (lastClose < last2Low) and ((-barDeltaPercent) > threshold)

                if bullish:
                    lo_ = float(min(last2High, currentLow))
                    hi_ = float(max(last2High, currentLow))
                    active_fvgs.insert(0, FVGActive(lo=lo_, hi=hi_, bias=BULLISH, left_time=lastTime))
                if bearish:
                    lo_ = float(min(currentHigh, last2Low))
                    hi_ = float(max(currentHigh, last2Low))
                    active_fvgs.insert(0, FVGActive(lo=lo_, hi=hi_, bias=BEARISH, left_time=lastTime))

                if len(active_fvgs) > 300:
                    del active_fvgs[300:]

            _fvg_mitigate(i_bar)
            prev_j = j

        def _fvg_mitigate(i_bar: int):
            cur_low = float(l[i_bar])
            cur_high = float(h[i_bar])
            for k in range(len(active_fvgs) - 1, -1, -1):
                g = active_fvgs[k]
                if (g.bias == BULLISH and cur_low < g.lo) or (g.bias == BEARISH and cur_high > g.hi):
                    active_fvgs.pop(k)

        # internal OBs
        active_obs: List[InternalOB] = []
        ob_id_seq = 1

        last_destroyed_pending = False
        last_destroyed_ob: Optional[InternalOB] = None

        awaiting_bos: List[InternalOB] = []
        bos_zones: List[BOSZone] = []
        swing_events: List[SwingEvent] = []
        swing_bias_series = np.zeros(n, dtype=int)

        # backtest state
        equity = float(INITIAL_BALANCE)
        eq_times: List[float] = []
        eq_values: List[float] = []

        pending_entry: Optional[dict] = None
        position: Optional[dict] = None
        trades: List[Trade] = []

        rule_counts = {1: 0, 2: 0, 3: 0}

        def pivot_update_on_leg_change(i_bar: int, size: int, is_internal: bool):
            if i_bar <= 0:
                return
            pivot_i = i_bar - size
            if pivot_i < 0:
                return

            if is_internal:
                prev_leg = int(leg_internal_series[i_bar - 1])
                cur_leg = int(leg_internal_series[i_bar])
                ch = cur_leg - prev_leg
                if ch == 0:
                    return
                if ch == +1:
                    p = internalLow
                    p.lastLevel = p.currentLevel
                    p.currentLevel = float(l[pivot_i])
                    p.crossed = False
                    p.barTime = float(t[pivot_i])
                    p.barIndex = int(pivot_i)
                elif ch == -1:
                    p = internalHigh
                    p.lastLevel = p.currentLevel
                    p.currentLevel = float(h[pivot_i])
                    p.crossed = False
                    p.barTime = float(t[pivot_i])
                    p.barIndex = int(pivot_i)
            else:
                prev_leg = int(leg_swing_series[i_bar - 1])
                cur_leg = int(leg_swing_series[i_bar])
                ch = cur_leg - prev_leg
                if ch == 0:
                    return
                if ch == +1:
                    p = swingLow
                    p.lastLevel = p.currentLevel
                    p.currentLevel = float(l[pivot_i])
                    p.crossed = False
                    p.barTime = float(t[pivot_i])
                    p.barIndex = int(pivot_i)

                    trailing.bottom = p.currentLevel
                    trailing.lastBottomTime = p.barTime
                    if trailing.top is None:
                        trailing.top = float(h[pivot_i])
                        trailing.lastTopTime = float(t[pivot_i])

                elif ch == -1:
                    p = swingHigh
                    p.lastLevel = p.currentLevel
                    p.currentLevel = float(h[pivot_i])
                    p.crossed = False
                    p.barTime = float(t[pivot_i])
                    p.barIndex = int(pivot_i)

                    trailing.top = p.currentLevel
                    trailing.lastTopTime = p.barTime
                    if trailing.bottom is None:
                        trailing.bottom = float(l[pivot_i])
                        trailing.lastBottomTime = float(t[pivot_i])

        def update_trailing_extremes(i_bar: int):
            hi = float(h[i_bar])
            lo = float(l[i_bar])

            if trailing.top is None or hi >= trailing.top:
                trailing.top = hi
                trailing.lastTopTime = float(t[i_bar])

            if trailing.bottom is None or lo <= trailing.bottom:
                trailing.bottom = lo
                trailing.lastBottomTime = float(t[i_bar])

        def mitigate_obs(i_bar: int):
            nonlocal last_destroyed_pending, last_destroyed_ob
            curH = float(h[i_bar])
            curL = float(l[i_bar])
            kept = []
            destroyed_here: List[InternalOB] = []

            for ob in active_obs:
                if ob.bias == BEARISH:
                    if curH > ob.hi:
                        ob.destroyed_idx = i_bar
                        destroyed_here.append(ob)
                        continue
                else:
                    if curL < ob.lo:
                        ob.destroyed_idx = i_bar
                        destroyed_here.append(ob)
                        continue
                kept.append(ob)

            active_obs[:] = kept

            if destroyed_here:
                destroyed_here.sort(key=lambda x: x.created_idx, reverse=True)
                last_destroyed_pending = True
                last_destroyed_ob = destroyed_here[0]

        def store_internal_ob(pivot: Pivot, bias: int, cur_bar_i: int) -> Optional[InternalOB]:
            nonlocal ob_id_seq
            if pivot.barIndex is None:
                return None
            a = int(pivot.barIndex)
            b = int(cur_bar_i)  # end exclusive like Pine
            if b <= a + 1 or a < 0 or b > n:
                return None

            if bias == BEARISH:
                seg = pH[a:b]
                if seg.size == 0:
                    return None
                idx = a + int(np.argmax(seg))
            else:
                seg = pL[a:b]
                if seg.size == 0:
                    return None
                idx = a + int(np.argmin(seg))

            lo_ = float(min(pH[idx], pL[idx]))
            hi_ = float(max(pH[idx], pL[idx]))
            ob = InternalOB(
                ob_id=ob_id_seq,
                lo=lo_,
                hi=hi_,
                origin_time=float(t[idx]),
                bias=int(bias),
                created_idx=int(cur_bar_i),
                destroyed_idx=None
            )
            ob_id_seq += 1
            active_obs.insert(0, ob)
            if len(active_obs) > 150:
                active_obs.pop()
            return ob

        def strong_to_weak_ok(trade_dir: int, cur_swing_bias: int) -> bool:
            # В вашем индикаторе:
            # swing_bias == BULLISH => Strong Low -> Weak High (лонг)
            # swing_bias == BEARISH => Strong High -> Weak Low (шорт)
            return (trade_dir == BULLISH and cur_swing_bias == BULLISH) or (trade_dir == BEARISH and cur_swing_bias == BEARISH)

        def pick_tp_candidate(trade_dir: int) -> Optional[float]:
            if trailing.top is None or trailing.bottom is None:
                return None
            # Цель на границе weak/strong:
            # лонг -> верхняя граница (Weak High при bullish bias, иначе будет Strong High, но граница остаётся)
            # шорт -> нижняя граница
            return float(trailing.top) if trade_dir == BULLISH else float(trailing.bottom)

        def compute_tp(entry: float, stop: float, cand_tp: Optional[float], trade_dir: int) -> float:
            risk = abs(entry - stop)
            if risk <= 1e-9:
                return entry

            if trade_dir == BULLISH:
                if cand_tp is None or cand_tp <= entry:
                    return entry + 2.0 * risk
                rr = (cand_tp - entry) / risk
                if rr >= 1.0:
                    return cand_tp
                return entry + 2.0 * risk
            else:
                if cand_tp is None or cand_tp >= entry:
                    return entry - 2.0 * risk
                rr = (entry - cand_tp) / risk
                if rr >= 1.0:
                    return cand_tp
                return entry - 2.0 * risk

        def cvd_dir_ok(trade_dir: int, pivot_idx: int, breakout_idx: int) -> bool:
            if pivot_idx < 0 or breakout_idx < 0 or pivot_idx >= n or breakout_idx >= n:
                return False
            d = float(cvd[breakout_idx] - cvd[pivot_idx])
            if trade_dir == BULLISH:
                return d > 0.0
            return d < 0.0

        def create_zone_from_ob_and_bos(ob: InternalOB, ev: SwingEvent) -> Optional[BOSZone]:
            # зона от границы OB до границы BOS
            if ev.direction == +1 and ob.bias == BULLISH:
                # bullish: from OB top to BOS level
                zlo = float(min(ob.hi, ev.level))
                zhi = float(max(ob.hi, ev.level))
                ok = cvd_dir_ok(BULLISH, ev.pivot_idx, ev.breakout_idx)
                return BOSZone(bias=BULLISH, zone_lo=zlo, zone_hi=zhi,
                               pivot_idx=ev.pivot_idx, breakout_idx=ev.breakout_idx, cvd_ok=ok)
            if ev.direction == -1 and ob.bias == BEARISH:
                # bearish: from BOS level to OB bottom
                zlo = float(min(ev.level, ob.lo))
                zhi = float(max(ev.level, ob.lo))
                ok = cvd_dir_ok(BEARISH, ev.pivot_idx, ev.breakout_idx)
                return BOSZone(bias=BEARISH, zone_lo=zlo, zone_hi=zhi,
                               pivot_idx=ev.pivot_idx, breakout_idx=ev.breakout_idx, cvd_ok=ok)
            return None

        # main loop
        for i_bar in range(n):
            # equity point (after this bar processing)
            # 1) execute pending entry at this bar open
            if pending_entry is not None and pending_entry.get("entry_idx") == i_bar and position is None:
                entry = float(o[i_bar])
                direction = int(pending_entry["direction"])
                stop = float(pending_entry["stop"])
                tp_candidate = pending_entry.get("tp_candidate", None)
                rule_id = int(pending_entry["rule"])

                # validate stop side
                if direction == BULLISH and stop >= entry:
                    pending_entry = None
                elif direction == BEARISH and stop <= entry:
                    pending_entry = None
                else:
                    risk_per_unit = abs(entry - stop)
                    if risk_per_unit <= 1e-9:
                        pending_entry = None
                    else:
                        risk_amount = equity * RISK_PCT
                        qty = float(risk_amount / risk_per_unit)
                        tp = float(compute_tp(entry, stop, tp_candidate, direction))

                        position = {
                            "rule": rule_id,
                            "direction": direction,
                            "entry_idx": i_bar,
                            "entry_time": float(t[i_bar]),
                            "entry": entry,
                            "stop": stop,
                            "tp": tp,
                            "qty": qty,
                            "risk_amount": risk_per_unit * qty,
                        }
                        pending_entry = None

            # 2) check exit within this bar
            if position is not None:
                direction = int(position["direction"])
                entry = float(position["entry"])
                stop = float(position["stop"])
                tp = float(position["tp"])
                qty = float(position["qty"])
                risk_amount = float(position["risk_amount"])

                hi = float(h[i_bar])
                lo = float(l[i_bar])

                exit_px = None
                exit_result = None

                if direction == BULLISH:
                    sl_hit = lo <= stop
                    tp_hit = hi >= tp
                    if sl_hit and tp_hit:
                        exit_px = stop
                        exit_result = "SL"
                    elif sl_hit:
                        exit_px = stop
                        exit_result = "SL"
                    elif tp_hit:
                        exit_px = tp
                        exit_result = "TP"
                else:
                    sl_hit = hi >= stop
                    tp_hit = lo <= tp
                    if sl_hit and tp_hit:
                        exit_px = stop
                        exit_result = "SL"
                    elif sl_hit:
                        exit_px = stop
                        exit_result = "SL"
                    elif tp_hit:
                        exit_px = tp
                        exit_result = "TP"

                if exit_px is not None:
                    exit_px = float(exit_px)
                    if direction == BULLISH:
                        pnl = (exit_px - entry) * qty
                    else:
                        pnl = (entry - exit_px) * qty

                    equity += pnl
                    rr = (pnl / risk_amount) if risk_amount > 1e-12 else 0.0

                    tr = Trade(
                        rule=int(position["rule"]),
                        direction=direction,
                        entry_idx=int(position["entry_idx"]),
                        exit_idx=i_bar,
                        entry_time=float(position["entry_time"]),
                        exit_time=float(t[i_bar]),
                        entry=entry,
                        exit=exit_px,
                        stop=stop,
                        tp=tp,
                        qty=qty,
                        pnl=float(pnl),
                        rr=float(rr),
                        result=str(exit_result),
                    )
                    trades.append(tr)
                    position = None

            # 3) update FVG for current bar (uses current bar range)
            fvg_update(i_bar)

            # 4) pivots update (leg changes)
            pivot_update_on_leg_change(i_bar, SWINGS_LEN, is_internal=False)
            pivot_update_on_leg_change(i_bar, INTERNAL_LEN, is_internal=True)

            # 5) trailing extremes
            if trailing.top is not None and trailing.bottom is not None:
                update_trailing_extremes(i_bar)
            else:
                update_trailing_extremes(i_bar)

            # 6) mitigate internal OBs (destruction)
            if i_bar >= 1:
                mitigate_obs(i_bar)

            # 7) internal structure crosses (create OBs)
            new_obs: List[InternalOB] = []
            prev_close = float(c[i_bar - 1] if i_bar > 0 else c[i_bar])
            cur_close = float(c[i_bar])

            if internalHigh.currentLevel is not None and not internalHigh.crossed:
                if crossover(prev_close, cur_close, float(internalHigh.currentLevel)):
                    internalHigh.crossed = True
                    internalTrend.bias = BULLISH
                    ob = store_internal_ob(internalHigh, BULLISH, i_bar)
                    if ob is not None:
                        new_obs.append(ob)
                        awaiting_bos.append(ob)

            if internalLow.currentLevel is not None and not internalLow.crossed:
                if crossunder(prev_close, cur_close, float(internalLow.currentLevel)):
                    internalLow.crossed = True
                    internalTrend.bias = BEARISH
                    ob = store_internal_ob(internalLow, BEARISH, i_bar)
                    if ob is not None:
                        new_obs.append(ob)
                        awaiting_bos.append(ob)

            # 8) swing structure crosses (BOS/CHOCH) + update swingTrend.bias
            if swingHigh.currentLevel is not None and swingHigh.barIndex is not None and not swingHigh.crossed:
                if i_bar > 0 and crossover(float(c[i_bar - 1]), float(c[i_bar]), float(swingHigh.currentLevel)):
                    tag = CHOCH if swingTrend.bias == BEARISH else BOS
                    swingHigh.crossed = True
                    swingTrend.bias = BULLISH
                    ev = SwingEvent(
                        tag=tag,
                        direction=+1,
                        level=float(swingHigh.currentLevel),
                        pivot_idx=int(swingHigh.barIndex),
                        breakout_idx=i_bar,
                        t0=float(swingHigh.barTime) if swingHigh.barTime is not None else float(t[int(swingHigh.barIndex)]),
                        t1=float(t[i_bar]),
                    )
                    swing_events.append(ev)

            if swingLow.currentLevel is not None and swingLow.barIndex is not None and not swingLow.crossed:
                if i_bar > 0 and crossunder(float(c[i_bar - 1]), float(c[i_bar]), float(swingLow.currentLevel)):
                    tag = CHOCH if swingTrend.bias == BULLISH else BOS
                    swingLow.crossed = True
                    swingTrend.bias = BEARISH
                    ev = SwingEvent(
                        tag=tag,
                        direction=-1,
                        level=float(swingLow.currentLevel),
                        pivot_idx=int(swingLow.barIndex),
                        breakout_idx=i_bar,
                        t0=float(swingLow.barTime) if swingLow.barTime is not None else float(t[int(swingLow.barIndex)]),
                        t1=float(t[i_bar]),
                    )
                    swing_events.append(ev)

            swing_bias_series[i_bar] = int(swingTrend.bias)

            # 9) build BOS zones (rule 3 precondition): "после нового OB произошел BOS и одинаковое направление"
            # Pair each BOS with the most recent awaiting OB of same direction created earlier
            if swing_events:
                last_ev = swing_events[-1]
                if last_ev.tag == BOS:
                    candidates = [ob for ob in awaiting_bos if ob.bias == (BULLISH if last_ev.direction == +1 else BEARISH) and ob.created_idx < last_ev.breakout_idx]
                    if candidates:
                        candidates.sort(key=lambda x: x.created_idx, reverse=True)
                        chosen = candidates[0]
                        # remove chosen from awaiting list
                        for k in range(len(awaiting_bos) - 1, -1, -1):
                            if awaiting_bos[k].ob_id == chosen.ob_id:
                                awaiting_bos.pop(k)
                                break
                        z = create_zone_from_ob_and_bos(chosen, last_ev)
                        if z is not None:
                            bos_zones.append(z)
                            if len(bos_zones) > 200:
                                bos_zones = bos_zones[-200:]

            # 10) entry signals (only if flat and no pending entry), execute next bar open
            if position is None and pending_entry is None and i_bar < n - 1 and new_obs:
                cur_swing_bias = int(swing_bias_series[i_bar])

                # in case of multiple new OBs in one bar, process in creation order (already)
                for ob in new_obs:
                    trade_dir = int(ob.bias)
                    ob_lo = float(ob.lo)
                    ob_hi = float(ob.hi)

                    # rule 3 (priority): new OB intersects BOS-zone, same direction, strong->weak filter, CVD ok in pivot->breakout
                    hit_rule3 = False
                    for z in bos_zones:
                        if z.used:
                            continue
                        if z.bias != trade_dir:
                            continue
                        if z.breakout_idx >= i_bar:
                            continue
                        if not z.cvd_ok:
                            continue
                        if not ranges_overlap(ob_lo, ob_hi, z.zone_lo, z.zone_hi):
                            continue
                        if not strong_to_weak_ok(trade_dir, cur_swing_bias):
                            continue

                        stop = ob_lo if trade_dir == BULLISH else ob_hi
                        tp_candidate = pick_tp_candidate(trade_dir)
                        pending_entry = {
                            "entry_idx": i_bar + 1,
                            "direction": trade_dir,
                            "stop": stop,
                            "tp_candidate": tp_candidate,
                            "rule": 3,
                        }
                        z.used = True
                        rule_counts[3] += 1
                        hit_rule3 = True
                        break

                    if hit_rule3:
                        break

                    # rule 2: new OB intersects any active FVG by price range and same direction + strong->weak filter
                    hit_rule2 = False
                    if strong_to_weak_ok(trade_dir, cur_swing_bias):
                        for g in active_fvgs:
                            if g.bias != trade_dir:
                                continue
                            if ranges_overlap(ob_lo, ob_hi, g.lo, g.hi):
                                stop = ob_lo if trade_dir == BULLISH else ob_hi
                                tp_candidate = pick_tp_candidate(trade_dir)
                                pending_entry = {
                                    "entry_idx": i_bar + 1,
                                    "direction": trade_dir,
                                    "stop": stop,
                                    "tp_candidate": tp_candidate,
                                    "rule": 2,
                                }
                                rule_counts[2] += 1
                                hit_rule2 = True
                                break
                    if hit_rule2:
                        break

                    # rule 1: last OB destroyed, next OB overlaps price range and opposite direction -> trade by new OB direction
                    if last_destroyed_pending and last_destroyed_ob is not None:
                        # this is the "next" OB after destruction (first seen)
                        destroyed = last_destroyed_ob
                        last_destroyed_pending = False

                        if destroyed.bias != trade_dir and ranges_overlap(destroyed.lo, destroyed.hi, ob_lo, ob_hi):
                            stop = ob_lo if trade_dir == BULLISH else ob_hi
                            tp_candidate = pick_tp_candidate(trade_dir)
                            pending_entry = {
                                "entry_idx": i_bar + 1,
                                "direction": trade_dir,
                                "stop": stop,
                                "tp_candidate": tp_candidate,
                                "rule": 1,
                            }
                            rule_counts[1] += 1
                            break

                # if destruction pending but no OB created for a while, it stays pending until the next OB appears;
                # we already clear it on first OB after destruction (match or not).

            # 11) equity curve point at bar close
            eq_times.append(float(t[i_bar]))
            eq_values.append(float(equity))

        # stats
        total_trades = len(trades)
        wins = sum(1 for tr in trades if tr.pnl > 0)
        winrate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_rr = (float(np.mean([tr.rr for tr in trades])) if total_trades > 0 else 0.0)

        eq_arr = np.array(eq_values, dtype=float) if eq_values else np.array([INITIAL_BALANCE], dtype=float)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr / np.maximum(peak, 1e-12)) - 1.0
        max_dd = float(np.min(dd) * 100.0) if dd.size else 0.0

        per_rule = {
            1: sum(1 for tr in trades if tr.rule == 1),
            2: sum(1 for tr in trades if tr.rule == 2),
            3: sum(1 for tr in trades if tr.rule == 3),
        }

        return {
            "ok": True,
            "tf": self.tf,
            "df": df,
            "eq_times": np.array(eq_times, dtype=float),
            "eq_values": np.array(eq_values, dtype=float),
            "trades": trades,
            "stats": {
                "balance_start": float(INITIAL_BALANCE),
                "balance_end": float(equity),
                "total_trades": int(total_trades),
                "winrate": float(winrate),
                "avg_rr": float(avg_rr),
                "max_dd_pct": float(max_dd),
                "per_rule": per_rule,
            }
        }


# =============================
# Worker thread
# =============================
class BacktestWorker(QtCore.QThread):
    resultReady = QtCore.Signal(object)

    def __init__(self, symbol: str, tf: str):
        super().__init__()
        self.symbol = symbol
        self.tf = tf

    def run(self):
        try:
            eng = BacktestEngine(self.symbol, self.tf)
            res = eng.run()
            self.resultReady.emit(res)
        except Exception as e:
            self.resultReady.emit({"ok": False, "error": str(e), "tf": self.tf})


# =============================
# UI
# =============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest OB/FVG/BOS + CVD (pyqtgraph)")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.tf_box = QtWidgets.QComboBox()
        self.tf_box.addItems(TF_LIST)
        self.tf_box.setCurrentText(DEFAULT_TF)

        self.run_btn = QtWidgets.QPushButton("Запустить бэктест")
        self.run_btn.clicked.connect(self.start_backtest)

        self.status = QtWidgets.QLabel("")
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        top.addWidget(QtWidgets.QLabel("Timeframe:"))
        top.addWidget(self.tf_box)
        top.addWidget(self.run_btn)
        top.addWidget(self.status, 1)

        self.stats_lbl = QtWidgets.QLabel("")
        self.stats_lbl.setWordWrap(True)
        layout.addWidget(self.stats_lbl)

        pg.setConfigOptions(antialias=False)
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, 1)

        self.price_plot = self.glw.addPlot(row=0, col=0, axisItems={"bottom": DateAxis(orientation="bottom")})
        self.eq_plot = self.glw.addPlot(row=1, col=0, axisItems={"bottom": DateAxis(orientation="bottom")})
        self.eq_plot.setXLink(self.price_plot)

        self.apply_style(self.price_plot)
        self.apply_style(self.eq_plot)

        self.candles = CandleItem()
        self.price_plot.addItem(self.candles)

        self.eq_curve = pg.PlotDataItem([], [], pen=pg.mkPen((255, 255, 255, 200), width=2))
        self.eq_plot.addItem(self.eq_curve)

        # crosshair on price plot
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 60)))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((255, 255, 255, 60)))
        self.price_plot.addItem(self.vline, ignoreBounds=True)
        self.price_plot.addItem(self.hline, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        self.trade_items: List[pg.GraphicsObject] = []
        self.worker: Optional[BacktestWorker] = None

        self.start_backtest()

    def apply_style(self, plot: pg.PlotItem):
        # PlotItem не имеет setBackground; фон задаём через GraphicsLayoutWidget (self.glw)
        try:
            self.glw.setBackground((0, 0, 0))
        except Exception:
            pass

        plot.showGrid(x=True, y=True, alpha=0.15)
        for ax in ("left", "bottom"):
            a = plot.getAxis(ax)
            a.setTextPen(pg.mkPen((255, 255, 255, 140)))
            a.setPen(pg.mkPen((255, 255, 255, 60)))
        vb = plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setDefaultPadding(0.02)


    def mouse_moved(self, evt):
        pos = evt[0]
        if self.price_plot.sceneBoundingRect().contains(pos):
            mp = self.price_plot.getViewBox().mapSceneToView(pos)
            self.vline.setPos(mp.x())
            self.hline.setPos(mp.y())

    def clear_trade_items(self):
        for it in self.trade_items:
            try:
                self.price_plot.removeItem(it)
            except Exception:
                pass
        self.trade_items.clear()

    def set_busy(self, busy: bool, text: str = ""):
        self.run_btn.setEnabled(not busy)
        self.tf_box.setEnabled(not busy)
        self.status.setText(text)

    def start_backtest(self):
        tf = self.tf_box.currentText()
        self.set_busy(True, "загрузка данных / расчёт...")
        self.stats_lbl.setText("")
        self.clear_trade_items()

        if self.worker is not None:
            try:
                self.worker.quit()
                self.worker.wait(200)
            except Exception:
                pass
            self.worker = None

        self.worker = BacktestWorker(SYMBOL, tf)
        self.worker.resultReady.connect(self.on_backtest_ready)
        self.worker.start()

    def on_backtest_ready(self, res: dict):
        if not res.get("ok", False):
            self.set_busy(False, "ошибка")
            self.stats_lbl.setText(f"Ошибка: {res.get('error', 'unknown')}")
            return

        tf = res["tf"]
        df: pd.DataFrame = res["df"]
        tf_sec = interval_seconds(tf)
        w = tf_sec * 0.7

        # draw candles
        self.candles.set_data(
            df["time"].to_numpy(dtype=float),
            df["open"].to_numpy(dtype=float),
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            w=w
        )
        self.price_plot.enableAutoRange(axis="xy", enable=True)
        self.price_plot.autoRange(padding=0.02)

        # draw equity
        eq_times = res["eq_times"]
        eq_values = res["eq_values"]
        self.eq_curve.setData(eq_times, eq_values)
        self.eq_plot.enableAutoRange(axis="xy", enable=True)
        self.eq_plot.autoRange(padding=0.05)

        # stats
        st = res["stats"]
        per_rule = st["per_rule"]
        self.stats_lbl.setText(
            f"TF={tf} | Старт={st['balance_start']:.2f} | Конец={st['balance_end']:.2f} | "
            f"Сделок={st['total_trades']} | Winrate={st['winrate']:.2f}% | "
            f"AvgRR={st['avg_rr']:.3f} | MaxDD={st['max_dd_pct']:.2f}% | "
            f"Rule1={per_rule[1]} Rule2={per_rule[2]} Rule3={per_rule[3]}"
        )

        # draw trades
        self.draw_trades(res["trades"])

        self.set_busy(False, f"{SYMBOL} | TF={tf} | 6м | UTC")

    def draw_trades(self, trades: List[Trade]):
        self.clear_trade_items()
        if not trades:
            return

        for tr in trades:
            entry_t = tr.entry_time
            exit_t = tr.exit_time
            entry = tr.entry
            exit_ = tr.exit
            stop = tr.stop
            tp = tr.tp
            rule = tr.rule
            direction = tr.direction

            # entry marker
            if direction == BULLISH:
                sym = "t1"
                brush = pg.mkBrush((0, 255, 85, 220))
                pen = pg.mkPen((0, 255, 85, 220))
            else:
                sym = "t"
                brush = pg.mkBrush((237, 72, 7, 220))
                pen = pg.mkPen((237, 72, 7, 220))

            entry_sc = pg.ScatterPlotItem([entry_t], [entry], symbol=sym, size=10, brush=brush, pen=pen)
            self.price_plot.addItem(entry_sc)
            self.trade_items.append(entry_sc)

            # exit marker
            exit_sc = pg.ScatterPlotItem([exit_t], [exit_], symbol="o", size=8, brush=pg.mkBrush((255, 255, 255, 220)), pen=pg.mkPen((255, 255, 255, 220)))
            self.price_plot.addItem(exit_sc)
            self.trade_items.append(exit_sc)

            # SL/TP lines
            sl_line = pg.PlotDataItem([entry_t, exit_t], [stop, stop], pen=pg.mkPen((255, 0, 0, 170), width=1))
            tp_line = pg.PlotDataItem([entry_t, exit_t], [tp, tp], pen=pg.mkPen((0, 255, 0, 170), width=1))
            self.price_plot.addItem(sl_line)
            self.price_plot.addItem(tp_line)
            self.trade_items.extend([sl_line, tp_line])

            # entry->exit line
            path_line = pg.PlotDataItem([entry_t, exit_t], [entry, exit_], pen=pg.mkPen((255, 255, 255, 120), width=1))
            self.price_plot.addItem(path_line)
            self.trade_items.append(path_line)

            # label (rule + result)
            txt = pg.TextItem(f"R{rule} {tr.result} RR={tr.rr:.2f}", color=(255, 255, 255))
            txt.setPos(exit_t, exit_)
            self.price_plot.addItem(txt)
            self.trade_items.append(txt)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1600, 950)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
