ispravlyay vsyu logiku etogo koda, esli posledniy ob razrushaetsya i sledushiy ob posle nego portivopoljniy po napravleniyu i peresikaet staromu po cene to proisxodit sdelka po novomu napravleniyu, ili esli poyavlyayetsya noviy order block kotoriy peresikaet fvg i oni odinakovie po napravlenui to otkrit sdelku


import sys
import time
import calendar
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


# =============================
# CONFIG
# =============================
SYMBOL = "BTCUSDT"
TF_LIST = ["15m", "4h", "1d"]
DEFAULT_TF = "15m"

BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = +1
BEARISH = -1

SWINGS_LEN = 50
INTERNAL_LEN = 5

ATR_LEN = 200
HIGH_VOL_MULT = 2.0

FVG_TF = "4h"
FVG_EXTEND = 3  # extend in LTF bars (Lux style: extend * (time-time[1]))

BINANCE_REST = "https://fapi.binance.com"

START_EQUITY = 1000.0
RISK_PCT = 0.01


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


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def utc_6_months_ago_ms() -> int:
    now = datetime.now(timezone.utc)
    y = now.year
    m = now.month - 6
    if m <= 0:
        y -= 1
        m += 12
    last_day = calendar.monthrange(y, m)[1]
    d = min(now.day, last_day)
    dt = datetime(y, m, d, now.hour, now.minute, now.second, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def s_to_dt_utc(s: float) -> datetime:
    return datetime.fromtimestamp(float(s), tz=timezone.utc)


def zones_intersect(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    a0, a1 = min(a_lo, a_hi), max(a_lo, a_hi)
    b0, b1 = min(b_lo, b_hi), max(b_lo, b_hi)
    return max(a0, b0) <= min(a1, b1)


# =============================
# REST: klines paging by startTime (6 months)
# =============================
def fetch_klines_from(symbol: str, tf: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    sec = interval_seconds(tf)
    limit = 1500
    out = []
    start_time = start_ms

    while True:
        params = {"symbol": symbol, "interval": tf, "limit": limit, "startTime": start_time}
        if end_ms is not None:
            params["endTime"] = end_ms

        r = requests.get(f"{BINANCE_REST}/fapi/v1/klines", params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)
        last_open_ms = int(data[-1][0])
        nxt = last_open_ms + sec * 1000
        if nxt <= start_time:
            break
        start_time = nxt

        if len(data) < limit:
            break

        time.sleep(0.06)

    dedup = {}
    for k in out:
        dedup[int(k[0])] = k
    keys = sorted(dedup.keys())
    rows = [dedup[t] for t in keys]

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    df["time"] = df["open_time"].astype(np.int64).apply(ms_to_s).astype(float)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# =============================
# Lux-like volatility parsing
# =============================
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


# =============================
# Lux-like LEG + pivots
# =============================
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
class OrderBlock:
    barHigh: float
    barLow: float
    barTime: float
    bias: int  # +1 bull, -1 bear


@dataclass
class FVG:
    top: float
    bottom: float
    bias: int  # +1 bull, -1 bear
    leftTime: float
    rightTime: float  # for drawing only


# =============================
# Backtest position
# =============================
@dataclass
class Position:
    direction: int
    entry: float
    qty: float
    stop: float
    take: float
    entry_time: float
    reason: str


def compute_tp_1to2(entry: float, stop: float, direction: int) -> float | None:
    if direction == BULLISH:
        risk = entry - stop
        if risk <= 0:
            return None
        return float(entry + 2.0 * risk)
    else:
        risk = stop - entry
        if risk <= 0:
            return None
        return float(entry - 2.0 * risk)


# =============================
# BACKTEST ENGINE (fixed per your two entry rules)
# =============================
def run_backtest(df: pd.DataFrame, df4h: pd.DataFrame):
    t = df["time"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    n = len(df)
    if n < max(800, SWINGS_LEN * 5):
        raise RuntimeError("Недостаточно баров для структуры")

    # parsed arrays
    atr = atr_wilder(h, l, c, ATR_LEN)
    pH, pL = parsed_hilo(h, l, atr)

    # legs
    leg_swing = np.zeros(n, dtype=int)
    leg_internal = np.zeros(n, dtype=int)
    cur_s = 0
    cur_i = 0
    for i in range(n):
        cur_s = leg_at(i, SWINGS_LEN, h, l, cur_s)
        cur_i = leg_at(i, INTERNAL_LEN, h, l, cur_i)
        leg_swing[i] = cur_s
        leg_internal[i] = cur_i

    # 4h arrays for FVG mapping
    t4 = df4h["time"].to_numpy(dtype=float)
    o4 = df4h["open"].to_numpy(dtype=float)
    h4 = df4h["high"].to_numpy(dtype=float)
    l4 = df4h["low"].to_numpy(dtype=float)
    c4 = df4h["close"].to_numpy(dtype=float)

    # state
    swingHigh = Pivot()
    swingLow = Pivot()
    internalHigh = Pivot()
    internalLow = Pivot()

    swingTrend = Trend(0)
    internalTrend = Trend(0)

    trailing = Trailing()

    internalOrderBlocks: list[OrderBlock] = []
    active_fvgs: list[FVG] = []

    # L1 replacement state
    waiting_replacement = False
    destroyed_ob: OrderBlock | None = None

    # FVG threshold state (Lux style)
    cum_abs = 0.0
    prev_4h_idx = -10
    ltf_sec = interval_seconds(DEFAULT_TF)  # will be overwritten per TF in wrapper via arg if needed

    # bookkeeping
    equity = START_EQUITY
    equity_curve = np.full(n, np.nan, dtype=float)
    pos: Position | None = None
    trades: list[dict] = []

    l1_count = 0
    l2_count = 0

    def update_trailing_extremes(i: int):
        if trailing.top is None:
            trailing.top = float(h[i])
            trailing.lastTopTime = float(t[i])
        else:
            if float(h[i]) >= trailing.top:
                trailing.top = float(h[i])
                trailing.lastTopTime = float(t[i])

        if trailing.bottom is None:
            trailing.bottom = float(l[i])
            trailing.lastBottomTime = float(t[i])
        else:
            if float(l[i]) <= trailing.bottom:
                trailing.bottom = float(l[i])
                trailing.lastBottomTime = float(t[i])

    def pivot_update(i: int, size: int, is_internal: bool):
        if i <= 0:
            return
        pivot_i = i - size
        if pivot_i < 0:
            return

        if is_internal:
            ch = leg_internal[i] - leg_internal[i - 1]
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
            ch = leg_swing[i] - leg_swing[i - 1]
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

    def store_internal_ob(pivot: Pivot, bias: int, cur_i: int) -> OrderBlock | None:
        if pivot.barIndex is None:
            return None
        a = int(pivot.barIndex)
        b = int(cur_i)  # Pine slice end is exclusive
        if b <= a + 1:
            return None
        if a < 0 or b > n:
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

        ob = OrderBlock(barHigh=float(pH[idx]), barLow=float(pL[idx]), barTime=float(t[idx]), bias=bias)
        internalOrderBlocks.insert(0, ob)
        if len(internalOrderBlocks) > 200:
            internalOrderBlocks.pop()
        return ob

    def mitigate_internal_obs(i: int) -> list[OrderBlock]:
        cur_high = float(h[i])
        cur_low = float(l[i])
        kept = []
        removed = []
        for ob in internalOrderBlocks:
            if ob.bias == BEARISH:
                if cur_high > ob.barHigh:
                    removed.append(ob)
                    continue
            else:
                if cur_low < ob.barLow:
                    removed.append(ob)
                    continue
            kept.append(ob)
        internalOrderBlocks[:] = kept
        return removed

    def mitigate_fvgs(i: int):
        cur_high = float(h[i])
        cur_low = float(l[i])
        # deleteFairValueGaps() style: remove if crossed
        for k in range(len(active_fvgs) - 1, -1, -1):
            g = active_fvgs[k]
            if (g.bias == BULLISH and cur_low < g.bottom) or (g.bias == BEARISH and cur_high > g.top):
                active_fvgs.pop(k)

    def maybe_create_fvg_on_new_4h(i: int):
        nonlocal cum_abs, prev_4h_idx
        if len(t4) < 5:
            return

        # map LTF time -> 4h index (rightmost open <= t[i])
        j = int(np.searchsorted(t4, float(t[i]), side="right") - 1)
        if j < 0:
            return

        new_tf = (j != prev_4h_idx)
        if new_tf and j >= 2:
            last_close = float(c4[j - 1])
            last_open = float(o4[j - 1])
            last_time = float(t4[j - 1])

            current_high = float(h4[j])
            current_low = float(l4[j])
            current_time = float(t4[j])

            last2_high = float(h4[j - 2])
            last2_low = float(l4[j - 2])

            denom = last_open * 100.0
            bar_delta_percent = ((last_close - last_open) / denom) if abs(denom) > 1e-12 else 0.0

            cum_abs += abs(bar_delta_percent)
            denom_idx = float(i) if i > 0 else 1.0
            threshold = (cum_abs / denom_idx) * 2.0

            bullish = (current_low > last2_high) and (last_close > last2_high) and (bar_delta_percent > threshold)
            bearish = (current_high < last2_low) and (last_close < last2_low) and ((-bar_delta_percent) > threshold)

            extend_sec = float(FVG_EXTEND * ltf_sec)

            if bullish:
                active_fvgs.insert(0, FVG(
                    top=float(current_low),
                    bottom=float(last2_high),
                    bias=BULLISH,
                    leftTime=last_time,
                    rightTime=float(current_time + extend_sec),
                ))
            if bearish:
                active_fvgs.insert(0, FVG(
                    top=float(current_high),
                    bottom=float(last2_low),
                    bias=BEARISH,
                    leftTime=last_time,
                    rightTime=float(current_time + extend_sec),
                ))

            if len(active_fvgs) > 400:
                del active_fvgs[400:]

        prev_4h_idx = j

    # MAIN LOOP
    for i in range(n):
        # --- exit management on bar i (position exists from earlier entry)
        if pos is not None:
            hi = float(h[i])
            lo = float(l[i])
            exit_price = None
            exit_reason = None

            if pos.direction == BULLISH:
                hit_sl = lo <= pos.stop
                hit_tp = hi >= pos.take
                if hit_sl and hit_tp:
                    exit_price = pos.stop
                    exit_reason = "SL&TP_same_bar->SL"
                elif hit_sl:
                    exit_price = pos.stop
                    exit_reason = "SL"
                elif hit_tp:
                    exit_price = pos.take
                    exit_reason = "TP"
            else:
                hit_sl = hi >= pos.stop
                hit_tp = lo <= pos.take
                if hit_sl and hit_tp:
                    exit_price = pos.stop
                    exit_reason = "SL&TP_same_bar->SL"
                elif hit_sl:
                    exit_price = pos.stop
                    exit_reason = "SL"
                elif hit_tp:
                    exit_price = pos.take
                    exit_reason = "TP"

            if exit_price is not None:
                eq_before = equity
                pnl = (pos.qty * (exit_price - pos.entry)) if pos.direction == BULLISH else (pos.qty * (pos.entry - exit_price))
                equity = equity + pnl

                trades.append({
                    "entry_time": s_to_dt_utc(pos.entry_time),
                    "exit_time": s_to_dt_utc(float(t[i])),
                    "entry_time_s": float(pos.entry_time),
                    "exit_time_s": float(t[i]),
                    "dir": "LONG" if pos.direction == BULLISH else "SHORT",
                    "entry": float(pos.entry),
                    "stop": float(pos.stop),
                    "take": float(pos.take),
                    "exit": float(exit_price),
                    "pnl": float(pnl),
                    "equity_before": float(eq_before),
                    "equity_after": float(equity),
                    "reason": pos.reason,
                    "exit_reason": exit_reason,
                })
                pos = None

        equity_curve[i] = equity

        # --- update structure states ALWAYS (even if in position)
        pivot_update(i, SWINGS_LEN, is_internal=False)
        pivot_update(i, INTERNAL_LEN, is_internal=True)

        if trailing.top is not None and trailing.bottom is not None:
            update_trailing_extremes(i)

        # --- OB mitigation FIRST (so “destroyed then next OB” is well-defined)
        pre_top_ob = internalOrderBlocks[0] if internalOrderBlocks else None
        removed_obs = mitigate_internal_obs(i)
        if pre_top_ob is not None and removed_obs:
            for rob in removed_obs:
                if (rob.bias == pre_top_ob.bias and
                    abs(rob.barHigh - pre_top_ob.barHigh) < 1e-9 and
                    abs(rob.barLow - pre_top_ob.barLow) < 1e-9 and
                    abs(rob.barTime - pre_top_ob.barTime) < 1e-9):
                    destroyed_ob = pre_top_ob
                    waiting_replacement = True
                    break

        # --- FVG: mitigate each bar, then create on new 4h
        mitigate_fvgs(i)
        maybe_create_fvg_on_new_4h(i)

        # --- Detect new internal OBs on this bar close
        new_obs: list[OrderBlock] = []
        if i > 0:
            prevc = float(c[i - 1])
            curc = float(c[i])

            if internalHigh.currentLevel is not None and not internalHigh.crossed:
                if crossover(prevc, curc, float(internalHigh.currentLevel)):
                    internalHigh.crossed = True
                    internalTrend.bias = BULLISH
                    ob = store_internal_ob(internalHigh, BULLISH, i)
                    if ob is not None:
                        new_obs.append(ob)

            if internalLow.currentLevel is not None and not internalLow.crossed:
                if crossunder(prevc, curc, float(internalLow.currentLevel)):
                    internalLow.crossed = True
                    internalTrend.bias = BEARISH
                    ob = store_internal_ob(internalLow, BEARISH, i)
                    if ob is not None:
                        new_obs.append(ob)

            # swingTrend update (not used as filter, but keep consistent state)
            if swingHigh.currentLevel is not None and not swingHigh.crossed:
                if crossover(prevc, curc, float(swingHigh.currentLevel)):
                    swingHigh.crossed = True
                    swingTrend.bias = BULLISH
            if swingLow.currentLevel is not None and not swingLow.crossed:
                if crossunder(prevc, curc, float(swingLow.currentLevel)):
                    swingLow.crossed = True
                    swingTrend.bias = BEARISH

        # --- Entry evaluation (only if flat and we have next bar)
        if i >= n - 1:
            continue
        if pos is not None:
            continue
        if not new_obs:
            continue

        # Next bar open entry
        entry_price = float(o[i + 1])

        # ========== L1: “destroyed then next OB opposite + intersects”
        # IMPORTANT: only the FIRST OB after destruction is eligible (“sledushiy OB”).
        if waiting_replacement and destroyed_ob is not None:
            first_ob = new_obs[0]
            if (first_ob.bias != destroyed_ob.bias) and zones_intersect(first_ob.barLow, first_ob.barHigh, destroyed_ob.barLow, destroyed_ob.barHigh):
                direction = BULLISH if first_ob.bias == BULLISH else BEARISH
                stop = float(first_ob.barLow) if direction == BULLISH else float(first_ob.barHigh)
                take = compute_tp_1to2(entry_price, stop, direction)
                if take is not None:
                    risk_per_unit = (entry_price - stop) if direction == BULLISH else (stop - entry_price)
                    if risk_per_unit > 0:
                        qty = (equity * RISK_PCT) / risk_per_unit
                        pos = Position(direction=direction, entry=entry_price, qty=qty, stop=stop, take=float(take),
                                       entry_time=float(t[i + 1]), reason="L1_destroyed_then_next_ob_opposite_and_intersects")
                        l1_count += 1
                        waiting_replacement = False
                        destroyed_ob = None
                        continue
            # “следующий” не подошёл -> сбрасываем ожидание (по твоей формулировке)
            waiting_replacement = False
            destroyed_ob = None

        # ========== L2: any new OB intersects any active FVG with same direction
        # Process OBs in order; take the first that matches.
        opened = False
        for ob in new_obs:
            direction = BULLISH if ob.bias == BULLISH else BEARISH

            ok = False
            for g in active_fvgs:
                if g.bias != ob.bias:
                    continue
                if zones_intersect(ob.barLow, ob.barHigh, g.bottom, g.top):
                    ok = True
                    break

            if not ok:
                continue

            stop = float(ob.barLow) if direction == BULLISH else float(ob.barHigh)
            take = compute_tp_1to2(entry_price, stop, direction)
            if take is None:
                continue

            risk_per_unit = (entry_price - stop) if direction == BULLISH else (stop - entry_price)
            if risk_per_unit <= 0:
                continue

            qty = (equity * RISK_PCT) / risk_per_unit
            pos = Position(direction=direction, entry=entry_price, qty=qty, stop=stop, take=float(take),
                           entry_time=float(t[i + 1]), reason="L2_new_ob_intersects_fvg_same_direction")
            l2_count += 1
            opened = True
            break

        if opened:
            continue

    out = df.copy()
    out["equity"] = equity_curve
    trades_df = pd.DataFrame(trades)
    return out, trades_df, l1_count, l2_count


# =============================
# UI items
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

        for x, oo, hh, ll, cc in zip(self.times, self.open, self.high, self.low, self.close):
            p.setPen(wick_pen)
            p.drawLine(QtCore.QPointF(x, ll), QtCore.QPointF(x, hh))

            up = cc >= oo
            p.setPen(up_pen if up else dn_pen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 85) if up else QtGui.QColor(237, 72, 7)))

            top = cc if up else oo
            bot = oo if up else cc
            rect = QtCore.QRectF(x - self.w / 2.0, bot, self.w, max(1e-9, top - bot))
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
# Backtest Window
# =============================
class BacktestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest 6m: L1 destroyed->next OB opposite+intersect | L2 OB∩FVG same dir")

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
        top.addWidget(QtWidgets.QLabel("Timeframe:"))
        top.addWidget(self.tf_box)

        self.status = QtWidgets.QLabel("")
        self.status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top.addWidget(self.status, 1)

        pg.setConfigOptions(antialias=False)
        self.plot_price = pg.PlotWidget(axisItems={"bottom": DateAxis(orientation="bottom")})
        self.plot_eq = pg.PlotWidget(axisItems={"bottom": DateAxis(orientation="bottom")})
        self._apply_style(self.plot_price)
        self._apply_style(self.plot_eq)
        self.plot_eq.setMaximumHeight(240)

        layout.addWidget(self.plot_price, 1)
        layout.addWidget(self.plot_eq, 0)

        self.candles = CandleItem()
        self.plot_price.addItem(self.candles)

        self.eq_curve = pg.PlotDataItem()
        self.plot_eq.addItem(self.eq_curve)

        self.trade_items = []
        self.tf_box.currentTextChanged.connect(lambda _: self.recalc())
        self.recalc()

    def _apply_style(self, plotw: pg.PlotWidget):
        plotw.setBackground((0, 0, 0))
        plotw.showGrid(x=True, y=True, alpha=0.15)
        for ax in ("left", "bottom"):
            a = plotw.getAxis(ax)
            a.setTextPen(pg.mkPen((255, 255, 255, 140)))
            a.setPen(pg.mkPen((255, 255, 255, 60)))
        vb = plotw.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setDefaultPadding(0.02)

    def _clear_trade_items(self):
        for it in self.trade_items:
            try:
                self.plot_price.removeItem(it)
            except Exception:
                pass
        self.trade_items.clear()

    def recalc(self):
        tf = self.tf_box.currentText()
        self.status.setText("loading 6 months...")
        QtWidgets.QApplication.processEvents()

        start_ms = utc_6_months_ago_ms()
        end_ms = utc_now_ms()

        df = fetch_klines_from(SYMBOL, tf, start_ms, end_ms)
        # extra earlier 4h for stable FVG warmup
        df4h = fetch_klines_from(SYMBOL, "4h", start_ms - 20 * interval_seconds("4h") * 1000, end_ms)

        if len(df) < 800:
            self.status.setText("too few bars")
            return

        out, trades, l1c, l2c = run_backtest(df, df4h)

        w = interval_seconds(tf) * 0.7
        self.candles.set_data(
            out["time"].to_numpy(),
            out["open"].to_numpy(),
            out["high"].to_numpy(),
            out["low"].to_numpy(),
            out["close"].to_numpy(),
            w=w
        )
        self.eq_curve.setData(out["time"].to_numpy(dtype=float), out["equity"].to_numpy(dtype=float))

        self._clear_trade_items()

        if len(trades) > 0:
            for _, tr in trades.iterrows():
                et = float(tr["entry_time_s"])
                xt = float(tr["exit_time_s"])
                entry = float(tr["entry"])
                stop = float(tr["stop"])
                take = float(tr["take"])
                exitp = float(tr["exit"])
                is_long = (tr["dir"] == "LONG")

                pnl_line = pg.PlotDataItem([et, xt], [entry, exitp], pen=pg.mkPen((255, 255, 255, 140), width=1))
                sl_line = pg.PlotDataItem([et, xt], [stop, stop], pen=pg.mkPen((255, 70, 70, 200), width=1, style=QtCore.Qt.DashLine))
                tp_line = pg.PlotDataItem([et, xt], [take, take], pen=pg.mkPen((70, 255, 140, 200), width=1, style=QtCore.Qt.DashLine))

                self.plot_price.addItem(pnl_line)
                self.plot_price.addItem(sl_line)
                self.plot_price.addItem(tp_line)
                self.trade_items.extend([pnl_line, sl_line, tp_line])

                entry_marker = pg.ScatterPlotItem([et], [entry],
                                                  symbol='t' if is_long else 't1',
                                                  size=12,
                                                  pen=pg.mkPen((255, 255, 255, 230)),
                                                  brush=pg.mkBrush((0, 0, 0, 0)))
                exit_marker = pg.ScatterPlotItem([xt], [exitp],
                                                 symbol='o',
                                                 size=9,
                                                 pen=pg.mkPen((255, 255, 255, 230)),
                                                 brush=pg.mkBrush((0, 0, 0, 0)))
                self.plot_price.addItem(entry_marker)
                self.plot_price.addItem(exit_marker)
                self.trade_items.extend([entry_marker, exit_marker])

        final_eq = float(out["equity"].iloc[-1])
        ntr = int(len(trades))
        self.status.setText(f"{SYMBOL} | TF={tf} | bars={len(out)} | trades={ntr} (L1={l1c}, L2={l2c}) | final={final_eq:.2f}")

        self.plot_price.enableAutoRange(axis="xy", enable=True)
        self.plot_price.autoRange(padding=0.02)
        self.plot_eq.enableAutoRange(axis="xy", enable=True)
        self.plot_eq.autoRange(padding=0.02)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BacktestWindow()
    win.resize(1550, 950)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
