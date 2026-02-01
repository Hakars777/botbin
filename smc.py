import sys
import json
import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd

from websocket import WebSocketApp  # websocket-client
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


# =============================
# FIXED SETTINGS
# =============================
# showInternalStructure      -> False   (we do NOT draw internal BOS/CHOCH)
# showSwingStructure         -> True    (we draw swing BOS/CHOCH)
# showStrong/WeakHigh/Low    -> True    (we draw trailing strong/weak)
# InternalOrderBlocks        -> True(5) (we compute + draw internal OBs)
# SwingOrderBlocks           -> False   (no swing OBs)
# EqualHighLow               -> False   (disabled)
# FairValueGaps              -> True, Auto Threshold, timeframe 4h
# ExtendFvg                  -> 5

SYMBOL = "BTCUSDT"
TF_LIST = ["15m", "4h", "1d"]
DEFAULT_TF = "15m"

HIST_TARGET_BARS = 9000

BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = +1
BEARISH = -1
BOS = "BOS"
CHOCH = "CHoCH"

GREEN = (8, 153, 129)
RED = (242, 54, 69)

# Structure lengths
SWINGS_LEN = 50
INTERNAL_LEN = 5

# Internal OBs
INTERNAL_OB_COUNT = 2
INTERNAL_BULL_OB_FILL = (49, 121, 245, 80)    # ~ #3179f5 alpha 80
INTERNAL_BEAR_OB_FILL = (247, 124, 128, 80)   # ~ #f77c80 alpha 80

# Volatility parsing
ATR_LEN = 200
HIGH_VOL_MULT = 2.0  # (high-low) >= 2*ATR

# FVG (4h, auto threshold, extend=5)
FVG_TF = "4h"
FVG_EXTEND = 5
FVG_BULL_COLOR = (0, 255, 104, 70)   # #00ff68 alpha 70
FVG_BEAR_COLOR = (255, 0, 8, 70)     # #ff0008 alpha 70

# Binance Futures endpoints
BINANCE_REST = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/ws"


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


def guess_base_tf_seconds(t: np.ndarray) -> int:
    if t.size < 3:
        return 60
    d = np.diff(t)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 60
    med = float(np.median(d))
    if not np.isfinite(med) or med <= 0:
        return 60
    return max(1, int(round(med)))


# =============================
# REST: klines paging
# =============================
def fetch_klines_paged(symbol: str, tf: str, target_bars: int) -> pd.DataFrame:
    sec = interval_seconds(tf)
    limit = 1500
    out = []
    start_time = None

    while len(out) < target_bars:
        params = {"symbol": symbol, "interval": tf, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time

        r = requests.get(f"{BINANCE_REST}/fapi/v1/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)
        last_open_ms = int(data[-1][0])
        start_time = last_open_ms + sec * 1000

        if len(data) < limit:
            break

        time.sleep(0.06)
        if len(out) > target_bars + 3000:
            break

    dedup = {}
    for k in out:
        dedup[int(k[0])] = k
    keys = sorted(dedup.keys())
    rows = [dedup[t] for t in keys][-target_bars:]

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    df["time"] = df["open_time"].astype(np.int64).apply(ms_to_s).astype(float)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


# =============================
# volatility parsing
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
    pH = np.where(hv, low, high)   # parsedHigh
    pL = np.where(hv, high, low)   # parsedLow
    return pH, pL


# =============================
# LEG logic
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
    barTime: float | None = None
    barIndex: int | None = None
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
    rightTime: float  # extended right edge (time coordinate)

def merge_order_blocks(order_blocks: list[OrderBlock]) -> list[OrderBlock]:
    if not order_blocks:
        return []

    by_bias: dict[int, list[OrderBlock]] = {BULLISH: [], BEARISH: []}
    for ob in order_blocks:
        lo = float(min(ob.barLow, ob.barHigh))
        hi = float(max(ob.barLow, ob.barHigh))
        by_bias[ob.bias].append(OrderBlock(barHigh=hi, barLow=lo, barTime=float(ob.barTime), bias=int(ob.bias)))

    merged_all: list[OrderBlock] = []
    eps = 1e-12

    for bias, lst in by_bias.items():
        if not lst:
            continue

        lst.sort(key=lambda x: x.barLow)

        cur = lst[0]
        for ob in lst[1:]:
            if ob.barLow <= cur.barHigh + eps:
                cur.barHigh = max(cur.barHigh, ob.barHigh)
                cur.barLow = min(cur.barLow, ob.barLow)
                cur.barTime = min(cur.barTime, ob.barTime)
            else:
                merged_all.append(cur)
                cur = ob
        merged_all.append(cur)

    merged_all.sort(key=lambda x: x.barTime, reverse=True)
    return merged_all


# =============================
# Core Engine
# =============================
def compute_overlays(df: pd.DataFrame, df4h: pd.DataFrame) -> dict:
    t = df["time"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    n = len(df)
    if n < max(600, SWINGS_LEN * 6):
        return {"swing_struct": [], "strongweak": None, "internal_obs": [], "fvgs": []}

    base_tf_sec = guess_base_tf_seconds(t)

    # Volatility parsing arrays (parsedHigh/parsedLow)
    atr = atr_wilder(h, l, c, ATR_LEN)
    pH, pL = parsed_hilo(h, l, atr)

    # Pivots
    swingHigh = Pivot()
    swingLow = Pivot()
    internalHigh = Pivot()
    internalLow = Pivot()

    swingTrend = Trend(0)
    internalTrend = Trend(0)

    trailing = Trailing()

    internalOrderBlocks: list[OrderBlock] = []
    swing_events: list[dict] = []

    # Precompute legs to emulate ta.change(leg)
    leg_swing_series = np.zeros(n, dtype=int)
    leg_internal_series = np.zeros(n, dtype=int)
    cur_s = 0
    cur_i = 0
    for i in range(n):
        cur_s = leg_at(i, SWINGS_LEN, h, l, cur_s)
        cur_i = leg_at(i, INTERNAL_LEN, h, l, cur_i)
        leg_swing_series[i] = cur_s
        leg_internal_series[i] = cur_i

    def store_internal_order_block(pivot: Pivot, bias: int, cur_bar_i: int):
        # slice parsedHighs/parsedLows from pivot.barIndex to bar_index (end exclusive),
        # pick max/min, then unshift, cap to 100
        if pivot.barIndex is None:
            return
        a = int(pivot.barIndex)
        b = int(cur_bar_i)  # end exclusive
        if b <= a:
            return
        if a < 0 or b > n:
            return

        if bias == BEARISH:
            seg = pH[a:b]
            if seg.size == 0:
                return
            idx = a + int(np.argmax(seg))  # first max
        else:
            seg = pL[a:b]
            if seg.size == 0:
                return
            idx = a + int(np.argmin(seg))  # first min

        ob = OrderBlock(barHigh=float(pH[idx]), barLow=float(pL[idx]), barTime=float(t[idx]), bias=bias)
        internalOrderBlocks.insert(0, ob)
        if len(internalOrderBlocks) > 100:
            internalOrderBlocks.pop()

    def mitigate_internal_obs(cur_bar_i: int):
        curH = float(h[cur_bar_i])
        curL = float(l[cur_bar_i])
        kept = []
        for ob in internalOrderBlocks:
            if ob.bias == BEARISH:
                if curH > ob.barHigh:
                    continue
            else:
                if curL < ob.barLow:
                    continue
            kept.append(ob)
        internalOrderBlocks[:] = kept

    def pivot_update_on_leg_change(i: int, size: int, is_internal: bool):
        if i <= 0:
            return
        pivot_i = i - size
        if pivot_i < 0:
            return

        if is_internal:
            prev_leg = leg_internal_series[i - 1]
            cur_leg = leg_internal_series[i]
        else:
            prev_leg = leg_swing_series[i - 1]
            cur_leg = leg_swing_series[i]

        ch = cur_leg - prev_leg
        if ch == 0:
            return

        if ch == +1:
            # startOfBullishLeg => pivotLow
            p = internalLow if is_internal else swingLow
            p.lastLevel = p.currentLevel
            p.currentLevel = float(l[pivot_i])
            p.crossed = False
            p.barTime = float(t[pivot_i])
            p.barIndex = int(pivot_i)

            if not is_internal:
                trailing.bottom = p.currentLevel
                trailing.barTime = p.barTime
                trailing.barIndex = p.barIndex
                trailing.lastBottomTime = p.barTime

        elif ch == -1:
            # startOfBearishLeg => pivotHigh
            p = internalHigh if is_internal else swingHigh
            p.lastLevel = p.currentLevel
            p.currentLevel = float(h[pivot_i])
            p.crossed = False
            p.barTime = float(t[pivot_i])
            p.barIndex = int(pivot_i)

            if not is_internal:
                trailing.top = p.currentLevel
                trailing.barTime = p.barTime
                trailing.barIndex = p.barIndex
                trailing.lastTopTime = p.barTime

    def update_trailing_extremes(i: int):
        hi = float(h[i])
        lo = float(l[i])

        if trailing.top is not None:
            if hi >= trailing.top:
                trailing.top = hi
                trailing.lastTopTime = float(t[i])

        if trailing.bottom is not None:
            if lo <= trailing.bottom:
                trailing.bottom = lo
                trailing.lastBottomTime = float(t[i])

    # Bar-by-bar execution
    for i in range(n):
        if trailing.top is not None and trailing.bottom is not None:
            update_trailing_extremes(i)

        pivot_update_on_leg_change(i, SWINGS_LEN, is_internal=False)
        pivot_update_on_leg_change(i, INTERNAL_LEN, is_internal=True)

        prev_close = float(c[i - 1] if i > 0 else c[i])
        cur_close = float(c[i])

        if internalHigh.currentLevel is not None and not internalHigh.crossed:
            if swingHigh.currentLevel is not None and internalHigh.currentLevel != swingHigh.currentLevel:
                if crossover(prev_close, cur_close, float(internalHigh.currentLevel)):
                    internalHigh.crossed = True
                    internalTrend.bias = BULLISH
                    store_internal_order_block(internalHigh, BULLISH, i)

        if internalLow.currentLevel is not None and not internalLow.crossed:
            if swingLow.currentLevel is not None and internalLow.currentLevel != swingLow.currentLevel:
                if crossunder(prev_close, cur_close, float(internalLow.currentLevel)):
                    internalLow.crossed = True
                    internalTrend.bias = BEARISH
                    store_internal_order_block(internalLow, BEARISH, i)

        # Swing structure drawn (BOS/CHOCH)
        if swingHigh.currentLevel is not None and swingHigh.barIndex is not None and not swingHigh.crossed:
            if i > 0 and crossover(float(c[i - 1]), float(c[i]), float(swingHigh.currentLevel)):
                tag = CHOCH if swingTrend.bias == BEARISH else BOS
                swingHigh.crossed = True
                swingTrend.bias = BULLISH

                mid_idx = int(np.floor(0.5 * (int(swingHigh.barIndex) + i) + 0.5)) 
                mid_idx = max(0, min(n - 1, mid_idx))
                swing_events.append({
                    "tag": tag,
                    "level": float(swingHigh.currentLevel),
                    "t0": float(swingHigh.barTime) if swingHigh.barTime is not None else float(t[int(swingHigh.barIndex)]),
                    "t1": float(t[i]),
                    "tmid": float(t[mid_idx]),
                    "dir": "UP",
                })

        if swingLow.currentLevel is not None and swingLow.barIndex is not None and not swingLow.crossed:
            if i > 0 and crossunder(float(c[i - 1]), float(c[i]), float(swingLow.currentLevel)):
                tag = CHOCH if swingTrend.bias == BULLISH else BOS
                swingLow.crossed = True
                swingTrend.bias = BEARISH

                mid_idx = int(np.floor(0.5 * (int(swingLow.barIndex) + i) + 0.5)) 
                mid_idx = max(0, min(n - 1, mid_idx))
                swing_events.append({
                    "tag": tag,
                    "level": float(swingLow.currentLevel),
                    "t0": float(swingLow.barTime) if swingLow.barTime is not None else float(t[int(swingLow.barIndex)]),
                    "t1": float(t[i]),
                    "tmid": float(t[mid_idx]),
                    "dir": "DOWN",
                })

        # deleteOrderBlocks(true) after displayStructure calls => mitigation on current bar
        if i >= 1:
            mitigate_internal_obs(i)

    # Strong/Weak text depends on CURRENT swingTrend.bias
    strongweak = None
    if (
        trailing.top is not None and trailing.bottom is not None
        and trailing.lastTopTime is not None and trailing.lastBottomTime is not None
    ):
        top_text = "Strong High" if swingTrend.bias == BEARISH else "Weak High"
        bot_text = "Strong Low" if swingTrend.bias == BULLISH else "Weak Low"
        strongweak = {
            "top": float(trailing.top),
            "bottom": float(trailing.bottom),
            "top_text": top_text,
            "bottom_text": bot_text,
            "lastTopTime": float(trailing.lastTopTime),
            "lastBottomTime": float(trailing.lastBottomTime),
        }

    fvgs: list[FVG] = []
    if df4h is not None and len(df4h) >= 5:
        t4 = df4h["time"].to_numpy(dtype=float)
        o4 = df4h["open"].to_numpy(dtype=float)
        h4 = df4h["high"].to_numpy(dtype=float)
        l4 = df4h["low"].to_numpy(dtype=float)
        c4 = df4h["close"].to_numpy(dtype=float)

        extend_sec = float(FVG_EXTEND * base_tf_sec)

        tmp: list[FVG] = []
        cum_abs = 0.0
        prev_j = -999999

        for i in range(n):
            cur_low = float(l[i])
            cur_high = float(h[i])
            for k in range(len(tmp) - 1, -1, -1):
                g = tmp[k]
                if (g.bias == BULLISH and cur_low < g.bottom) or (g.bias == BEARISH and cur_high > g.top):
                    tmp.pop(k)

            j = int(np.searchsorted(t4, t[i], side="right") - 1)
            if j < 0:
                continue

            new_tf = (j != prev_j)
            if new_tf and j >= 2:
                lastClose = float(c4[j - 1])
                lastOpen = float(o4[j - 1])
                lastTime = float(t4[j - 1])

                currentHigh = float(h4[j])
                currentLow = float(l4[j])
                currentTime = float(t4[j])

                last2High = float(h4[j - 2])
                last2Low = float(l4[j - 2])

                denom = (lastOpen * 100.0)
                barDeltaPercent = ((lastClose - lastOpen) / denom) if abs(denom) > 1e-12 else 0.0

                # threshold = ta.cum(abs(newTF ? barDeltaPercent : 0))/bar_index*2
                cum_abs += abs(barDeltaPercent)
                denom_idx = float(i) if i > 0 else 1.0
                threshold = (cum_abs / denom_idx) * 2.0

                bullish = (currentLow > last2High) and (lastClose > last2High) and (barDeltaPercent > threshold)
                bearish = (currentHigh < last2Low) and (lastClose < last2Low) and ((-barDeltaPercent) > threshold)

                if bullish:
                    tmp.insert(0, FVG(
                        top=currentLow,
                        bottom=last2High,
                        bias=BULLISH,
                        leftTime=lastTime,
                        rightTime=currentTime + extend_sec,
                    ))
                if bearish:
                    tmp.insert(0, FVG(
                        top=currentHigh,
                        bottom=last2Low,
                        bias=BEARISH,
                        leftTime=lastTime,
                        rightTime=currentTime + extend_sec,
                    ))
                if len(tmp) > 500:
                    tmp = tmp[:500]

            prev_j = j

        fvgs = tmp[:200]

    merged_internal_obs = merge_order_blocks(internalOrderBlocks)    

    return {
        "swing_struct": swing_events[-250:],
        "strongweak": strongweak,
        "internal_obs": merged_internal_obs[:INTERNAL_OB_COUNT], 
        "fvgs": fvgs,
    }


# =============================
# pyqtgraph items
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
        wick_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 255))
        wick_pen.setWidthF(1.8)

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
# WebSocket worker
# =============================
class KlineWS:
    def __init__(self, symbol: str, tf: str, q: Queue, tag: str):
        self.symbol = symbol.lower()
        self.tf = tf
        self.q = q
        self.tag = tag
        self.ws = None
        self.thread = None
        self.stop_evt = threading.Event()

    def start(self):
        self.stop_evt.clear()
        url = f"{BINANCE_WS_BASE}/{self.symbol}@kline_{self.tf}"
        self.ws = WebSocketApp(
            url,
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
        )
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    def run(self):
        while not self.stop_evt.is_set():
            try:
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            time.sleep(1.0)

    def on_open(self, ws):
        self.q.put({"type": "status", "tag": self.tag, "msg": "ws open"})

    def on_close(self, ws, *args):
        self.q.put({"type": "status", "tag": self.tag, "msg": "ws close"})

    def on_error(self, ws, err):
        self.q.put({"type": "status", "tag": self.tag, "msg": f"ws error: {err}"})

    def on_message(self, ws, message: str):
        try:
            obj = json.loads(message)
            k = obj.get("k", {})
            if not k:
                return
            self.q.put({
                "type": "kline",
                "tag": self.tag,
                "tf": self.tf,
                "t": ms_to_s(int(k["t"])),
                "o": float(k["o"]),
                "h": float(k["h"]),
                "l": float(k["l"]),
                "c": float(k["c"]),
                "v": float(k["v"]),
                "closed": bool(k["x"]),
            })
        except Exception:
            pass


# =============================
# UI
# =============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(SYMBOL + " Futures")
        self.q = Queue()

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
        self.plot = pg.PlotWidget(axisItems={"bottom": DateAxis(orientation="bottom")})
        layout.addWidget(self.plot, 1)
        self.apply_style()

        self.candles = CandleItem()
        self.plot.addItem(self.candles)

        # crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((255, 255, 255, 60)))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((255, 255, 255, 60)))
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        # overlay items
        self.ob_items = []
        self.fvg_items = []
        self.struct_items = []
        self.sw_items = []

        self.df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        self.df4h = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        self.ws_tf = None
        self.ws_4h = None

        self.tf_box.currentTextChanged.connect(self.on_tf_change)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(50)

        self.load_tf(DEFAULT_TF)

        # Always keep 4h stream for FVG
        self.ws_4h = KlineWS(SYMBOL, "4h", self.q, "4h")
        self.ws_4h.start()

    def apply_style(self):
        self.plot.setBackground((0, 0, 0))
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        for ax in ("left", "bottom"):
            a = self.plot.getAxis(ax)
            a.setTextPen(pg.mkPen((255, 255, 255, 140)))
            a.setPen(pg.mkPen((255, 255, 255, 60)))
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setDefaultPadding(0.02)

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot.sceneBoundingRect().contains(pos):
            mp = self.plot.getViewBox().mapSceneToView(pos)
            self.vline.setPos(mp.x())
            self.hline.setPos(mp.y())

    def closeEvent(self, e):
        try:
            if self.ws_tf:
                self.ws_tf.stop()
            if self.ws_4h:
                self.ws_4h.stop()
        except Exception:
            pass
        super().closeEvent(e)

    def set_status(self, s: str):
        self.status.setText(s)

    def clear_items(self, items):
        for it in items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        items.clear()

    def on_tf_change(self, tf: str):
        self.load_tf(tf)

    def load_tf(self, tf: str):
        self.set_status("loading...")

        if self.ws_tf:
            self.ws_tf.stop()
            self.ws_tf = None

        self.df = fetch_klines_paged(SYMBOL, tf, HIST_TARGET_BARS)
        self.df4h = fetch_klines_paged(SYMBOL, "4h", max(3000, HIST_TARGET_BARS // 3))

        w = interval_seconds(tf) * 0.7
        self.candles.set_data(
            self.df["time"].to_numpy(),
            self.df["open"].to_numpy(),
            self.df["high"].to_numpy(),
            self.df["low"].to_numpy(),
            self.df["close"].to_numpy(),
            w=w
        )
        self.plot.enableAutoRange(axis="xy", enable=True)
        self.plot.autoRange(padding=0.02)

        self.recalc_and_draw(tf)

        self.ws_tf = KlineWS(SYMBOL, tf, self.q, "tf")
        self.ws_tf.start()

        self.set_status(f"{SYMBOL} | TF={tf} | bars={len(self.df)} | UTC")

    def recalc_and_draw(self, tf: str):
        self.clear_items(self.ob_items)
        self.clear_items(self.fvg_items)
        self.clear_items(self.struct_items)
        self.clear_items(self.sw_items)

        overlays = compute_overlays(self.df, self.df4h)
        tf_sec = interval_seconds(tf)

        t_last = float(self.df["time"].iloc[-1])
        rightTimeBar = t_last + 20.0 * tf_sec

        # ---- Swing BOS/CHOCH
        for ev in overlays["swing_struct"]:
            level = ev["level"]
            t0 = ev["t0"]
            t1 = ev["t1"]
            tmid = ev["tmid"]
            tag = ev["tag"]

            if ev.get("dir") == "UP":
                col = (*GREEN, 200)
            else:
                col = (*RED, 200)

            line = pg.PlotDataItem([t0, t1], [level, level], pen=pg.mkPen(col, width=2))
            self.plot.addItem(line)
            self.struct_items.append(line)

            txt = pg.TextItem(tag, color=(255, 255, 255))
            txt.setPos(tmid, level)
            self.plot.addItem(txt)
            self.struct_items.append(txt)

        # ---- Strong/Weak High/Low
        sw = overlays["strongweak"]
        if sw:
            top_y = sw["top"]
            bot_y = sw["bottom"]
            top_t0 = sw["lastTopTime"]
            bot_t0 = sw["lastBottomTime"]

            top_line = pg.PlotDataItem([top_t0, rightTimeBar], [top_y, top_y],
                                       pen=pg.mkPen((*RED, 200), width=2))
            bot_line = pg.PlotDataItem([bot_t0, rightTimeBar], [bot_y, bot_y],
                                       pen=pg.mkPen((*GREEN, 200), width=2))
            self.plot.addItem(top_line)
            self.plot.addItem(bot_line)
            self.sw_items.extend([top_line, bot_line])

            top_lbl = pg.TextItem(sw["top_text"], color=(255, 255, 255))
            bot_lbl = pg.TextItem(sw["bottom_text"], color=(255, 255, 255))
            top_lbl.setPos(rightTimeBar, top_y)
            bot_lbl.setPos(rightTimeBar, bot_y)
            self.plot.addItem(top_lbl)
            self.plot.addItem(bot_lbl)
            self.sw_items.extend([top_lbl, bot_lbl])

        # ---- Internal Order Blocks (no merge, no border, extend right)
        for ob in overlays["internal_obs"]:
            x0 = float(ob.barTime)
            x1 = rightTimeBar

            y_low = float(min(ob.barLow, ob.barHigh))
            y_high = float(max(ob.barLow, ob.barHigh))
            if y_high <= y_low:
                continue

            if ob.bias == BULLISH:
                brush = pg.mkBrush(INTERNAL_BULL_OB_FILL)
            else:
                brush = pg.mkBrush(INTERNAL_BEAR_OB_FILL)

            rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x0, y_low, max(1.0, x1 - x0), y_high - y_low))
            rect.setBrush(brush)
            rect.setPen(QtGui.QPen(QtCore.Qt.NoPen)) 
            rect.setZValue(-10)
            self.plot.addItem(rect)
            self.ob_items.append(rect)

        # ---- FVG 4h 
        for g in overlays["fvgs"]:
            x0 = float(g.leftTime)
            x1 = float(min(g.rightTime, rightTimeBar))

            y_low = float(min(g.top, g.bottom))
            y_high = float(max(g.top, g.bottom))
            if y_high <= y_low:
                continue

            y_mid = 0.5 * (y_low + y_high)

            if g.bias == BULLISH:
                col = FVG_BULL_COLOR
            else:
                col = FVG_BEAR_COLOR

            brush = pg.mkBrush(col)
            pen = pg.mkPen(col, width=1)

            # bottom half
            rect1 = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x0, y_low, max(1.0, x1 - x0), max(1e-9, y_mid - y_low)))
            rect1.setBrush(brush)
            rect1.setPen(pen)
            rect1.setZValue(-20)
            self.plot.addItem(rect1)
            self.fvg_items.append(rect1)

            # top half
            rect2 = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x0, y_mid, max(1.0, x1 - x0), max(1e-9, y_high - y_mid)))
            rect2.setBrush(brush)
            rect2.setPen(pen)
            rect2.setZValue(-20)
            self.plot.addItem(rect2)
            self.fvg_items.append(rect2)

    def on_timer(self):
        tf_now = self.tf_box.currentText()
        changed_closed = False
        changed_4h_closed = False

        try:
            while True:
                msg = self.q.get_nowait()

                if msg["type"] == "status":
                    if msg.get("tag") == "tf":
                        self.set_status(f"{SYMBOL} | TF={tf_now} | bars={len(self.df)} | {msg.get('msg')}")
                    continue

                if msg["type"] == "kline":
                    tag = msg["tag"]
                    tt = float(msg["t"])
                    oo = float(msg["o"])
                    hh = float(msg["h"])
                    ll = float(msg["l"])
                    cc = float(msg["c"])
                    vv = float(msg["v"])
                    closed = bool(msg["closed"])

                    if tag == "4h":
                        if len(self.df4h) > 0:
                            last_t = float(self.df4h["time"].iloc[-1])
                            if abs(tt - last_t) < 1e-9:
                                self.df4h.loc[len(self.df4h) - 1, ["open", "high", "low", "close", "volume"]] = [oo, hh, ll, cc, vv]
                            elif tt > last_t:
                                row = {"time": tt, "open": oo, "high": hh, "low": ll, "close": cc, "volume": vv}
                                self.df4h = pd.concat([self.df4h, pd.DataFrame([row])], ignore_index=True)
                                if len(self.df4h) > 4500:
                                    self.df4h = self.df4h.iloc[-4500:].reset_index(drop=True)
                        changed_4h_closed = changed_4h_closed or closed
                        continue

                    if len(self.df) == 0:
                        continue
                    last_t = float(self.df["time"].iloc[-1])
                    if abs(tt - last_t) < 1e-9:
                        self.df.loc[len(self.df) - 1, ["open", "high", "low", "close", "volume"]] = [oo, hh, ll, cc, vv]
                    elif tt > last_t:
                        row = {"time": tt, "open": oo, "high": hh, "low": ll, "close": cc, "volume": vv}
                        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
                        if len(self.df) > HIST_TARGET_BARS:
                            self.df = self.df.iloc[-HIST_TARGET_BARS:].reset_index(drop=True)

                    w = interval_seconds(tf_now) * 0.7
                    self.candles.set_data(
                        self.df["time"].to_numpy(),
                        self.df["open"].to_numpy(),
                        self.df["high"].to_numpy(),
                        self.df["low"].to_numpy(),
                        self.df["close"].to_numpy(),
                        w=w
                    )

                    changed_closed = changed_closed or closed

        except Empty:
            pass

        if changed_closed or changed_4h_closed:
            self.recalc_and_draw(tf_now)
            self.set_status(f"{SYMBOL} | TF={tf_now} | bars={len(self.df)} | UTC")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1550, 900)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()