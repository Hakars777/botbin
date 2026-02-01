from __future__ import annotations

import sys
import json
import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, List, Any

import requests
import numpy as np
import pandas as pd
from websocket import WebSocketApp  # websocket-client


# =============================
# CONFIG (заполнено как пример)
# =============================
TELEGRAM_BOT_TOKEN = "8562555085:AAHTh9rv-yKYmHkVCHxcwbleMsTBwMTZuOM"
TELEGRAM_CHAT_ID = "1205943698"

SYMBOL = "BTCUSDT"
TF_LIST = ["15m", "4h", "1d"]
DEFAULT_TF = "15m"

# Сколько баров истории грузить
HIST_TARGET_BARS = 9000

# сохранение состояния (чтобы не дублировать алерты после рестарта)
STATE_FILE = "smc_alert_state.json"


# =============================
# FIXED SETTINGS (как в вашем коде)
# =============================
BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = +1
BEARISH = -1
BOS = "BOS"
CHOCH = "CHoCH"

# Structure lengths
SWINGS_LEN = 50
INTERNAL_LEN = 5

# Internal OBs
INTERNAL_OB_COUNT = 2

# Volatility parsing
ATR_LEN = 200
HIGH_VOL_MULT = 2.0

# FVG (4h)
FVG_TF = "4h"
FVG_EXTEND = 5

# Binance Futures endpoints
BINANCE_REST = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/ws"


# =============================
# Telegram
# =============================
class TelegramBot:
    def __init__(self, token: str, chat_id: str, timeout: int = 15):
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.base = f"https://api.telegram.org/bot{token}"
        self._last_send_ts = 0.0

    def send(self, text: str) -> bool:
        # мягкий анти-спам (не чаще ~1 сообщения/сек)
        now = time.time()
        dt = now - self._last_send_ts
        if dt < 1.0:
            time.sleep(1.0 - dt)

        try:
            r = requests.post(
                f"{self.base}/sendMessage",
                data={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=self.timeout,
            )
            ok = (r.status_code == 200)
            self._last_send_ts = time.time()
            return ok
        except Exception:
            self._last_send_ts = time.time()
            return False


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


def ts_utc_str(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def r6(x: float) -> float:
    return float(round(float(x), 6))


# =============================
# REST: klines paging
# =============================
def fetch_klines_paged(symbol: str, tf: str, target_bars: int) -> pd.DataFrame:
    sec = interval_seconds(tf)
    limit = 1500
    out: List[Any] = []
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

    dedup: Dict[int, Any] = {}
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


def parsed_hilo(high: np.ndarray, low: np.ndarray, atr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    currentLevel: Optional[float] = None
    lastLevel: Optional[float] = None
    crossed: bool = False
    barTime: Optional[float] = None
    barIndex: Optional[int] = None


@dataclass
class Trend:
    bias: int = 0  # +1 bull, -1 bear


@dataclass
class Trailing:
    top: Optional[float] = None
    bottom: Optional[float] = None
    barTime: Optional[float] = None
    barIndex: Optional[int] = None
    lastTopTime: Optional[float] = None
    lastBottomTime: Optional[float] = None


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
    rightTime: float


def merge_order_blocks(order_blocks: List[OrderBlock]) -> List[OrderBlock]:
    if not order_blocks:
        return []

    by_bias: Dict[int, List[OrderBlock]] = {BULLISH: [], BEARISH: []}
    for ob in order_blocks:
        lo = float(min(ob.barLow, ob.barHigh))
        hi = float(max(ob.barLow, ob.barHigh))
        by_bias[ob.bias].append(OrderBlock(barHigh=hi, barLow=lo, barTime=float(ob.barTime), bias=int(ob.bias)))

    merged_all: List[OrderBlock] = []
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
# Core Engine (ваш compute_overlays без UI)
# =============================
def compute_overlays(df: pd.DataFrame, df4h: pd.DataFrame) -> Dict[str, Any]:
    t = df["time"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    n = len(df)
    if n < max(600, SWINGS_LEN * 6):
        return {"swing_struct": [], "strongweak": None, "internal_obs": [], "fvgs": []}

    base_tf_sec = guess_base_tf_seconds(t)

    atr = atr_wilder(h, l, c, ATR_LEN)
    pH, pL = parsed_hilo(h, l, atr)

    swingHigh = Pivot()
    swingLow = Pivot()
    internalHigh = Pivot()
    internalLow = Pivot()

    swingTrend = Trend(0)
    internalTrend = Trend(0)

    trailing = Trailing()

    internalOrderBlocks: List[OrderBlock] = []
    swing_events: List[Dict[str, Any]] = []

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
        if pivot.barIndex is None:
            return
        a = int(pivot.barIndex)
        b = int(cur_bar_i)
        if b <= a:
            return
        if a < 0 or b > n:
            return

        if bias == BEARISH:
            seg = pH[a:b]
            if seg.size == 0:
                return
            idx = a + int(np.argmax(seg))
        else:
            seg = pL[a:b]
            if seg.size == 0:
                return
            idx = a + int(np.argmin(seg))

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

        # Swing structure (BOS/CHOCH)
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

        if i >= 1:
            mitigate_internal_obs(i)

    fvgs: List[FVG] = []
    if df4h is not None and len(df4h) >= 5:
        t4 = df4h["time"].to_numpy(dtype=float)
        o4 = df4h["open"].to_numpy(dtype=float)
        h4 = df4h["high"].to_numpy(dtype=float)
        l4 = df4h["low"].to_numpy(dtype=float)
        c4 = df4h["close"].to_numpy(dtype=float)

        extend_sec = float(FVG_EXTEND * base_tf_sec)

        tmp: List[FVG] = []
        cum_abs = 0.0
        prev_j = -999999

        for i in range(n):
            cur_low = float(l[i])
            cur_high = float(h[i])

            # remove mitigated FVGs
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
        "internal_obs": merged_internal_obs[:INTERNAL_OB_COUNT],
        "fvgs": fvgs,
    }


# =============================
# WebSocket worker
# =============================
class KlineWS:
    def __init__(self, symbol: str, tf: str, q: Queue, tag: str):
        self.symbol = symbol.lower()
        self.tf = tf
        self.q = q
        self.tag = tag
        self.ws: Optional[WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
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
                "t": ms_to_s(int(k["t"])),   # open time
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
# State
# =============================
def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "fvg_keys": [],
            "bos_keys": [],
            "ob_keys": [],
        }


def save_state(path: str, state: Dict[str, Any]) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        try:
            import os
            os.replace(tmp, path)
        except Exception:
            # fallback
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def fvg_key(g: FVG) -> str:
    # ключ без rightTime (он плавает)
    return f"{int(g.bias)}|{r6(g.top)}|{r6(g.bottom)}|{int(g.leftTime)}"


def bos_key(ev: Dict[str, Any]) -> str:
    # уникальность по времени пробоя t1 + уровень/направление/тег
    return f"{ev.get('tag')}|{ev.get('dir')}|{r6(float(ev.get('level', 0.0)))}|{int(float(ev.get('t1', 0.0)))}"


def ob_key(ob: OrderBlock) -> str:
    lo = r6(float(min(ob.barLow, ob.barHigh)))
    hi = r6(float(max(ob.barLow, ob.barHigh)))
    return f"{int(ob.bias)}|{lo}|{hi}|{int(float(ob.barTime))}"


# =============================
# Alerts Logic
# =============================
def format_fvg_msg(symbol: str, tf: str, g: FVG, last_price: float) -> str:
    side = "BULLISH" if g.bias == BULLISH else "BEARISH"
    lo = float(min(g.top, g.bottom))
    hi = float(max(g.top, g.bottom))
    return (
        f"{symbol} | {tf}\n"
        f"FVG: {side}\n"
        f"Диапазон: {lo:.2f} - {hi:.2f}\n"
        f"LeftTime: {ts_utc_str(g.leftTime)}\n"
        f"Цена сейчас: {last_price:.2f}"
    )


def format_bos_msg(symbol: str, tf: str, ev: Dict[str, Any], last_price: float) -> str:
    direction = "UP" if ev.get("dir") == "UP" else "DOWN"
    level = float(ev.get("level", 0.0))
    t1 = float(ev.get("t1", 0.0))
    return (
        f"{symbol} | {tf}\n"
        f"BOS: {direction}\n"
        f"Уровень: {level:.2f}\n"
        f"Время пробоя: {ts_utc_str(t1)}\n"
        f"Цена сейчас: {last_price:.2f}"
    )


def format_ob_break_msg(symbol: str, tf: str, ob: OrderBlock, last_price: float, bar_time: float) -> str:
    side = "BULLISH" if ob.bias == BULLISH else "BEARISH"
    lo = float(min(ob.barLow, ob.barHigh))
    hi = float(max(ob.barLow, ob.barHigh))
    return (
        f"{symbol} | {tf}\n"
        f"ORDER BLOCK BROKEN: {side}\n"
        f"OB диапазон: {lo:.2f} - {hi:.2f}\n"
        f"OB время: {ts_utc_str(ob.barTime)}\n"
        f"Сработало на баре: {ts_utc_str(bar_time)}\n"
        f"Цена сейчас: {last_price:.2f}"
    )


# =============================
# Main loop
# =============================
def run(symbol: str, tf: str) -> None:
    bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    q: Queue = Queue()

    # load history
    df = fetch_klines_paged(symbol, tf, HIST_TARGET_BARS)
    df4h = fetch_klines_paged(symbol, "4h", max(3000, HIST_TARGET_BARS // 3))

    overlays = compute_overlays(df, df4h)

    last_price = float(df["close"].iloc[-1]) if len(df) else 0.0

    # state load
    st = load_state(STATE_FILE)
    sent_fvg = set(st.get("fvg_keys", []))
    sent_bos = set(st.get("bos_keys", []))

    # чтобы при первом запуске не слать старые события:
    # синхронизируем "sent" с текущими найденными структурами
    cur_fvg_keys = set(fvg_key(g) for g in overlays.get("fvgs", []))
    cur_bos_keys = set(bos_key(ev) for ev in overlays.get("swing_struct", []) if ev.get("tag") == BOS)

    if not sent_fvg:
        sent_fvg = set(cur_fvg_keys)
    if not sent_bos:
        sent_bos = set(cur_bos_keys)

    # tracked OBs (активные, как на графике: INTERNAL_OB_COUNT)
    tracked_obs: Dict[str, OrderBlock] = {}
    for ob in overlays.get("internal_obs", []):
        k = ob_key(ob)
        tracked_obs[k] = ob

    # ws
    ws_tf = KlineWS(symbol, tf, q, "tf")
    ws_4h = KlineWS(symbol, "4h", q, "4h")
    ws_tf.start()
    ws_4h.start()

    print(f"RUN: {symbol} tf={tf} | bars={len(df)} | fvg={len(cur_fvg_keys)} | bos={len(cur_bos_keys)}")

    last_state_save_ts = 0.0

    try:
        while True:
            changed_closed = False
            changed_4h_closed = False

            # drain queue (как в вашем on_timer)
            try:
                while True:
                    msg = q.get_nowait()

                    if msg["type"] == "status":
                        # можно печатать, но без спама
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
                            if len(df4h) > 0:
                                last_t = float(df4h["time"].iloc[-1])
                                if abs(tt - last_t) < 1e-9:
                                    df4h.loc[len(df4h) - 1, ["open", "high", "low", "close", "volume"]] = [oo, hh, ll, cc, vv]
                                elif tt > last_t:
                                    row = {"time": tt, "open": oo, "high": hh, "low": ll, "close": cc, "volume": vv}
                                    df4h = pd.concat([df4h, pd.DataFrame([row])], ignore_index=True)
                                    if len(df4h) > 4500:
                                        df4h = df4h.iloc[-4500:].reset_index(drop=True)
                            changed_4h_closed = changed_4h_closed or closed
                            continue

                        # tf stream
                        if len(df) == 0:
                            continue
                        last_t = float(df["time"].iloc[-1])
                        if abs(tt - last_t) < 1e-9:
                            df.loc[len(df) - 1, ["open", "high", "low", "close", "volume"]] = [oo, hh, ll, cc, vv]
                        elif tt > last_t:
                            row = {"time": tt, "open": oo, "high": hh, "low": ll, "close": cc, "volume": vv}
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                            if len(df) > HIST_TARGET_BARS:
                                df = df.iloc[-HIST_TARGET_BARS:].reset_index(drop=True)

                        changed_closed = changed_closed or closed

            except Empty:
                pass

            if changed_closed or changed_4h_closed:
                if len(df) < max(600, SWINGS_LEN * 6):
                    continue

                # текущий бар (закрытый) для проверки OB break
                last_bar_high = float(df["high"].iloc[-1])
                last_bar_low = float(df["low"].iloc[-1])
                last_bar_time = float(df["time"].iloc[-1])
                last_price = float(df["close"].iloc[-1])

                # 1) OrderBlock break (по ранее отслеживаемым OB)
                # bearish OB broken, если high > ob.barHigh
                # bullish OB broken, если low < ob.barLow
                broken_keys: List[str] = []
                for k_ob, ob in tracked_obs.items():
                    if ob.bias == BEARISH:
                        if last_bar_high > float(ob.barHigh):
                            ok = bot.send(format_ob_break_msg(symbol, tf, ob, last_price, last_bar_time))
                            if ok:
                                broken_keys.append(k_ob)
                    else:
                        if last_bar_low < float(ob.barLow):
                            ok = bot.send(format_ob_break_msg(symbol, tf, ob, last_price, last_bar_time))
                            if ok:
                                broken_keys.append(k_ob)
                for bk in broken_keys:
                    tracked_obs.pop(bk, None)

                # 2) Recalc overlays (как у вас после закрытия свечи)
                overlays = compute_overlays(df, df4h)

                # 3) New FVG
                new_fvgs: List[FVG] = overlays.get("fvgs", [])
                for g in new_fvgs:
                    kf = fvg_key(g)
                    if kf not in sent_fvg:
                        if bot.send(format_fvg_msg(symbol, tf, g, last_price)):
                            sent_fvg.add(kf)

                # 4) New BOS (только BOS, не CHoCH)
                for ev in overlays.get("swing_struct", []):
                    if ev.get("tag") != BOS:
                        continue
                    kb = bos_key(ev)
                    if kb not in sent_bos:
                        if bot.send(format_bos_msg(symbol, tf, ev, last_price)):
                            sent_bos.add(kb)

                # 5) обновить tracked OBs (активные internal_obs)
                # добавляем новые, сохраняя уже имеющиеся (если не сломались)
                cur_internal_obs: List[OrderBlock] = overlays.get("internal_obs", [])
                new_tracked: Dict[str, OrderBlock] = {}
                for ob in cur_internal_obs:
                    k = ob_key(ob)
                    new_tracked[k] = ob

                # сохраняем только те, что сейчас активны по overlays (как на графике)
                tracked_obs = {k: v for k, v in tracked_obs.items() if k in new_tracked}
                for k, v in new_tracked.items():
                    if k not in tracked_obs:
                        tracked_obs[k] = v

                # 6) периодически сохраняем state
                now = time.time()
                if now - last_state_save_ts > 2.0:
                    st_out = {
                        "fvg_keys": sorted(list(sent_fvg))[-5000:],
                        "bos_keys": sorted(list(sent_bos))[-5000:],
                        "ob_keys": sorted(list(tracked_obs.keys()))[-200:],
                    }
                    save_state(STATE_FILE, st_out)
                    last_state_save_ts = now

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ws_tf.stop()
            ws_4h.stop()
        except Exception:
            pass

        st_out = {
            "fvg_keys": sorted(list(sent_fvg))[-5000:],
            "bos_keys": sorted(list(sent_bos))[-5000:],
            "ob_keys": sorted(list(tracked_obs.keys()))[-200:],
        }
        save_state(STATE_FILE, st_out)


def main():
    symbol = SYMBOL
    tf = DEFAULT_TF

    if len(sys.argv) >= 2:
        symbol = str(sys.argv[1]).upper().strip()
    if len(sys.argv) >= 3:
        tf = str(sys.argv[2]).strip()

    if tf not in TF_LIST:
        print(f"TF должен быть один из: {TF_LIST}")
        sys.exit(2)

    run(symbol, tf)


if __name__ == "__main__":
    main()