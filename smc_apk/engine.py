"""Shared SMC computation engine (no Kivy dependency)."""
import os
import json
import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty

import numpy as np
import requests

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

try:
    from websocket import WebSocketApp
    HAS_WS = True
except ImportError:
    HAS_WS = False

SYMBOL = "BTCUSDT"
TF_LIST = ["15m", "4h", "1d"]
DEFAULT_TF = "15m"
HIST_BARS = 1000

BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = 1
BEARISH = -1
BOS_TAG = "BOS"
CHOCH_TAG = "CHoCH"

SWINGS_LEN = 50
INTERNAL_LEN = 5
INTERNAL_OB_COUNT = 2
ATR_LEN = 200
HIGH_VOL_MULT = 2.0
FVG_EXTEND = 5

BINANCE_REST = "https://api.binance.com"
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


def interval_seconds(tf):
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    return 60


def ms_to_s(ms):
    return ms / 1000.0


def guess_base_tf_sec(t):
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


def empty_data():
    return {k: np.array([], dtype=float) for k in ("time", "open", "high", "low", "close", "volume")}


@dataclass
class Pivot:
    currentLevel: float = None
    lastLevel: float = None
    crossed: bool = False
    barTime: float = None
    barIndex: int = None


@dataclass
class Trend:
    bias: int = 0


@dataclass
class Trailing:
    top: float = None
    bottom: float = None
    barTime: float = None
    barIndex: int = None
    lastTopTime: float = None
    lastBottomTime: float = None


@dataclass
class OrderBlock:
    barHigh: float
    barLow: float
    barTime: float
    bias: int


@dataclass
class FVGap:
    top: float
    bottom: float
    bias: int
    leftTime: float
    rightTime: float


def fetch_klines(symbol, tf, target_bars):
    limit = 1000
    all_data = []
    end_time = None
    while len(all_data) < target_bars:
        params = {"symbol": symbol, "interval": tf, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(f"{BINANCE_REST}/api/v3/klines", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_data = data + all_data
        end_time = int(data[0][0]) - 1
        if len(data) < limit:
            break
        time.sleep(0.05)
    all_data = all_data[-target_bars:]
    if not all_data:
        return empty_data()
    return {
        "time": np.array([ms_to_s(int(k[0])) for k in all_data], dtype=float),
        "open": np.array([float(k[1]) for k in all_data], dtype=float),
        "high": np.array([float(k[2]) for k in all_data], dtype=float),
        "low": np.array([float(k[3]) for k in all_data], dtype=float),
        "close": np.array([float(k[4]) for k in all_data], dtype=float),
        "volume": np.array([float(k[5]) for k in all_data], dtype=float),
    }


def fetch_latest(symbol, tf, count=3):
    try:
        r = requests.get(f"{BINANCE_REST}/api/v3/klines",
                         params={"symbol": symbol, "interval": tf, "limit": count},
                         timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return []
        return [{"t": ms_to_s(int(k[0])),
                 "o": float(k[1]), "h": float(k[2]),
                 "l": float(k[3]), "c": float(k[4]),
                 "v": float(k[5])} for k in rows]
    except Exception:
        return []


def true_range(h, l, pc):
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr_wilder(high, low, close, n):
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = true_range(high, low, pc)
    atr = np.zeros_like(tr); atr[0] = tr[0]
    a = 1.0 / float(n)
    for i in range(1, len(tr)):
        atr[i] = (1 - a) * atr[i - 1] + a * tr[i]
    return atr


def parsed_hilo(high, low, atr):
    hv = (high - low) >= (HIGH_VOL_MULT * atr)
    return np.where(hv, low, high), np.where(hv, high, low)


def leg_at(i, size, high, low, prev):
    if i < size:
        return prev
    ch, cl = high[i - size], low[i - size]
    wh = np.max(high[i - size + 1:i + 1])
    wl = np.min(low[i - size + 1:i + 1])
    if ch > wh:
        return BEARISH_LEG
    if cl < wl:
        return BULLISH_LEG
    return prev


def crossover(p, c, lv):
    return p <= lv and c > lv


def crossunder(p, c, lv):
    return p >= lv and c < lv


def merge_obs(obs):
    if not obs:
        return []
    by = {BULLISH: [], BEARISH: []}
    for ob in obs:
        lo, hi = min(ob.barLow, ob.barHigh), max(ob.barLow, ob.barHigh)
        by[ob.bias].append(OrderBlock(hi, lo, ob.barTime, ob.bias))
    merged = []
    for bias, lst in by.items():
        if not lst:
            continue
        lst.sort(key=lambda x: x.barLow)
        cur = lst[0]
        for ob in lst[1:]:
            if ob.barLow <= cur.barHigh + 1e-12:
                cur.barHigh = max(cur.barHigh, ob.barHigh)
                cur.barLow = min(cur.barLow, ob.barLow)
                cur.barTime = min(cur.barTime, ob.barTime)
            else:
                merged.append(cur)
                cur = ob
        merged.append(cur)
    merged.sort(key=lambda x: x.barTime, reverse=True)
    return merged


def compute_overlays(data, data_4h):
    t, h, l, c = data["time"], data["high"], data["low"], data["close"]
    n = len(t)
    if n < SWINGS_LEN:
        return {"swing_struct": [], "strongweak": None, "internal_obs": [], "fvgs": []}

    base_sec = guess_base_tf_sec(t)
    atr = atr_wilder(h, l, c, ATR_LEN)
    pH, pL = parsed_hilo(h, l, atr)

    sH, sL = Pivot(), Pivot()
    iH, iL = Pivot(), Pivot()
    sTr, iTr = Trend(0), Trend(0)
    trail = Trailing()
    iobs = []
    sevts = []

    ls = np.zeros(n, dtype=int)
    li = np.zeros(n, dtype=int)
    cs, ci = 0, 0
    for i in range(n):
        cs = leg_at(i, SWINGS_LEN, h, l, cs)
        ci = leg_at(i, INTERNAL_LEN, h, l, ci)
        ls[i], li[i] = cs, ci

    def _store_ob(piv, bias, ci_):
        if piv.barIndex is None:
            return
        a, b = int(piv.barIndex), int(ci_)
        if b <= a or a < 0 or b > n:
            return
        seg = pH[a:b] if bias == BEARISH else pL[a:b]
        if seg.size == 0:
            return
        idx = a + (int(np.argmax(seg)) if bias == BEARISH else int(np.argmin(seg)))
        iobs.insert(0, OrderBlock(float(pH[idx]), float(pL[idx]), float(t[idx]), bias))
        if len(iobs) > 100:
            iobs.pop()

    def _mitigate(ci_):
        cH, cL = float(h[ci_]), float(l[ci_])
        iobs[:] = [ob for ob in iobs
                    if not (ob.bias == BEARISH and cH > ob.barHigh)
                    and not (ob.bias == BULLISH and cL < ob.barLow)]

    def _piv_upd(i, sz, internal):
        if i <= 0:
            return
        pi = i - sz
        if pi < 0:
            return
        arr = li if internal else ls
        ch = int(arr[i]) - int(arr[i - 1])
        if ch == 0:
            return
        if ch == 1:
            p = iL if internal else sL
            p.lastLevel, p.currentLevel = p.currentLevel, float(l[pi])
            p.crossed, p.barTime, p.barIndex = False, float(t[pi]), pi
            if not internal:
                trail.bottom = p.currentLevel
                trail.barTime, trail.barIndex = p.barTime, p.barIndex
                trail.lastBottomTime = p.barTime
        elif ch == -1:
            p = iH if internal else sH
            p.lastLevel, p.currentLevel = p.currentLevel, float(h[pi])
            p.crossed, p.barTime, p.barIndex = False, float(t[pi]), pi
            if not internal:
                trail.top = p.currentLevel
                trail.barTime, trail.barIndex = p.barTime, p.barIndex
                trail.lastTopTime = p.barTime

    for i in range(n):
        if trail.top is not None and trail.bottom is not None:
            hi_, lo_ = float(h[i]), float(l[i])
            if hi_ >= trail.top:
                trail.top, trail.lastTopTime = hi_, float(t[i])
            if lo_ <= trail.bottom:
                trail.bottom, trail.lastBottomTime = lo_, float(t[i])

        _piv_upd(i, SWINGS_LEN, False)
        _piv_upd(i, INTERNAL_LEN, True)

        pc = float(c[i - 1] if i > 0 else c[i])
        cc = float(c[i])

        if iH.currentLevel is not None and not iH.crossed:
            if sH.currentLevel is not None and iH.currentLevel != sH.currentLevel:
                if crossover(pc, cc, float(iH.currentLevel)):
                    iH.crossed = True; iTr.bias = BULLISH; _store_ob(iH, BULLISH, i)

        if iL.currentLevel is not None and not iL.crossed:
            if sL.currentLevel is not None and iL.currentLevel != sL.currentLevel:
                if crossunder(pc, cc, float(iL.currentLevel)):
                    iL.crossed = True; iTr.bias = BEARISH; _store_ob(iL, BEARISH, i)

        if sH.currentLevel is not None and sH.barIndex is not None and not sH.crossed:
            if i > 0 and crossover(float(c[i - 1]), cc, float(sH.currentLevel)):
                tag = CHOCH_TAG if sTr.bias == BEARISH else BOS_TAG
                sH.crossed = True; sTr.bias = BULLISH
                mid = max(0, min(n - 1, int(round(0.5 * (sH.barIndex + i)))))
                sevts.append({"tag": tag, "level": float(sH.currentLevel),
                              "t0": float(sH.barTime or t[sH.barIndex]),
                              "t1": float(t[i]), "tmid": float(t[mid]), "dir": "UP"})

        if sL.currentLevel is not None and sL.barIndex is not None and not sL.crossed:
            if i > 0 and crossunder(float(c[i - 1]), cc, float(sL.currentLevel)):
                tag = CHOCH_TAG if sTr.bias == BULLISH else BOS_TAG
                sL.crossed = True; sTr.bias = BEARISH
                mid = max(0, min(n - 1, int(round(0.5 * (sL.barIndex + i)))))
                sevts.append({"tag": tag, "level": float(sL.currentLevel),
                              "t0": float(sL.barTime or t[sL.barIndex]),
                              "t1": float(t[i]), "tmid": float(t[mid]), "dir": "DOWN"})

        if i >= 1:
            _mitigate(i)

    sw = None
    if (trail.top is not None and trail.bottom is not None
            and trail.lastTopTime is not None and trail.lastBottomTime is not None):
        sw = {
            "top": float(trail.top), "bottom": float(trail.bottom),
            "top_text": "Strong High" if sTr.bias == BEARISH else "Weak High",
            "bottom_text": "Strong Low" if sTr.bias == BULLISH else "Weak Low",
            "lastTopTime": float(trail.lastTopTime),
            "lastBottomTime": float(trail.lastBottomTime),
        }

    fvgs = []
    if data_4h is not None and len(data_4h["time"]) >= 5:
        t4, o4, h4, l4, c4 = (data_4h[k] for k in ("time", "open", "high", "low", "close"))
        ext = float(FVG_EXTEND * base_sec)
        tmp = []
        cum = 0.0
        pj = -999999
        for i in range(n):
            cL, cH = float(l[i]), float(h[i])
            tmp[:] = [g for g in tmp
                      if not (g.bias == BULLISH and cL < g.bottom)
                      and not (g.bias == BEARISH and cH > g.top)]
            j = int(np.searchsorted(t4, t[i], side="right") - 1)
            if j < 0:
                continue
            if j != pj and j >= 2:
                lC, lO, lT = float(c4[j - 1]), float(o4[j - 1]), float(t4[j - 1])
                cHi, cLo, cT = float(h4[j]), float(l4[j]), float(t4[j])
                l2H, l2L = float(h4[j - 2]), float(l4[j - 2])
                dn = lO * 100.0
                bdp = ((lC - lO) / dn) if abs(dn) > 1e-12 else 0.0
                cum += abs(bdp)
                thr = (cum / max(1.0, float(i))) * 2.0
                if cLo > l2H and lC > l2H and bdp > thr:
                    tmp.insert(0, FVGap(cLo, l2H, BULLISH, lT, cT + ext))
                if cHi < l2L and lC < l2L and (-bdp) > thr:
                    tmp.insert(0, FVGap(cHi, l2L, BEARISH, lT, cT + ext))
                if len(tmp) > 500:
                    tmp = tmp[:500]
            pj = j
        fvgs = tmp[:200]

    m = merge_obs(iobs)
    return {"swing_struct": sevts[-250:], "strongweak": sw,
            "internal_obs": m[:INTERNAL_OB_COUNT], "fvgs": fvgs}


class AlertTracker:
    def __init__(self):
        self.prev_bos = set()
        self.prev_obs = set()
        self.prev_fvg = set()
        self.ready = False

    @staticmethod
    def _bos_key(ev):
        return f"{ev['dir']}|{ev['level']:.2f}|{int(ev['t1'])}"

    @staticmethod
    def _ob_key(ob):
        return f"{ob.bias}|{ob.barTime:.0f}|{ob.barHigh:.2f}|{ob.barLow:.2f}"

    @staticmethod
    def _fvg_key(g):
        return f"{g.bias}|{g.leftTime:.0f}|{g.top:.2f}|{g.bottom:.2f}"

    def check(self, ov):
        alerts = []
        cur_bos = {self._bos_key(e) for e in ov.get("swing_struct", []) if e["tag"] == BOS_TAG}
        cur_obs = {self._ob_key(ob) for ob in ov.get("internal_obs", [])}
        cur_fvg = {self._fvg_key(g) for g in ov.get("fvgs", [])}

        if self.ready:
            for k in cur_bos - self.prev_bos:
                p = k.split("|")
                d = "Bullish" if p[0] == "UP" else "Bearish"
                alerts.append(("BOS", f"{SYMBOL} {d} BOS @ {p[1]}"))
            for k in self.prev_obs - cur_obs:
                p = k.split("|")
                b = "Bull" if p[0] == "1" else "Bear"
                alerts.append(("OB Mitigation", f"{SYMBOL} {b} OB mitigated ({p[3]}-{p[2]})"))
            for k in cur_fvg - self.prev_fvg:
                p = k.split("|")
                b = "Bullish" if p[0] == "1" else "Bearish"
                alerts.append(("FVG", f"{SYMBOL} {b} FVG ({p[3]}-{p[2]})"))

        self.prev_bos, self.prev_obs, self.prev_fvg = cur_bos, cur_obs, cur_fvg
        if not self.ready:
            self.ready = True
        return alerts


class KlineWS:
    def __init__(self, symbol, tf, q, tag):
        self.sym = symbol.lower()
        self.tf = tf
        self.q = q
        self.tag = tag
        self.ws = None
        self.stop_evt = threading.Event()

    def start(self):
        self.stop_evt.clear()
        url = f"{BINANCE_WS_BASE}/{self.sym}@kline_{self.tf}"
        self.ws = WebSocketApp(url, on_message=self._msg,
                               on_open=lambda w: None, on_close=lambda w, *a: None,
                               on_error=lambda w, e: None)
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    def _run(self):
        while not self.stop_evt.is_set():
            try:
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            time.sleep(1.0)

    def _msg(self, ws, message):
        try:
            k = json.loads(message).get("k", {})
            if not k:
                return
            self.q.put({"type": "kline", "tag": self.tag,
                        "t": ms_to_s(int(k["t"])),
                        "o": float(k["o"]), "h": float(k["h"]),
                        "l": float(k["l"]), "c": float(k["c"]),
                        "v": float(k["v"]), "closed": bool(k["x"])})
        except Exception:
            pass
