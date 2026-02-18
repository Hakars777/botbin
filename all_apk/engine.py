"""all.py engine ported to numpy (no pandas/PySide6)."""
import os, json, time, threading
from dataclasses import dataclass
from queue import Queue, Empty
import numpy as np, requests

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
HIST_BARS = 2000

BULLISH_LEG = 1; BEARISH_LEG = 0; BULLISH = 1; BEARISH = -1
BOS_TAG = "BOS"; CHOCH_TAG = "CHoCH"
SWINGS_LEN = 50; INTERNAL_LEN = 5; INTERNAL_OB_COUNT = 2
ATR_LEN = 200; HIGH_VOL_MULT = 2.0; FVG_EXTEND = 5

BINANCE_REST = "https://api.binance.com"
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


def interval_seconds(tf):
    if tf.endswith("m"): return int(tf[:-1]) * 60
    if tf.endswith("h"): return int(tf[:-1]) * 3600
    if tf.endswith("d"): return int(tf[:-1]) * 86400
    return 60

def ms_to_s(ms): return ms / 1000.0

def guess_base_tf_sec(t):
    if t.size < 3: return 60
    d = np.diff(t); d = d[np.isfinite(d)]
    if d.size == 0: return 60
    med = float(np.median(d))
    return max(1, int(round(med))) if np.isfinite(med) and med > 0 else 60

def empty_data():
    return {k: np.array([], dtype=float) for k in ("time","open","high","low","close","volume")}


@dataclass
class Pivot:
    currentLevel: float = None; lastLevel: float = None; crossed: bool = False
    barTime: float = None; barIndex: int = None

@dataclass
class Trend:
    bias: int = 0

@dataclass
class Trailing:
    top: float = None; bottom: float = None; barTime: float = None; barIndex: int = None
    lastTopTime: float = None; lastBottomTime: float = None

@dataclass
class OrderBlock:
    barHigh: float; barLow: float; barTime: float; bias: int

@dataclass
class FVGap:
    top: float; bottom: float; bias: int; leftTime: float; rightTime: float; createTime: float


def fetch_klines(symbol, tf, target_bars):
    limit = 1000; all_data = []; end_time = None
    while len(all_data) < target_bars:
        params = {"symbol": symbol, "interval": tf, "limit": limit}
        if end_time is not None: params["endTime"] = end_time
        r = requests.get(f"{BINANCE_REST}/api/v3/klines", params=params, timeout=20)
        r.raise_for_status(); data = r.json()
        if not data: break
        all_data = data + all_data; end_time = int(data[0][0]) - 1
        if len(data) < limit: break
        time.sleep(0.05)
    all_data = all_data[-target_bars:]
    if not all_data: return empty_data()
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
                         params={"symbol": symbol, "interval": tf, "limit": count}, timeout=10)
        r.raise_for_status()
        return [{"t": ms_to_s(int(k[0])), "o": float(k[1]), "h": float(k[2]),
                 "l": float(k[3]), "c": float(k[4]), "v": float(k[5])} for k in r.json()]
    except Exception:
        return []


def true_range(h, l, pc):
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))

def atr_wilder(high, low, close, n):
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = true_range(high, low, pc)
    atr = np.zeros_like(tr); atr[0] = tr[0]; a = 1.0 / float(n)
    for i in range(1, len(tr)): atr[i] = (1 - a) * atr[i-1] + a * tr[i]
    return atr

def parsed_hilo(high, low, atr):
    hv = (high - low) >= (HIGH_VOL_MULT * atr)
    return np.where(hv, low, high), np.where(hv, high, low)

def leg_at(i, size, high, low, prev):
    if i < size: return prev
    ch, cl = high[i - size], low[i - size]
    wh = np.max(high[i - size + 1:i + 1]); wl = np.min(low[i - size + 1:i + 1])
    if ch > wh: return BEARISH_LEG
    if cl < wl: return BULLISH_LEG
    return prev

def crossover(p, c, lv): return p <= lv and c > lv
def crossunder(p, c, lv): return p >= lv and c < lv

def _ob_key(ob):
    return (int(round(float(ob.barTime))), int(ob.bias),
            round(float(min(ob.barLow, ob.barHigh)), 8),
            round(float(max(ob.barLow, ob.barHigh)), 8))

def _ob_range(ob):
    return float(min(ob.barLow, ob.barHigh)), float(max(ob.barLow, ob.barHigh))

def _ranges_overlap(a_lo, a_hi, b_lo, b_hi):
    ov_lo = max(a_lo, b_lo); ov_hi = min(a_hi, b_hi)
    return (ov_hi >= ov_lo - 1e-12), ov_lo, ov_hi


def compute_overlays(data, data_4h):
    t, h, l, c = data["time"], data["high"], data["low"], data["close"]
    n = len(t)
    empty = {"swing_struct": [], "strongweak": None, "internal_obs": [],
             "fvgs": [], "ob_flip_signals": [], "bos_range_signals": [], "fvg_ob_signals": []}
    if n < SWINGS_LEN:
        return empty

    base_sec = guess_base_tf_sec(t)
    atr = atr_wilder(h, l, c, ATR_LEN)
    pH, pL = parsed_hilo(h, l, atr)

    sH, sL = Pivot(), Pivot()
    iH, iL = Pivot(), Pivot()
    sTr, iTr = Trend(0), Trend(0)
    trail = Trailing()
    iobs = []; sevts = []

    mitigated_keys = set()
    ob_flip_signals = []
    last_created_ob = None

    bos_ranges = []
    bos_range_signals = []
    fvg_ob_signals = []

    ls = np.zeros(n, dtype=int); li = np.zeros(n, dtype=int)
    cs, ci = 0, 0
    for i in range(n):
        cs = leg_at(i, SWINGS_LEN, h, l, cs)
        ci = leg_at(i, INTERNAL_LEN, h, l, ci)
        ls[i], li[i] = cs, ci

    def _store_ob(piv, bias, ci_):
        if piv.barIndex is None: return None
        a, b = int(piv.barIndex), int(ci_)
        if b <= a or a < 0 or b > n: return None
        seg = pH[a:b] if bias == BEARISH else pL[a:b]
        if seg.size == 0: return None
        idx = a + (int(np.argmax(seg)) if bias == BEARISH else int(np.argmin(seg)))
        ob = OrderBlock(float(pH[idx]), float(pL[idx]), float(t[idx]), bias)
        iobs.insert(0, ob)
        if len(iobs) > 100: iobs.pop()
        return ob

    def _mitigate(ci_):
        cH, cL = float(h[ci_]), float(l[ci_])
        kept = []
        for ob in iobs:
            if ob.bias == BEARISH and cH > ob.barHigh:
                mitigated_keys.add(_ob_key(ob)); continue
            if ob.bias == BULLISH and cL < ob.barLow:
                mitigated_keys.add(_ob_key(ob)); continue
            kept.append(ob)
        iobs[:] = kept

    def _piv_upd(i, sz, internal):
        if i <= 0: return
        pi = i - sz
        if pi < 0: return
        arr = li if internal else ls
        ch = int(arr[i]) - int(arr[i-1])
        if ch == 0: return
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

    def _pick_ob_for_bias(bias):
        for ob in iobs:
            if int(ob.bias) == int(bias): return ob
        return None

    def _add_bos_range(ob, bos_level, bias, bos_time):
        boundary = float(ob.barHigh) if bias == BULLISH else float(ob.barLow)
        lo_ = float(min(boundary, bos_level)); hi_ = float(max(boundary, bos_level))
        if hi_ <= lo_ + 1e-12: return
        bos_ranges.append({"lo": lo_, "hi": hi_, "bias": bias, "key": _ob_key(ob),
                           "bos_level": bos_level, "bos_time": bos_time})
        if len(bos_ranges) > 500: bos_ranges[:] = bos_ranges[-500:]

    def _prune_bos():
        if not bos_ranges or not mitigated_keys: return
        bos_ranges[:] = [r for r in bos_ranges if r.get("key") not in mitigated_keys]

    def _check_ob_vs_bos(new_ob, trend_bias, trig_time):
        o_lo, o_hi = _ob_range(new_ob)
        for r in bos_ranges:
            if int(new_ob.bias) != int(r["bias"]) or int(trend_bias) != int(r["bias"]): continue
            ok, ov_lo, ov_hi = _ranges_overlap(r["lo"], r["hi"], o_lo, o_hi)
            if ok:
                bos_range_signals.append({"x": trig_time, "y": 0.5*(ov_lo+ov_hi), "bias": r["bias"]})

    def _check_ob_vs_fvg(new_ob, trend_bias, trig_time):
        if int(new_ob.bias) != int(trend_bias): return
        o_lo, o_hi = _ob_range(new_ob)
        ob_time = float(new_ob.barTime)
        for g in tmp_fvgs:
            if int(g.bias) != int(new_ob.bias): continue
            if ob_time <= float(g.createTime) + 1e-12: continue
            g_lo, g_hi = float(min(g.top, g.bottom)), float(max(g.top, g.bottom))
            ok, ov_lo, ov_hi = _ranges_overlap(o_lo, o_hi, g_lo, g_hi)
            if ok:
                fvg_ob_signals.append({"x": trig_time, "y": 0.5*(ov_lo+ov_hi), "bias": int(new_ob.bias)})

    # FVG state
    have_4h = data_4h is not None and len(data_4h["time"]) >= 5
    tmp_fvgs = []; cum_abs = 0.0; prev_j = -999999
    t4 = o4 = h4 = l4 = c4 = None
    ext = float(FVG_EXTEND * base_sec)
    if have_4h:
        t4, o4, h4, l4, c4 = (data_4h[k] for k in ("time","open","high","low","close"))

    def _update_fvgs(i):
        nonlocal cum_abs, prev_j
        if not have_4h: return
        cL_, cH_ = float(l[i]), float(h[i])
        tmp_fvgs[:] = [g for g in tmp_fvgs
                       if not (g.bias == BULLISH and cL_ < min(g.top, g.bottom))
                       and not (g.bias == BEARISH and cH_ > max(g.top, g.bottom))]
        j = int(np.searchsorted(t4, t[i], side="right") - 1)
        if j < 0: return
        if j != prev_j and j >= 2:
            lC, lO, lT = float(c4[j-1]), float(o4[j-1]), float(t4[j-1])
            cHi, cLo, cT = float(h4[j]), float(l4[j]), float(t4[j])
            l2H, l2L = float(h4[j-2]), float(l4[j-2])
            dn = lO * 100.0
            bdp = ((lC - lO) / dn) if abs(dn) > 1e-12 else 0.0
            cum_abs += abs(bdp)
            thr = (cum_abs / max(1.0, float(i))) * 2.0
            if cLo > l2H and lC > l2H and bdp > thr:
                tmp_fvgs.insert(0, FVGap(cLo, l2H, BULLISH, lT, cT + ext, cT))
            if cHi < l2L and lC < l2L and (-bdp) > thr:
                tmp_fvgs.insert(0, FVGap(cHi, l2L, BEARISH, lT, cT + ext, cT))
            if len(tmp_fvgs) > 800: tmp_fvgs[:] = tmp_fvgs[:800]
        prev_j = j

    for i in range(n):
        _prune_bos()
        _update_fvgs(i)

        if trail.top is not None and trail.bottom is not None:
            hi_, lo_ = float(h[i]), float(l[i])
            if hi_ >= trail.top: trail.top, trail.lastTopTime = hi_, float(t[i])
            if lo_ <= trail.bottom: trail.bottom, trail.lastBottomTime = lo_, float(t[i])

        _piv_upd(i, SWINGS_LEN, False)
        _piv_upd(i, INTERNAL_LEN, True)

        pc = float(c[i-1] if i > 0 else c[i]); cc = float(c[i])
        pending_pairs = []

        if iH.currentLevel is not None and not iH.crossed:
            if sH.currentLevel is not None and iH.currentLevel != sH.currentLevel:
                if crossover(pc, cc, float(iH.currentLevel)):
                    iH.crossed = True; iTr.bias = BULLISH
                    new_ob = _store_ob(iH, BULLISH, i)
                    if new_ob is not None:
                        pending_pairs.append((last_created_ob, new_ob))
                        last_created_ob = new_ob
                        _check_ob_vs_bos(new_ob, sTr.bias, float(t[i]))
                        _check_ob_vs_fvg(new_ob, sTr.bias, float(t[i]))

        if iL.currentLevel is not None and not iL.crossed:
            if sL.currentLevel is not None and iL.currentLevel != sL.currentLevel:
                if crossunder(pc, cc, float(iL.currentLevel)):
                    iL.crossed = True; iTr.bias = BEARISH
                    new_ob = _store_ob(iL, BEARISH, i)
                    if new_ob is not None:
                        pending_pairs.append((last_created_ob, new_ob))
                        last_created_ob = new_ob
                        _check_ob_vs_bos(new_ob, sTr.bias, float(t[i]))
                        _check_ob_vs_fvg(new_ob, sTr.bias, float(t[i]))

        if sH.currentLevel is not None and sH.barIndex is not None and not sH.crossed:
            if i > 0 and crossover(float(c[i-1]), cc, float(sH.currentLevel)):
                tag = CHOCH_TAG if sTr.bias == BEARISH else BOS_TAG
                sH.crossed = True; sTr.bias = BULLISH
                mid = max(0, min(n-1, int(round(0.5*(sH.barIndex + i)))))
                sevts.append({"tag": tag, "level": float(sH.currentLevel),
                              "t0": float(sH.barTime or t[sH.barIndex]),
                              "t1": float(t[i]), "tmid": float(t[mid]), "dir": "UP"})
                if tag == BOS_TAG:
                    src = _pick_ob_for_bias(BULLISH)
                    if src: _add_bos_range(src, float(sH.currentLevel), BULLISH, float(t[i]))

        if sL.currentLevel is not None and sL.barIndex is not None and not sL.crossed:
            if i > 0 and crossunder(float(c[i-1]), cc, float(sL.currentLevel)):
                tag = CHOCH_TAG if sTr.bias == BULLISH else BOS_TAG
                sL.crossed = True; sTr.bias = BEARISH
                mid = max(0, min(n-1, int(round(0.5*(sL.barIndex + i)))))
                sevts.append({"tag": tag, "level": float(sL.currentLevel),
                              "t0": float(sL.barTime or t[sL.barIndex]),
                              "t1": float(t[i]), "tmid": float(t[mid]), "dir": "DOWN"})
                if tag == BOS_TAG:
                    src = _pick_ob_for_bias(BEARISH)
                    if src: _add_bos_range(src, float(sL.currentLevel), BEARISH, float(t[i]))

        if i >= 1:
            _mitigate(i); _prune_bos()

        for prev_ob, new_ob in pending_pairs:
            if prev_ob is None: continue
            if _ob_key(prev_ob) not in mitigated_keys: continue
            if int(prev_ob.bias) == int(new_ob.bias): continue
            p_lo, p_hi = _ob_range(prev_ob); n_lo, n_hi = _ob_range(new_ob)
            ok, ov_lo, ov_hi = _ranges_overlap(p_lo, p_hi, n_lo, n_hi)
            if ok:
                y = 0.5*(ov_lo+ov_hi)
                x = float(max(prev_ob.barTime, new_ob.barTime))
                ob_flip_signals.append({"x": x, "y": y, "bias": int(new_ob.bias)})

    sw = None
    if (trail.top is not None and trail.bottom is not None
            and trail.lastTopTime is not None and trail.lastBottomTime is not None):
        sw = {"top": float(trail.top), "bottom": float(trail.bottom),
              "top_text": "Strong High" if sTr.bias == BEARISH else "Weak High",
              "bottom_text": "Strong Low" if sTr.bias == BULLISH else "Weak Low",
              "lastTopTime": float(trail.lastTopTime), "lastBottomTime": float(trail.lastBottomTime)}

    return {"swing_struct": sevts[-250:], "strongweak": sw,
            "internal_obs": iobs[:INTERNAL_OB_COUNT], "fvgs": tmp_fvgs[:200],
            "ob_flip_signals": ob_flip_signals[-250:],
            "bos_range_signals": bos_range_signals[-250:],
            "fvg_ob_signals": fvg_ob_signals[-250:]}


class SignalTracker:
    def __init__(self):
        self.prev_obf = set(); self.prev_bos = set(); self.prev_fvg = set()
        self.ready = False

    @staticmethod
    def _key(s): return f"{s['bias']}|{s['x']:.0f}|{s['y']:.2f}"

    def check(self, ov):
        alerts = []
        cur_obf = {self._key(s) for s in ov.get("ob_flip_signals", [])}
        cur_bos = {self._key(s) for s in ov.get("bos_range_signals", [])}
        cur_fvg = {self._key(s) for s in ov.get("fvg_ob_signals", [])}

        if self.ready:
            for k in cur_obf - self.prev_obf:
                p = k.split("|")
                d = "Bull" if p[0] == "1" else "Bear"
                alerts.append(("OB Flip", f"{SYMBOL} {d} OB Flip @ {p[2]}"))
            for k in cur_bos - self.prev_bos:
                p = k.split("|")
                d = "Bull" if p[0] == "1" else "Bear"
                alerts.append(("BOS Range", f"{SYMBOL} {d} BOS-Range signal @ {p[2]}"))
            for k in cur_fvg - self.prev_fvg:
                p = k.split("|")
                d = "Bull" if p[0] == "1" else "Bear"
                alerts.append(("FVG→OB", f"{SYMBOL} {d} FVG→OB signal @ {p[2]}"))

        self.prev_obf, self.prev_bos, self.prev_fvg = cur_obf, cur_bos, cur_fvg
        if not self.ready: self.ready = True
        return alerts


class KlineWS:
    def __init__(self, symbol, tf, q, tag):
        self.sym = symbol.lower(); self.tf = tf; self.q = q; self.tag = tag
        self.ws = None; self.stop_evt = threading.Event()

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
            if self.ws: self.ws.close()
        except Exception: pass

    def _run(self):
        while not self.stop_evt.is_set():
            try: self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception: pass
            time.sleep(1.0)

    def _msg(self, ws, message):
        try:
            k = json.loads(message).get("k", {})
            if not k: return
            self.q.put({"type": "kline", "tag": self.tag,
                        "t": ms_to_s(int(k["t"])), "o": float(k["o"]), "h": float(k["h"]),
                        "l": float(k["l"]), "c": float(k["c"]), "v": float(k["v"]),
                        "closed": bool(k["x"])})
        except Exception: pass
