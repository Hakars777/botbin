"""Android foreground service: monitors signals (OBF / BOS-range / FVG->OB) in background."""
import os, time, threading

os.environ["KIVY_NO_FILELOG"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "0"

from engine import (
    SYMBOL, DEFAULT_TF, HIST_BARS,
    HAS_WS, interval_seconds, fetch_klines, fetch_latest,
    compute_overlays, SignalTracker, KlineWS, empty_data
)
from queue import Queue, Empty

POLL_SEC = 15
CHANNEL_ID = "all_signal_alerts"


def setup_foreground():
    try:
        from jnius import autoclass
        PythonService = autoclass("org.kivy.android.PythonService")
        PythonService.mService.setAutoRestartService(True)
        service = PythonService.mService
        Context = autoclass("android.content.Context")
        nm = service.getSystemService(Context.NOTIFICATION_SERVICE)

        Build = autoclass("android.os.Build")
        if Build.VERSION.SDK_INT >= 26:
            NotifChannel = autoclass("android.app.NotificationChannel")
            ch = NotifChannel(CHANNEL_ID, "Signal Alerts", nm.IMPORTANCE_LOW)
            nm.createNotificationChannel(ch)

        Builder = autoclass("android.app.Notification$Builder")
        if Build.VERSION.SDK_INT >= 26:
            b = Builder(service, CHANNEL_ID)
        else:
            b = Builder(service)
        b.setContentTitle("All Signal Monitor")
        b.setContentText("Monitoring signals...")
        b.setSmallIcon(service.getApplicationInfo().icon)
        service.startForeground(1, b.build())
        return nm, service
    except Exception:
        return None, None


def send_notification(nm, service, title, message):
    if nm is None:
        return
    try:
        from jnius import autoclass
        Build = autoclass("android.os.Build")
        Builder = autoclass("android.app.Notification$Builder")
        if Build.VERSION.SDK_INT >= 26:
            b = Builder(service, CHANNEL_ID)
        else:
            b = Builder(service)
        b.setContentTitle(title)
        b.setContentText(message)
        b.setSmallIcon(service.getApplicationInfo().icon)
        b.setAutoCancel(True)
        nid = int(time.time()) % 100000 + 100
        nm.notify(nid, b.build())
    except Exception:
        pass


def monitor_loop():
    nm, service = setup_foreground()
    tracker = SignalTracker()
    q = Queue()
    ws_tf = ws_4h = None

    try:
        data = fetch_klines(SYMBOL, DEFAULT_TF, HIST_BARS)
        data_4h = fetch_klines(SYMBOL, "4h", max(1000, HIST_BARS // 3))
    except Exception:
        data = empty_data()
        data_4h = empty_data()

    ov = compute_overlays(data, data_4h)
    tracker.check(ov)

    if HAS_WS:
        try:
            ws_tf = KlineWS(SYMBOL, DEFAULT_TF, q, "tf"); ws_tf.start()
            ws_4h = KlineWS(SYMBOL, "4h", q, "4h"); ws_4h.start()
        except Exception:
            pass

    import numpy as np
    while True:
        time.sleep(POLL_SEC)
        try:
            rows = fetch_latest(SYMBOL, DEFAULT_TF, 3)
            rows_4h = fetch_latest(SYMBOL, "4h", 3)
            new_candle = False
            for row in rows:
                tt = row["t"]
                vals = [row["o"], row["h"], row["l"], row["c"], row["v"]]
                if len(data["time"]) == 0:
                    continue
                last_t = float(data["time"][-1])
                if abs(tt - last_t) < 1e-9:
                    for i, k in enumerate(("open","high","low","close","volume")):
                        data[k][-1] = vals[i]
                elif tt > last_t:
                    data["time"] = np.append(data["time"], tt)
                    for i, k in enumerate(("open","high","low","close","volume")):
                        data[k] = np.append(data[k], vals[i])
                    if len(data["time"]) > HIST_BARS:
                        for k in data: data[k] = data[k][-HIST_BARS:]
                    new_candle = True

            for row in rows_4h:
                tt = row["t"]
                vals = [row["o"], row["h"], row["l"], row["c"], row["v"]]
                if len(data_4h["time"]) == 0:
                    continue
                last_t = float(data_4h["time"][-1])
                if abs(tt - last_t) < 1e-9:
                    for i, k in enumerate(("open","high","low","close","volume")):
                        data_4h[k][-1] = vals[i]
                elif tt > last_t:
                    data_4h["time"] = np.append(data_4h["time"], tt)
                    for i, k in enumerate(("open","high","low","close","volume")):
                        data_4h[k] = np.append(data_4h[k], vals[i])
                    if len(data_4h["time"]) > 4500:
                        for k in data_4h: data_4h[k] = data_4h[k][-4500:]
                    new_candle = True

            if new_candle:
                ov = compute_overlays(data, data_4h)
                alerts = tracker.check(ov)
                for title, msg in alerts:
                    send_notification(nm, service, f"Signal: {title}", msg)
        except Exception:
            pass


if __name__ == "__main__":
    monitor_loop()
