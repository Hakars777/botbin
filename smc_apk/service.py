"""Android foreground service for SMC background notifications."""
import os
import sys
import time
import json
import threading
from queue import Queue, Empty

import numpy as np

from engine import (
    SYMBOL, DEFAULT_TF, HIST_BARS, HAS_WS,
    fetch_klines, compute_overlays, interval_seconds, ms_to_s,
    AlertTracker, KlineWS
)

IS_ANDROID = False
PythonService = None
try:
    from jnius import autoclass
    PythonService = autoclass("org.kivy.android.PythonService")
    IS_ANDROID = True
except Exception:
    pass

NOTIF_CHANNEL = "smc_alerts"
NOTIF_ID_BASE = 9000


def setup_foreground():
    if not IS_ANDROID or PythonService is None:
        return
    try:
        service = PythonService.mService
        service.setAutoRestartService(True)

        Context = autoclass("android.content.Context")
        NotifChannel = autoclass("android.app.NotificationChannel")
        NotifManager = autoclass("android.app.NotificationManager")
        NotifBuilder = autoclass("android.app.Notification$Builder")

        nm = service.getSystemService(Context.NOTIFICATION_SERVICE)
        channel = NotifChannel(NOTIF_CHANNEL, "SMC Trading Alerts",
                               NotifManager.IMPORTANCE_LOW)
        channel.setDescription("Background monitoring for SMC signals")
        nm.createNotificationChannel(channel)

        builder = NotifBuilder(service, NOTIF_CHANNEL)
        builder.setContentTitle("SMC Monitor")
        builder.setContentText("Monitoring BOS / OB / FVG signals...")
        builder.setSmallIcon(service.getApplicationInfo().icon)
        builder.setOngoing(True)
        service.startForeground(1, builder.build())
    except Exception as e:
        print(f"[service] foreground setup error: {e}")


def send_notification(title, message):
    if not IS_ANDROID or PythonService is None:
        print(f"[service] ALERT: {title} - {message}")
        return
    try:
        service = PythonService.mService
        Context = autoclass("android.content.Context")
        NotifBuilder = autoclass("android.app.Notification$Builder")
        NotifManager = autoclass("android.app.NotificationManager")

        nm = service.getSystemService(Context.NOTIFICATION_SERVICE)

        alert_channel_id = "smc_alerts_high"
        try:
            NotifChannel = autoclass("android.app.NotificationChannel")
            ch = NotifChannel(alert_channel_id, "SMC Signal Alerts",
                              NotifManager.IMPORTANCE_HIGH)
            ch.setDescription("High-priority SMC trading alerts")
            nm.createNotificationChannel(ch)
        except Exception:
            alert_channel_id = NOTIF_CHANNEL

        builder = NotifBuilder(service, alert_channel_id)
        builder.setContentTitle(f"SMC: {title}")
        builder.setContentText(message)
        builder.setSmallIcon(service.getApplicationInfo().icon)
        builder.setAutoCancel(True)

        try:
            BigTextStyle = autoclass("android.app.Notification$BigTextStyle")
            style = BigTextStyle()
            style.bigText(message)
            builder.setStyle(style)
        except Exception:
            pass

        nid = (NOTIF_ID_BASE + int(time.time())) % 100000
        nm.notify(nid, builder.build())
    except Exception as e:
        print(f"[service] notification error: {e}")


def monitor_loop():
    tracker = AlertTracker()
    q = Queue()
    ws_tf = None
    ws_4h = None
    data = None
    data_4h = None
    tf = DEFAULT_TF

    while True:
        try:
            print(f"[service] fetching klines {SYMBOL} {tf} ...")
            data = fetch_klines(SYMBOL, tf, HIST_BARS)
            data_4h = fetch_klines(SYMBOL, "4h", max(1000, HIST_BARS // 3))
            if len(data["time"]) > 0:
                ov = compute_overlays(data, data_4h)
                alerts = tracker.check(ov)
                for at, am in alerts:
                    send_notification(at, am)
                print(f"[service] loaded {len(data['time'])} bars, {len(alerts)} alerts")
            break
        except Exception as e:
            print(f"[service] fetch error: {e}, retrying in 10s")
            time.sleep(10)

    if HAS_WS:
        try:
            ws_tf = KlineWS(SYMBOL, tf, q, "tf")
            ws_tf.start()
            ws_4h = KlineWS(SYMBOL, "4h", q, "4h")
            ws_4h.start()
            print("[service] websockets started")
        except Exception as e:
            print(f"[service] ws start error: {e}")

    last_recompute = time.time()
    while True:
        try:
            tf_closed = False
            h4_closed = False
            try:
                while True:
                    msg = q.get(timeout=1.0)
                    if msg["type"] != "kline":
                        continue
                    tag = msg["tag"]
                    tt = float(msg["t"])
                    vals = [float(msg["o"]), float(msg["h"]),
                            float(msg["l"]), float(msg["c"]), float(msg["v"])]
                    closed = bool(msg["closed"])
                    tgt = data_4h if tag == "4h" else data
                    if tgt is None or len(tgt["time"]) == 0:
                        continue
                    last_t = float(tgt["time"][-1])
                    if abs(tt - last_t) < 1e-9:
                        for i, k in enumerate(("open", "high", "low", "close", "volume")):
                            tgt[k][-1] = vals[i]
                    elif tt > last_t:
                        tgt["time"] = np.append(tgt["time"], tt)
                        for i, k in enumerate(("open", "high", "low", "close", "volume")):
                            tgt[k] = np.append(tgt[k], vals[i])
                        mx = 4500 if tag == "4h" else HIST_BARS
                        if len(tgt["time"]) > mx:
                            for k in tgt:
                                tgt[k] = tgt[k][-mx:]
                    if tag == "4h":
                        if closed:
                            h4_closed = True
                    else:
                        if closed:
                            tf_closed = True
            except Empty:
                pass

            if tf_closed or h4_closed:
                ov = compute_overlays(data, data_4h)
                alerts = tracker.check(ov)
                for at, am in alerts:
                    send_notification(at, am)
                last_recompute = time.time()
                if alerts:
                    print(f"[service] candle closed, {len(alerts)} alerts sent")

            if not HAS_WS and time.time() - last_recompute > 60:
                try:
                    data = fetch_klines(SYMBOL, tf, HIST_BARS)
                    data_4h = fetch_klines(SYMBOL, "4h", max(1000, HIST_BARS // 3))
                    if len(data["time"]) > 0:
                        ov = compute_overlays(data, data_4h)
                        alerts = tracker.check(ov)
                        for at, am in alerts:
                            send_notification(at, am)
                    last_recompute = time.time()
                except Exception as e:
                    print(f"[service] poll error: {e}")

        except Exception as e:
            print(f"[service] loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    print("[service] starting SMC monitor service")
    setup_foreground()
    monitor_loop()
