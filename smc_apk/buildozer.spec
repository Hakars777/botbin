[app]
title = SMC Alert
package.name = smcalert
package.domain = org.smcalert
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0.0
requirements = python3,kivy,numpy,requests,websocket-client,plyer,certifi,urllib3,charset-normalizer,idna
icon.filename = %(source.dir)s/icon.png
orientation = portrait
fullscreen = 0
android.permissions = INTERNET,ACCESS_NETWORK_STATE,VIBRATE,POST_NOTIFICATIONS,WAKE_LOCK
android.api = 33
android.minapi = 21
android.archs = arm64-v8a
android.allow_backup = True
android.accept_sdk_license = True
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1
