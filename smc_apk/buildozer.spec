[app]
title = SMC Alert
package.name = smcalert
package.domain = org.smcalert
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.1.0
requirements = python3,kivy,numpy,requests,websocket-client,plyer,certifi,urllib3,charset-normalizer,idna,pyjnius

orientation = all
fullscreen = 0
android.permissions = INTERNET,ACCESS_NETWORK_STATE,VIBRATE,POST_NOTIFICATIONS,WAKE_LOCK,FOREGROUND_SERVICE,FOREGROUND_SERVICE_DATA_SYNC,RECEIVE_BOOT_COMPLETED
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.allow_backup = True
android.accept_sdk_license = True
p4a.branch = develop

services = Monitor:service.py:foreground

[buildozer]
log_level = 2
warn_on_root = 1
