[app]
title = AllSignal
package.name = allsignal
package.domain = org.allsignal
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,java
version = 1.0.0

requirements = python3,kivy,numpy,requests,websocket-client,plyer,certifi,urllib3,charset-normalizer,idna,pyjnius

orientation = portrait, landscape
fullscreen = 0

android.permissions = INTERNET,ACCESS_NETWORK_STATE,WAKE_LOCK,POST_NOTIFICATIONS,FOREGROUND_SERVICE,FOREGROUND_SERVICE_DATA_SYNC,RECEIVE_BOOT_COMPLETED
android.api = 33
android.minapi = 24
android.ndk_api = 24
android.archs = arm64-v8a

android.add_src = java_src

services = Monitor:service.py:foreground

[buildozer]
log_level = 2
warn_on_root = 1
