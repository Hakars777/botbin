[app]
title = SMC Alert
package.name = smcalert
package.domain = org.smcalert
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,java
version = 1.2.0
requirements = python3,kivy,numpy,requests,websocket-client,plyer,certifi,urllib3,charset-normalizer,idna,pyjnius

orientation = portrait, landscape
fullscreen = 0
android.permissions = INTERNET,ACCESS_NETWORK_STATE,VIBRATE,POST_NOTIFICATIONS,WAKE_LOCK,FOREGROUND_SERVICE,FOREGROUND_SERVICE_DATA_SYNC,RECEIVE_BOOT_COMPLETED
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.allow_backup = True
android.accept_sdk_license = True
p4a.branch = develop

services = Monitor:service.py:foreground

android.add_src = java_src
android.extra_manifest_application_arguments = <receiver android:name=".BootReceiver" android:enabled="true" android:exported="true" android:directBootAware="false"><intent-filter><action android:name="android.intent.action.BOOT_COMPLETED" /><action android:name="android.intent.action.QUICKBOOT_POWERON" /></intent-filter></receiver>

[buildozer]
log_level = 2
warn_on_root = 1
