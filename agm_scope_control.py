"""
ISAPI control for AGM Taipan (Hikvision OEM) — palette, digital zoom, brightness/contrast.

Uses GET-modify-PUT against the same HTTP/ISAPI channel the Android app uses.
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional

try:
    import requests
    from requests.auth import HTTPDigestAuth
except ImportError:
    requests = None

HOST = "10.15.12.1"
USER = "admin"
PWD = "abcd1234"
CHANNEL = 1

# Taipan ISAPI uses PascalCase palette modes (WhiteHot not whiteHot)
PALETTES = [
    ("Black Hot", "BlackHot"),
    ("White Hot", "WhiteHot"),
    ("Red Hot", "RedHot"),
]

ZOOM_STEPS = [1, 2, 4, 8]


class ScopeControl:
    """Remote control for scope image settings over ISAPI."""

    NS = {"h": "http://www.std-cgi.com/ver20/XMLSchema"}

    def __init__(
        self,
        host: str = HOST,
        user: str = USER,
        password: str = PWD,
        channel: int = CHANNEL,
        enabled: bool = True,
    ):
        self.host = host
        self.auth = HTTPDigestAuth(user, password) if requests else None
        self.channel = channel
        self.enabled = enabled and requests is not None
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._debounce = 0.35
        self._api_timeout = 1.5
        self.palette_index = 1
        self.zoom_index = 0
        self.brightness = 50
        self.contrast = 50
        self._connected = False

    def _url(self, path: str) -> str:
        return f"http://{self.host}{path}"

    def _throttle(self) -> bool:
        now = time.time()
        if now - self._last_call < self._debounce:
            return False
        self._last_call = now
        return True

    def _get(self, path: str) -> tuple[int, str]:
        if not self.enabled:
            return -1, ""
        r = requests.get(self._url(path), auth=self.auth, timeout=self._api_timeout)
        return r.status_code, r.text

    def _put(self, path: str, body: str) -> tuple[int, str]:
        if not self.enabled:
            return -1, ""
        r = requests.put(
            self._url(path),
            data=body.encode("utf-8"),
            auth=self.auth,
            headers={"Content-Type": "application/xml"},
            timeout=self._api_timeout,
        )
        return r.status_code, r.text

    def _run_async(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    def ping(self) -> bool:
        try:
            code, _ = self._get("/ISAPI/System/deviceInfo")
            self._connected = code == 200
        except Exception:
            self._connected = False
        return self._connected

    def _set_xml_text(self, xml: str, tag: str, value: str) -> str:
        pattern = rf"(<{tag}[^>]*>)(.*?)(</{tag}>)"
        if re.search(pattern, xml, re.S):
            return re.sub(pattern, rf"\g<1>{value}\g<3>", xml, count=1, flags=re.S)
        return xml.replace("</", f"<{tag}>{value}</{tag}></", 1)

    def set_palette(self, index: int) -> bool:
        index = max(0, min(len(PALETTES) - 1, index))
        self.palette_index = index
        self._run_async(self._set_palette_sync, index)
        return True

    def _set_palette_sync(self, index: int):
        _, mode = PALETTES[index]
        path = f"/ISAPI/Image/channels/{self.channel}/Palettes"
        with self._lock:
            try:
                code, xml = self._get(path)
            except Exception as e:
                print(f"Palette read failed: {e}")
                return
            if code != 200:
                print(f"Palette read HTTP {code}")
                return
            body = self._set_xml_text(xml, "mode", mode)
            try:
                pc, resp = self._put(path, body)
            except Exception as e:
                print(f"Palette write failed: {e}")
                return
            if pc in (200, 204):
                print(f"Palette -> {PALETTES[index][0]} ({mode})")
                self._connected = True
            else:
                print(f"Palette HTTP {pc}: {resp[:120]}")

    def set_zoom(self, index: int) -> bool:
        index = max(0, min(len(ZOOM_STEPS) - 1, index))
        self.zoom_index = index
        self._run_async(self._set_zoom_sync, index)
        return True

    def _set_zoom_sync(self, index: int):
        level = ZOOM_STEPS[index]
        if not self._throttle():
            return
        with self._lock:
            zoom_val = {1: 0, 2: 1, 4: 2, 8: 3}.get(level, 0)
            body = f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZData version="2.0" xmlns="http://www.std-cgi.com/ver20/XMLSchema">
<digitalZoomLevel>{zoom_val}</digitalZoomLevel>
</PTZData>"""
            path = f"/ISAPI/PTZCtrl/channels/{self.channel}/digital"
            try:
                pc, _ = self._put(path, body)
            except Exception:
                pc = -1
            if pc in (200, 204):
                print(f"Zoom -> {level}x")
                self._connected = True
                return
            path = f"/ISAPI/Image/channels/{self.channel}"
            try:
                code, xml = self._get(path)
            except Exception:
                return
            if code != 200:
                return
            b = xml
            for tag in ("digitalZoom", "digitalZoomRatio", "zoomScale"):
                if f"<{tag}>" in b:
                    b = self._set_xml_text(b, tag, str(level))
                    break
            else:
                b = self._set_xml_text(b, "digitalZoom", str(level))
            try:
                pc2, _ = self._put(path, b)
            except Exception:
                return
            if pc2 in (200, 204):
                print(f"Zoom -> {level}x (image channel)")
                self._connected = True

    def set_brightness_contrast(self, brightness: int, contrast: int) -> bool:
        self.brightness = max(0, min(100, brightness))
        self.contrast = max(0, min(100, contrast))
        self._run_async(self._set_brightness_sync, self.brightness, self.contrast)
        return True

    def _set_brightness_sync(self, brightness: int, contrast: int):
        if not self._throttle():
            return
        with self._lock:
            path = f"/ISAPI/Image/channels/{self.channel}"
            try:
                code, xml = self._get(path)
            except Exception:
                return
            if code != 200:
                return
            body = self._set_xml_text(xml, "brightnessLevel", str(brightness))
            body = self._set_xml_text(body, "contrastLevel", str(contrast))
            try:
                pc, _ = self._put(path, body)
            except Exception:
                return
            if pc in (200, 204):
                self._connected = True

    def sync_from_device(self) -> None:
        if not self.enabled:
            return
        try:
            code, xml = self._get(f"/ISAPI/Image/channels/{self.channel}/Palettes")
        except Exception:
            return
        if code == 200:
            for i, (_, mode) in enumerate(PALETTES):
                if mode in xml:
                    self.palette_index = i
                    break

    @property
    def palette_name(self) -> str:
        return PALETTES[self.palette_index][0]

    @property
    def zoom_label(self) -> str:
        return f"{ZOOM_STEPS[self.zoom_index]}x"

    def as_dict(self) -> dict:
        return {
            "connected": self._connected,
            "enabled": self.enabled,
            "palette_index": self.palette_index,
            "palette_name": self.palette_name,
            "zoom_index": self.zoom_index,
            "zoom_label": self.zoom_label,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "palettes": [p[0] for p in PALETTES],
            "zoom_steps": ZOOM_STEPS,
        }
