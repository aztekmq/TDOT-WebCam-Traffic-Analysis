#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDOT SmartWay – Nearby Camera Selection, Lightweight Vehicle Counting,
and Real-time Charting of Counts & Speeds.

Author: Rob Lee
Contributors: ChatGPT (assistant)
License: MIT (or your preferred license)

Description
-----------
- Discovers the TDOT OpenData API key from SmartWay's `config.prod.json`.
- Fetches roadway camera metadata and selects the nearest cameras.
- Falls back to a cached camera inventory on 204/empty responses.
- Opens the selected camera stream (or snapshot) and runs a simple motion-based tracker.
- Draws two user-defined direction lines (A and B) for crossing counts.
- Supports a two-point pixel→feet calibration to estimate speed.
- Displays a real-time chart with:
    * ongoing (running) average speed in direction A
    * ongoing (running) average speed in direction B
    * cumulative car count in direction A
    * cumulative car count in direction B
    * fastest speed so far in direction A
    * fastest speed so far in direction B

Controls
--------
- d : define direction lines (click two points for A, press 'd' again, two points for B)
- k : calibrate speed (click two points spanning a known distance, then enter feet)
- q : quit

Change History
--------------
2025-08-12  RL  Initial documented cleanup:
                - Normalize indentation and imports
                - Add full module/class/function docstrings
                - Pull TDOT apiBaseUrl from config.prod.json when available
                - Harden API key handling (never confuse Google Maps key)
                - Add real-time chart for ongoing averages, cumulative counts, fastest speeds
                - Add --display-scale to control window size by scaling frames
2025-08-12  RL  Robust camera fetch:
                - Handle 204/empty responses with short backoff and cache-busting retries
                - Try multiple auth styles (query/header/x-api-key) and $format=json fallback
                - Add camera inventory cache (.tdot_cameras_cache.json) and fallback
                - Add --no-cache-fallback to disable cache usage
2025-08-12  RL  Chart display reliability:
                - Force interactive Matplotlib backend (TkAgg/Qt5Agg) before pyplot import
                - Call plt.show(block=False) after creating figure
                - Flush GUI events each update; set window title
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from urllib.parse import urljoin

# ---- Matplotlib backend selection (must be BEFORE pyplot import) ----
# Try to ensure an interactive backend so a chart window actually appears.
import matplotlib as _mpl
if not os.environ.get("MPLBACKEND"):  # allow user override
    try:
        _mpl.use("TkAgg", force=True)
    except Exception:
        try:
            _mpl.use("Qt5Agg", force=True)
        except Exception:
            # Fall back to whatever is available; we'll warn later if non-interactive.
            pass
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Constants (TDOT / SmartWay)
# --------------------------------------------------------------------

TDOT_CONFIG_URL = "https://smartway.tn.gov/config/config.prod.json"  # contains apiBaseUrl + apiKey

# Fallbacks if config doesn't provide apiBaseUrl or it's unreachable
TDOT_BASES_FALLBACK: List[str] = [
    "https://www.tdot.tn.gov/opendata/api/public/",
    "https://tdot.tn.gov/opendata/api/public/",
]

TDOT_CAMERAS_RESOURCE = "RoadwayCameras"  # per config
APIKEY_CACHE_FILE = ".smartway_api_key.txt"          # caches *TDOT OpenData* apiKey only
CAMS_CACHE_FILE   = ".tdot_cameras_cache.json"       # caches last-good raw camera payload (list of dicts)

# Typical browser-ish headers (helpful, but not strictly required)
DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://smartway.tn.gov",
    "Referer": "https://smartway.tn.gov/traffic",
}

# --------------------------------------------------------------------
# HTTP session helpers
# --------------------------------------------------------------------

def session_with_retries() -> requests.Session:
    """
    Build a requests.Session with sensible retries for flaky endpoints.
    """
    s = requests.Session()
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(DEFAULT_HEADERS)
    return s

# --------------------------------------------------------------------
# Geo helpers
# --------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in *miles* between two lat/lon points.
    """
    R = 3958.8  # miles
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# --------------------------------------------------------------------
# Small file cache helpers
# --------------------------------------------------------------------

def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception:
        return None

def _write_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass

def _read_cached_key() -> Optional[str]:
    txt = _read_text(APIKEY_CACHE_FILE)
    return txt.strip() if txt else None

def _write_cached_key(key: str) -> None:
    if key:
        _write_text(APIKEY_CACHE_FILE, key.strip())

def _read_cams_cache(debug: bool = False) -> Optional[List[Dict[str, Any]]]:
    import json
    txt = _read_text(CAMS_CACHE_FILE)
    if not txt:
        return None
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            if debug:
                print(f"[cache] Loaded {len(data)} cameras from {CAMS_CACHE_FILE}")
            return data
    except Exception:
        pass
    return None

def _write_cams_cache(raw_items: List[Dict[str, Any]], debug: bool = False) -> None:
    import json
    try:
        _write_text(CAMS_CACHE_FILE, json.dumps(raw_items, ensure_ascii=False))
        if debug:
            print(f"[cache] Saved {len(raw_items)} cameras to {CAMS_CACHE_FILE}")
    except Exception:
        pass

# --------------------------------------------------------------------
# API key validation
# --------------------------------------------------------------------

def _is_google_maps_key(key: str) -> bool:
    """Heuristic: Google Maps browser keys typically start with 'AIza'."""
    return bool(key) and key.startswith("AIza")

def _is_probably_tdot_key(key: str) -> bool:
    """
    Observed TDOT OpenData key is 32 hex chars. Allow 32–64 hex to be safe.
    """
    return bool(re.fullmatch(r"[0-9a-fA-F]{32,64}", key or ""))

# --------------------------------------------------------------------
# Config & API key discovery
# --------------------------------------------------------------------

def load_tdot_config(debug: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Load SmartWay config JSON and return (api_base_url, api_key).
    """
    try:
        s = session_with_retries()
        r = s.get(TDOT_CONFIG_URL, timeout=20)
        r.raise_for_status()
        cfg = r.json()
        api_base = cfg.get("apiBaseUrl") or cfg.get("apiBaseURL") or cfg.get("baseUrl")
        api_key = cfg.get("apiKey")
        if debug:
            print(f"[config] apiBaseUrl={api_base}  apiKey={'present' if api_key else 'missing'}")
        return api_base, api_key
    except Exception as e:
        if debug:
            print(f"[config] Failed to fetch config.prod.json: {e}")
        return None, None

def get_tdot_api_key(debug: bool = False, force_refresh: bool = False) -> Optional[str]:
    """
    Resolve the TDOT OpenData apiKey from (in order):
      1) Environment variable SMARTWAY_API_KEY (if not a Google Maps key)
      2) Cache file (ignoring Google Maps keys)
      3) SmartWay config.prod.json (authoritative source)
    """
    env_key = os.getenv("SMARTWAY_API_KEY")
    if env_key and not _is_google_maps_key(env_key):
        if debug:
            print("[apikey] Using SMARTWAY_API_KEY from environment")
        _write_cached_key(env_key)
        return env_key

    if not force_refresh:
        cached = _read_cached_key()
        if cached and not _is_google_maps_key(cached):
            if debug:
                print(f"[apikey] Using cached key from {APIKEY_CACHE_FILE}")
            return cached
        elif cached and _is_google_maps_key(cached) and debug:
            print(f"[apikey] Ignoring cached Google Maps key in {APIKEY_CACHE_FILE}")

    api_base, api_key = load_tdot_config(debug=debug)
    if api_key and _is_probably_tdot_key(api_key):
        if debug:
            print(f"[apikey] From config: {api_key[:6]}…")
        _write_cached_key(api_key)
        return api_key
    elif debug and api_key and _is_google_maps_key(api_key):
        print("[apikey] Config provided a Google Maps key; ignoring for OpenData.")
    return None

# --------------------------------------------------------------------
# Camera fetch (robust)
# --------------------------------------------------------------------

def _candidate_bases(debug: bool = False) -> List[str]:
    """
    Combine base URL from config with known fallbacks, preserving order and uniqueness.
    """
    api_base, _ = load_tdot_config(debug=debug)
    bases: List[str] = []
    if api_base:
        if not api_base.endswith("/"):
            api_base = api_base + "/"
        bases.append(api_base)
    bases.extend(TDOT_BASES_FALLBACK)
    seen: set = set()
    uniq = [b for b in bases if not (b in seen or seen.add(b))]
    if debug:
        print("[bases]", uniq)
    return uniq

def _decode_json_or_log(r: requests.Response, debug: bool, url: str) -> Optional[list]:
    """
    Try to parse JSON list from response. If it fails or isn't JSON, print
    diagnostics and return None so the caller can try alternatives.
    """
    ctype = (r.headers.get("Content-Type") or "").lower()
    text = r.text or ""
    if "application/json" in ctype:
        try:
            data = r.json()
            return data if isinstance(data, list) else None
        except ValueError as e:
            if debug:
                print(f"[cams] JSON decode error from {url}: {e}")
                print(f"[cams] Status={r.status_code} Content-Type={ctype} Len={len(text)} Snippet={text[:240]!r}")
            return None
    else:
        if debug:
            print(f"[cams] Non-JSON from {url}: Status={r.status_code} Content-Type={ctype} Len={len(text)}")
            if text:
                print(f"[cams] Snippet={text[:240]!r}")
        return None

def fetch_cameras(debug_key: bool = False, allow_cache_fallback: bool = True) -> List[Dict]:
    """
    Fetch camera metadata from TDOT OpenData and normalize key fields.
    Retries multiple auth styles, handles 204/empty responses with cache-busting,
    and adds $format=json fallback if needed. If all attempts fail and a cached
    inventory exists, returns the cached inventory.

    Parameters
    ----------
    debug_key : bool
        Verbose logs for key/config/fetch flow.
    allow_cache_fallback : bool
        If True, load last-good camera list from cache when live fetch fails.
    """
    import json

    api_key = get_tdot_api_key(debug=debug_key)  # may be None if config failed
    s = session_with_retries()

    last_err: Optional[Exception] = None
    raw_items: Optional[List[Dict[str, Any]]] = None

    for base in _candidate_bases(debug=debug_key):
        url = urljoin(base, TDOT_CAMERAS_RESOURCE)

        # Build attempt matrix: (params, extra_headers, label)
        attempts: List[Tuple[Dict[str, str], Dict[str, str], str]] = []
        if api_key:
            attempts += [
                ({"apiKey": api_key}, {}, "key=query"),
                ({}, {"ApiKey": api_key}, "key=ApiKeyHdr"),
                ({}, {"x-api-key": api_key}, "key=x-api-key"),
            ]
        attempts.append(({}, {}, "no-key"))

        # Try each attempt; if payload isn't JSON, retry once with $format=json
        for params, extra_hdrs, label in attempts:
            hdrs = dict(s.headers)
            hdrs.update(extra_hdrs)

            try:
                if debug_key:
                    print(f"[cams] GET {url} params={params} ({label})")
                r = s.get(url, timeout=30, params=params, headers=hdrs)

                # Treat 204/empty as transient; retry with short backoff and cache-buster
                if r.status_code == 204 or not (r.text or r.content):
                    if debug_key:
                        print(f"[cams] 204/empty from {r.url}; retrying with cache-buster…")
                    tried_nonempty = False
                    for _ in range(2):  # up to 2 extra tries
                        time.sleep(0.6)
                        p2 = dict(params)
                        p2["_"] = str(int(time.time() * 1000))  # cache-buster query param
                        r2 = s.get(url, timeout=30, params=p2, headers=hdrs)
                        if r2.status_code not in (204,) and (r2.text or r2.content):
                            r = r2
                            tried_nonempty = True
                            break
                    if not tried_nonempty and debug_key:
                        print(f"[cams] Still empty after cache-buster tries.")

                # auth/permission handling
                if r.status_code in (401, 403):
                    if params or extra_hdrs:
                        if debug_key:
                            print(f"[cams] {r.status_code} with key; clearing cache and refreshing from config…")
                        try:
                            os.remove(APIKEY_CACHE_FILE)
                        except FileNotFoundError:
                            pass
                        fresh = get_tdot_api_key(debug=debug_key, force_refresh=True)
                        if fresh and fresh != api_key:
                            api_key = fresh
                            # restart attempts with fresh key
                            break
                    last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                    continue

                r.raise_for_status()

                items = _decode_json_or_log(r, debug_key, r.url)
                if items is None:
                    # one more try with $format=json
                    if "$format" not in params:
                        p2 = dict(params)
                        p2["$format"] = "json"
                        if debug_key:
                            print(f"[cams] Retrying with $format=json → {url} params={p2} ({label}+format)")
                        r2 = s.get(url, timeout=30, params=p2, headers=hdrs)
                        if r2.status_code in (401, 403):
                            last_err = requests.HTTPError(f"{r2.status_code} {r2.reason}")
                            continue
                        r2.raise_for_status()
                        items = _decode_json_or_log(r2, debug_key, r2.url)

                if items is None:
                    last_err = ValueError("Non-JSON or empty response")
                    continue

                # Success
                if not isinstance(items, list):
                    last_err = ValueError(f"Unexpected JSON shape: {type(items)}")
                    continue

                raw_items = items
                if debug_key:
                    print(f"[cams] OK {r.status_code}: received {len(raw_items)} cameras from {r.url}")
                break

            except Exception as e:
                last_err = e
                if debug_key:
                    print(f"[cams] Failed {url} ({label}): {e}")

        if raw_items:
            break  # success for this base

    # Cache or fallback
    if raw_items:
        _write_cams_cache(raw_items, debug=debug_key)
    else:
        if allow_cache_fallback:
            cached = _read_cams_cache(debug=debug_key)
            if cached:
                raw_items = cached
                if debug_key:
                    print("[cams] Using cached camera inventory due to live fetch failure.")
        if not raw_items:
            raise RuntimeError(f"Failed to fetch cameras from TDOT OpenData. Last error: {last_err}")

    # Normalize fields needed downstream
    out: List[Dict] = []
    for c in raw_items:
        lat = c.get("lat") or c.get("latitude")
        lng = c.get("lng") or c.get("longitude") or c.get("lon")
        stream = (
            c.get("httpsVideoUrl")
            or c.get("httpVideoUrl")
            or c.get("rtspVideoUrl")
            or c.get("videoUrl")
            or c.get("streamUrl")
            or c.get("m3u8Url")
            or c.get("imageUrl")
        )
        if lat is None or lng is None or not stream:
            continue
        out.append(
            {
                "id": c.get("id") or c.get("cameraId") or c.get("name") or len(out),
                "title": c.get("title") or c.get("name") or c.get("roadName") or "TDOT Camera",
                "lat": float(lat),
                "lng": float(lng),
                "_stream": str(stream),
                "active": bool(c.get("active", True)),
            }
        )

    out = [c for c in out if c["active"] and c["_stream"]]
    return out

# --------------------------------------------------------------------
# Lightweight tracker & speed helpers
# --------------------------------------------------------------------

class CentroidTracker:
    """
    Very simple centroid tracker for low-density scenes.
    Assigns IDs, matches by nearest centroid, tracks short histories.
    """

    def __init__(self, max_lost: int = 20) -> None:
        self.next_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}  # id -> (cx, cy)
        self.lost: Dict[int, int] = {}                 # id -> frames since seen
        self.max_lost = max_lost
        self.history: Dict[int, deque] = {}            # id -> deque of (t, x, y)

    def update(self, detections: List[Tuple[int, int, int, int]], ts: float):
        """
        detections: list of (x, y, w, h)
        ts: current timestamp (seconds)
        """
        centroids = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in detections]

        used_objs, used_dets = set(), set()
        matches: List[Tuple[int, int]] = []

        # Greedy nearest matching
        for oid, (ox, oy) in list(self.objects.items()):
            best_j, best_dist = None, 1_000_000
            for j, (cx, cy) in enumerate(centroids):
                if j in used_dets:
                    continue
                dist = (ox - cx) ** 2 + (oy - cy) ** 2
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None and best_dist < 80 ** 2:  # max jump limit
                self.objects[oid] = centroids[best_j]
                self.lost[oid] = 0
                used_objs.add(oid)
                used_dets.add(best_j)
                matches.append((oid, j))

        # New objects
        for j, c in enumerate(centroids):
            if j not in used_dets:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.lost[oid] = 0
                matches.append((oid, j))

        # Lost objects housekeeping
        for oid in list(self.objects.keys()):
            if oid not in [m[0] for m in matches]:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.objects.pop(oid, None)
                    self.lost.pop(oid, None)
                    self.history.pop(oid, None)

        # Append to history
        for oid, _ in matches:
            cx, cy = self.objects[oid]
            if oid not in self.history:
                self.history[oid] = deque(maxlen=30)
            self.history[oid].append((ts, cx, cy))

        return self.objects, self.history

def point_line_side(p: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Return sign indicating which side of line AB the point P lies."""
    (x, y), (x1, y1), (x2, y2) = p, a, b
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def segment_length_pixels(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean length in pixels between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])

def estimate_speed_mph(track: deque, px_to_ft: Optional[float]) -> Optional[float]:
    """
    Very rough speed estimate using the last 5 frames of centroid motion.
    Requires prior calibration to convert pixels to feet.
    """
    if px_to_ft is None or len(track) < 5:
        return None
    (t1, x1, y1) = track[-5]
    (t2, x2, y2) = track[-1]
    dt = t2 - t1
    if dt <= 0:
        return None
    dist_px = math.hypot(x2 - x1, y2 - y1)
    dist_ft = dist_px * px_to_ft
    fps = dist_ft / dt
    return fps * 0.681818  # 1 ft/s = 0.681818 mph

def draw_line(img, a: Tuple[int, int], b: Tuple[int, int], label: str) -> None:
    """Draw a labeled line on the frame."""
    cv2.line(img, a, b, (255, 255, 255), 2)
    cv2.circle(img, a, 4, (255, 255, 255), -1)
    cv2.circle(img, b, 4, (255, 255, 255), -1)
    cv2.putText(img, label, (a[0], a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --------------------------------------------------------------------
# Real-time chart – counts (left axis) and speeds (right axis)
# --------------------------------------------------------------------

def _backend_is_interactive() -> bool:
    b = plt.get_backend().lower()
    return any(key in b for key in ("qt", "tk", "wx", "gtk", "macosx"))

class RealTimeChart:
    """
    Simple dual-axis (counts vs speeds) real-time chart.

    Left axis:
      - cumulative cars A
      - cumulative cars B
    Right axis:
      - ongoing (running) average mph A
      - ongoing (running) average mph B
      - fastest mph A
      - fastest mph B
    """

    def __init__(self) -> None:
        plt.ion()
        self.fig, self.ax1 = plt.subplots(constrained_layout=True)
        self.ax2 = self.ax1.twinx()

        self.x: List[float] = []
        self.countA: List[float] = []
        self.countB: List[float] = []
        self.avgA: List[float] = []
        self.avgB: List[float] = []
        self.fastA: List[float] = []
        self.fastB: List[float] = []

        (self.l_countA,) = self.ax1.plot([], [], label="A cars (cum)")
        (self.l_countB,) = self.ax1.plot([], [], label="B cars (cum)")
        (self.l_avgA,) = self.ax2.plot([], [], label="A avg mph")
        (self.l_avgB,) = self.ax2.plot([], [], label="B avg mph")
        (self.l_fastA,) = self.ax2.plot([], [], label="A fastest mph")
        (self.l_fastB,) = self.ax2.plot([], [], label="B fastest mph")

        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Cars (cumulative)")
        self.ax2.set_ylabel("Speed (mph)")
        self.ax1.grid(True, alpha=0.3)

        # Separate legends to avoid overlap
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper right")

        # Give the window a clear title; show non-blocking
        try:
            self.fig.canvas.manager.set_window_title("SmartWay: Live Chart")
        except Exception:
            pass
        if not _backend_is_interactive():
            print(f"[chart] Warning: Matplotlib backend '{plt.get_backend()}' is not interactive. "
                  f"Set MPLBACKEND=TkAgg (or install tkinter/Qt).")
        plt.show(block=False)

    def push(
        self,
        t: float,
        countA: int,
        countB: int,
        avgA: Optional[float],
        avgB: Optional[float],
        fastA: Optional[float],
        fastB: Optional[float],
    ) -> None:
        """Append one time step to the chart."""
        self.x.append(t)
        self.countA.append(countA)
        self.countB.append(countB)
        self.avgA.append(avgA if avgA is not None else np.nan)
        self.avgB.append(avgB if avgB is not None else np.nan)
        self.fastA.append(fastA if fastA is not None else np.nan)
        self.fastB.append(fastB if fastB is not None else np.nan)

        self.l_countA.set_data(self.x, self.countA)
        self.l_countB.set_data(self.x, self.countB)
        self.l_avgA.set_data(self.x, self.avgA)
        self.l_avgB.set_data(self.x, self.avgB)
        self.l_fastA.set_data(self.x, self.fastA)
        self.l_fastB.set_data(self.x, self.fastB)

        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax2.relim(); self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(0.001)

# --------------------------------------------------------------------
# Video helpers & CLI
# --------------------------------------------------------------------

def open_video(stream: str) -> Optional[cv2.VideoCapture]:
    """
    Try to open a stream (HLS/MJPEG/etc.). Returns an opened VideoCapture or None.
    """
    cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(stream)
    return cap if cap.isOpened() else None

def parse_args() -> argparse.Namespace:
    """
    CLI arguments.
    """
    p = argparse.ArgumentParser(
        description="TDOT SmartWay vehicle counting with real-time charting."
    )
    p.add_argument("--lat", type=float, default=36.1602364)
    p.add_argument("--lon", type=float, default=-86.7619816)
    p.add_argument("--radius", type=float, default=5.0, help="Miles search radius")
    p.add_argument("--camera-id", type=str, help="Use this camera ID directly")
    p.add_argument("--auto", action="store_true", help="Auto-select nearest camera without prompting")
    p.add_argument("--api-key", help="(Optional) TDOT OpenData apiKey; else pulled from config & cached")
    p.add_argument("--debug-key", action="store_true", help="Print debug logs during apiKey discovery")
    p.add_argument(
        "--chart-interval",
        type=float,
        default=0.25,
        help="Seconds between live chart updates (e.g., 0.25).",
    )
    p.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Scale factor for the video window (e.g., 0.75 for 75%% size, 1.5 to upscale).",
    )
    p.add_argument(
        "--no-cache-fallback",
        action="store_true",
        help="Do not use cached camera list when TDOT API returns 204/empty.",
    )
    return p.parse_args()

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def nearest_cameras(cams, lat, lon, max_items=8, max_miles=5.0):
    scored = []
    for c in cams:
        d = haversine(lat, lon, c["lat"], c["lng"])
        if d <= max_miles:
            scored.append((d, c))
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:max_items]]

def main() -> None:
    args = parse_args()

    print(f"[chart] Matplotlib backend: {plt.get_backend()}")

    # If provided a key, use it (but ignore Google Maps keys)
    if args.api_key and not _is_google_maps_key(args.api_key):
        os.environ["SMARTWAY_API_KEY"] = args.api_key
        _write_cached_key(args.api_key)

    print("Fetching cameras…")
    cams = fetch_cameras(
        debug_key=args.debug_key,
        allow_cache_fallback=not args.no_cache_fallback
    )

    # Choose a camera near the given point
    nearby = nearest_cameras(cams, args.lat, args.lon, max_items=8, max_miles=args.radius)
    if not nearby:
        print("No cameras found within radius. Try increasing --radius.")
        return

    if args.camera_id:
        cam = next((c for c in cams if str(c["id"]) == str(args.camera_id)), None)
        if not cam:
            print("Camera ID not found. Nearby options:")
            for c in nearby:
                dist = haversine(args.lat, args.lon, c["lat"], c["lng"])
                print(f'  id={c["id"]}  {c["title"]}  {dist:4.2f} mi  stream={c["_stream"]}')
            return
    else:
        if args.auto:
            cam = nearby[0]
            dist = haversine(args.lat, args.lon, cam["lat"], cam["lng"])
            print(f"Auto-selected nearest camera: id={cam['id']}  {cam['title']}  ({dist:4.2f} mi)")
        else:
            print("Nearby cameras:")
            for c in nearby:
                dist = haversine(args.lat, args.lon, c["lat"], c["lng"])
                print(f'id={c["id"]:<6}  {c["title"]:<40} {dist:4.2f} mi  stream={c["_stream"]}')
            cam_id = input("Enter camera id to use: ").strip()
            cam = next((c for c in cams if str(c["id"]) == cam_id), None)
            if not cam:
                print("Invalid camera id.")
                return

    stream = cam["_stream"]
    print(f"\nOpening stream: {stream}\n(If this fails, install FFmpeg and ensure OpenCV can use it.)")
    cap = open_video(stream)
    if not cap:
        print("Failed to open video stream.")
        return

    # Background subtraction & morphology
    backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # User-defined geometry
    lineA: Optional[Tuple[int, int]] = None  # (p1, p2)
    lineB: Optional[Tuple[int, int]] = None
    calib: Optional[Tuple[Tuple[int, int], Tuple[int, int], float]] = None  # (p1, p2, px_to_ft)
    setting_mode: Optional[str] = None
    clicks: List[Tuple[int, int]] = []

    # Counts & tracker
    counts = {"A": 0, "B": 0}
    crossed = set()  # (object_id, 'A'/'B')
    tracker = CentroidTracker(max_lost=30)

    # Cumulative stats for real-time chart
    cum = {
        "A": 0,
        "B": 0,            # cumulative counts (cars)
        "sumA": 0.0,
        "sumB": 0.0,       # sums of speeds at crossings
        "nA": 0,
        "nB": 0,           # number of speed samples
        "fastA": None,
        "fastB": None,     # fastest mph so far
    }
    chart = RealTimeChart()
    t0 = time.time()
    last_chart = 0.0  # throttle chart updates

    # Mouse input
    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, setting_mode, lineA, lineB, calib
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            if setting_mode == "A" and len(clicks) == 2:
                lineA = (clicks[0], clicks[1])
                clicks = []
                setting_mode = None
            elif setting_mode == "B" and len(clicks) == 2:
                lineB = (clicks[0], clicks[1])
                clicks = []
                setting_mode = None
            elif setting_mode == "K" and len(clicks) == 2:
                p1, p2 = clicks
                px = segment_length_pixels(p1, p2)
                print(f"Selected pixel distance: {px:.1f}px")
                feet_str = input("Enter real-world distance between these points (feet): ").strip()
                try:
                    feet = float(feet_str)
                    px_to_ft = feet / px if px > 0 else None
                    if px_to_ft and px_to_ft > 0:
                        calib = (p1, p2, px_to_ft)
                        print(f"Calibrated: 1px = {px_to_ft:.4f} ft")
                    else:
                        print("Invalid calibration.")
                except Exception:
                    print("Invalid number.")
                clicks = []
                setting_mode = None

    cv2.namedWindow("SmartWay")  # autosize to frame
    cv2.setMouseCallback("SmartWay", on_mouse)

    print("\nControls:")
    print("  d : define direction lines (click 2 points for A, then press d again, 2 points for B)")
    print("  k : calibrate for speed (click 2 points spanning known distance, then enter feet)")
    print("  q : quit\n")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.2)
            continue

        ts = time.time()
        fg = backsub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 500 and h > 18 and w > 18:
                dets.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        objects, history = tracker.update(dets, ts)

        if lineA:
            draw_line(frame, lineA[0], lineA[1], "Dir A")
        if lineB:
            draw_line(frame, lineB[0], lineB[1], "Dir B")
        if calib:
            cv2.line(frame, calib[0], calib[1], (0, 0, 0), 2)
            cv2.putText(frame, "Calibrated", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Draw IDs and detect crossings
        for oid, (cx, cy) in objects.items():
            cv2.putText(frame, f"ID{oid}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Crossings for A
            if lineA and len(history[oid]) >= 2:
                side_now = point_line_side((cx, cy), lineA[0], lineA[1])
                _, px_prev, py_prev = history[oid][-2]
                side_prev = point_line_side((px_prev, py_prev), lineA[0], lineA[1])
                if side_now * side_prev < 0:
                    key = (oid, "A")
                    if key not in crossed:
                        counts["A"] += 1
                        crossed.add(key)
                        mph_cross = estimate_speed_mph(history[oid], calib[2]) if calib else None
                        cum["A"] += 1
                        if mph_cross is not None:
                            cum["sumA"] += mph_cross
                            cum["nA"] += 1
                            cum["fastA"] = mph_cross if cum["fastA"] is None else max(cum["fastA"], mph_cross)

            # Crossings for B
            if lineB and len(history[oid]) >= 2:
                side_now = point_line_side((cx, cy), lineB[0], lineB[1])
                _, px_prev, py_prev = history[oid][-2]
                side_prev = point_line_side((px_prev, py_prev), lineB[0], lineB[1])
                if side_now * side_prev < 0:
                    key = (oid, "B")
                    if key not in crossed:
                        counts["B"] += 1
                        crossed.add(key)
                        mph_cross = estimate_speed_mph(history[oid], calib[2]) if calib else None
                        cum["B"] += 1
                        if mph_cross is not None:
                            cum["sumB"] += mph_cross
                            cum["nB"] += 1
                            cum["fastB"] = mph_cross if cum["fastB"] is None else max(cum["fastB"], mph_cross)

        # Heads-up overlay (counts)
        cv2.putText(
            frame,
            f"Counts A:{counts['A']}  B:{counts['B']}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Scale the frame for display if requested
        scale = args.display_scale if args.display_scale and args.display_scale > 0 else 1.0
        disp_frame = frame
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            disp_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)

        cv2.imshow("SmartWay", disp_frame)

        # Update the live chart (ongoing avg, cumulative counts, fastest)
        now = time.time()
        if now - last_chart >= args.chart_interval:
            avgA = (cum["sumA"] / cum["nA"]) if cum["nA"] > 0 else None
            avgB = (cum["sumB"] / cum["nB"]) if cum["nB"] > 0 else None
            chart.push(
                now - t0,
                cum["A"],
                cum["B"],
                avgA,
                avgB,
                cum["fastA"],
                cum["fastB"],
            )
            last_chart = now

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            if not lineA:
                print("Click two points for Direction A line.")
                setting_mode = "A"
                clicks = []
            elif not lineB:
                print("Click two points for Direction B line.")
                setting_mode = "B"
                clicks = []
            else:
                print("Both lines set. Press 'd' again to redefine.")
                lineA = None
                lineB = None
        elif key == ord("k"):
            print("Click two points spanning a known real-world distance (feet)…")
            setting_mode = "K"
            clicks = []

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------------------------------

if __name__ == "__main__":
    main()