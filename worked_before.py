#!/usr/bin/env python3
import time
import math
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import argparse
import matplotlib.pyplot as plt
from requests.adapters import HTTPAdapter, Retry
from typing import Optional
from urllib.parse import urljoin, urlparse
import os, re, json

# --------------------------------------------------------------------
# TDOT / SmartWay endpoints & config
# --------------------------------------------------------------------
TDOT_CONFIG_URL = "https://smartway.tn.gov/config/config.prod.json"  # contains apiBaseUrl + apiKey
TDOT_BASES = [
    "https://www.tdot.tn.gov/opendata/api/public/",
    "https://tdot.tn.gov/opendata/api/public/",  # secondary; may fail DNS sometimes
]
TDOT_CAMERAS_RESOURCE = "RoadwayCameras"  # per config.prod.json

SMARTWAY_PAGE = "https://smartway.tn.gov/traffic"
APIKEY_CACHE_FILE = ".smartway_api_key.txt"  # stores the TDOT *OpenData* apiKey only

# Browser-ish headers (same idea the SmartWay web app uses; no auth needed)
DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://smartway.tn.gov",
    "Referer": "https://smartway.tn.gov/traffic",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# --------------------------------------------------------------------
# HTTP session with retries
# --------------------------------------------------------------------
def session_with_retries() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6, connect=6, read=6, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(DEFAULT_HEADERS)
    return s

# --------------------------------------------------------------------
# Distance helpers (miles)
# --------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2)
    return 2*R*math.asin(math.sqrt(a))

# --------------------------------------------------------------------
# Cache helpers (for the TDOT OpenData apiKey)
# --------------------------------------------------------------------
def _read_cached_key() -> Optional[str]:
    try:
        with open(APIKEY_CACHE_FILE, "r", encoding="utf-8") as f:
            k = f.read().strip()
            return k or None
    except FileNotFoundError:
        return None

def _write_cached_key(key: str):
    try:
        with open(APIKEY_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(key.strip())
    except Exception:
        pass

def _is_google_maps_key(key: str) -> bool:
    return bool(key) and key.startswith("AIza")

def _is_probably_tdot_key(key: str) -> bool:
    # TDOT config key observed: 32 hex chars. Allow 32–64 hex to be safe.
    return bool(re.fullmatch(r"[0-9a-fA-F]{32,64}", key or ""))

# --------------------------------------------------------------------
# Get the TDOT OpenData apiKey from config.prod.json (authoritative)
# (Falls back to env or cache; ignores Google Maps keys)
# --------------------------------------------------------------------
def get_tdot_api_key(debug: bool=False, force_refresh: bool=False) -> Optional[str]:
    # 1) explicit env override
    env_key = os.getenv("SMARTWAY_API_KEY")
    if env_key and not _is_google_maps_key(env_key):
        if debug:
            print("[apikey] Using SMARTWAY_API_KEY from environment")
        _write_cached_key(env_key)
        return env_key

    # 2) fetch SmartWay config (canonical source)
    try:
        s = session_with_retries()
        r = s.get(TDOT_CONFIG_URL, timeout=20)
        r.raise_for_status()
        cfg = r.json()
        api_key = cfg.get("apiKey")
        if api_key and _is_probably_tdot_key(api_key):
            if debug:
                print(f"[apikey] Found TDOT apiKey in config: {api_key[:6]}…")
            _write_cached_key(api_key)
            return api_key
        else:
            if debug:
                print("[apikey] config.prod.json loaded but 'apiKey' missing or doesn’t look like TDOT key")
    except Exception as e:
        if debug:
            print(f"[apikey] Failed to fetch config.prod.json: {e}")

    # 3) cached value (ignore Google Maps keys)
    if not force_refresh:
        cached = _read_cached_key()
        if cached and not _is_google_maps_key(cached):
            if debug:
                print(f"[apikey] Using cached key from {APIKEY_CACHE_FILE}")
            return cached
        elif cached and _is_google_maps_key(cached) and debug:
            print(f"[apikey] Ignoring cached Google Maps key in {APIKEY_CACHE_FILE}")

    # 4) nothing found
    return None

# --------------------------------------------------------------------
# Camera fetcher (uses TDOT OpenData; passes apiKey as query param)
# --------------------------------------------------------------------
def fetch_cameras(debug_key: bool=False):
    api_key = get_tdot_api_key(debug=debug_key)  # may be None if config failed
    s = session_with_retries()

    last_err = None
    cams = None

    for base in TDOT_BASES:
        url = urljoin(base, TDOT_CAMERAS_RESOURCE)

        # try WITH apiKey first if we have one, then try without
        param_sets = []
        if api_key:
            param_sets.append({"apiKey": api_key})
        param_sets.append({})

        for params in param_sets:
            try:
                if debug_key:
                    print(f"[cams] GET {url} params={params}")
                r = s.get(url, timeout=30, params=params)
                if r.status_code in (401, 403):
                    # If we sent a bad/expired key, clear cache and try once more using config
                    if params and debug_key:
                        print(f"[cams] {r.status_code} with apiKey; clearing cache and refreshing from config…")
                    if params:
                        try:
                            os.remove(APIKEY_CACHE_FILE)
                        except FileNotFoundError:
                            pass
                        # one fresh attempt from config
                        fresh = get_tdot_api_key(debug=debug_key, force_refresh=True)
                        if fresh and fresh != api_key:
                            api_key = fresh
                            continue  # will loop again with new key
                    # if we got 401/403 with NO key, just continue to next base/attempt
                    last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
                    continue

                r.raise_for_status()
                items = r.json()
                if not isinstance(items, list):
                    raise ValueError(f"Unexpected JSON shape: {type(items)}")
                cams = items
                break
            except Exception as e:
                last_err = e
                if debug_key:
                    print(f"[cams] Failed {url} {('with key' if params else 'no key')}: {e}")
        if cams:
            break

    if not cams:
        raise RuntimeError(f"Failed to fetch cameras from TDOT OpenData. Last error: {last_err}")

    # Normalize fields needed downstream
    out = []
    for c in cams:
        lat = c.get("lat") or c.get("latitude")
        lng = c.get("lng") or c.get("longitude") or c.get("lon")
        # choose a stream/snapshot URL if any
        stream = (c.get("httpsVideoUrl") or c.get("httpVideoUrl") or
                  c.get("rtspVideoUrl") or c.get("videoUrl") or
                  c.get("streamUrl") or c.get("m3u8Url") or c.get("imageUrl"))
        if lat is None or lng is None or not stream:
            continue
        out.append({
            "id": c.get("id") or c.get("cameraId") or c.get("name") or len(out),
            "title": c.get("title") or c.get("name") or c.get("roadName") or "TDOT Camera",
            "lat": float(lat),
            "lng": float(lng),
            "_stream": stream,
            "active": c.get("active", True),
        })

    # Keep active + stream-capable
    out = [c for c in out if c["active"] and c["_stream"]]
    return out

def nearest_cameras(cams, lat, lon, max_items=8, max_miles=5.0):
    scored = []
    for c in cams:
        d = haversine(lat, lon, c["lat"], c["lng"])
        if d <= max_miles:
            scored.append((d, c))
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:max_items]]

# -------------------------
# Simple motion-based tracker
# -------------------------
class CentroidTracker:
    """Very simple centroid tracker to get direction of travel axis-crossings."""
    def __init__(self, max_lost=20):
        self.next_id = 1
        self.objects = {}      # id -> (cx, cy)
        self.lost = {}         # id -> frames since seen
        self.max_lost = max_lost
        self.history = {}      # id -> deque of (t, x, y) for speed

    def update(self, detections, ts):
        # detections: list of (x,y,w,h)
        centroids = [(int(x + w/2), int(y + h/2)) for (x,y,w,h) in detections]

        used_objs, used_dets = set(), set()
        matches = []

        # Greedy nearest matching (good enough for low density)
        for oid, (ox, oy) in list(self.objects.items()):
            best_j, best_dist = None, 999999
            for j, (cx, cy) in enumerate(centroids):
                if j in used_dets:
                    continue
                dist = (ox-cx)**2 + (oy-cy)**2
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None and best_dist < 80**2:  # max jump
                self.objects[oid] = centroids[best_j]
                self.lost[oid] = 0
                used_objs.add(oid); used_dets.add(best_j)
                matches.append((oid, best_j))

        # New objects
        for j, c in enumerate(centroids):
            if j not in used_dets:
                oid = self.next_id; self.next_id += 1
                self.objects[oid] = c
                self.lost[oid] = 0
                matches.append((oid, j))

        # Increase lost count
        for oid in list(self.objects.keys()):
            if oid not in [m[0] for m in matches]:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.objects.pop(oid, None)
                    self.lost.pop(oid, None)
                    self.history.pop(oid, None)

        # Append to history for speed
        for oid, j in matches:
            cx, cy = self.objects[oid]
            if oid not in self.history:
                self.history[oid] = deque(maxlen=30)
            self.history[oid].append((ts, cx, cy))

        return self.objects, self.history

# Geometry helpers
def point_line_side(p, a, b):
    (x,y), (x1,y1), (x2,y2) = p, a, b
    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

def segment_length_pixels(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

# Speed estimation
def estimate_speed_mph(track, px_to_ft):
    if len(track) < 5 or not px_to_ft:
        return None
    (t1, x1, y1) = track[-5]
    (t2, x2, y2) = track[-1]
    dt = t2 - t1
    if dt <= 0:
        return None
    dist_px = math.hypot(x2 - x1, y2 - y1)
    dist_ft = dist_px * px_to_ft
    fps = dist_ft / dt
    mph = fps * 0.681818  # 1 ft/s = 0.681818 mph
    return mph

def draw_line(img, a, b, label):
    cv2.line(img, a, b, (255,255,255), 2)
    cv2.circle(img, a, 4, (255,255,255), -1)
    cv2.circle(img, b, 4, (255,255,255), -1)
    cv2.putText(img, label, (a[0], a[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def parse_args():
    p = argparse.ArgumentParser(description="TDOT SmartWay vehicle counting (per direction) with 5-min stats and speed charts.")
    p.add_argument("--lat", type=float, default=36.1602364)
    p.add_argument("--lon", type=float, default=-86.7619816)
    p.add_argument("--radius", type=float, default=5.0, help="Miles search radius")
    p.add_argument("--camera-id", type=int, help="Use this camera ID directly")
    p.add_argument("--auto", action="store_true", help="Auto-select nearest camera without prompting")
    p.add_argument("--outfile", default="data.csv")
    p.add_argument("--api-key", help="(Optional) TDOT OpenData apiKey; else pulled from config & cached")
    p.add_argument("--debug-key", action="store_true", help="Print debug logs during apiKey discovery")
    return p.parse_args()

def open_video(stream):
    # Try with FFMPEG backend hint first (helps with HLS .m3u8)
    cap = cv2.VideoCapture(stream, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap
    # Fallback
    cap = cv2.VideoCapture(stream)
    return cap if cap.isOpened() else None

def main():
    args = parse_args()

    # If they provided a key, use it and overwrite cache/env for this run (but ignore 'AIza…' maps keys)
    if args.api_key and not _is_google_maps_key(args.api_key):
        os.environ["SMARTWAY_API_KEY"] = args.api_key
        _write_cached_key(args.api_key)

    print("Fetching cameras…")
    cams = fetch_cameras(debug_key=args.debug_key)
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

    backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    lineA = None  # (p1, p2)
    lineB = None
    calib = None  # (p1, p2, px_to_ft)
    setting_mode = None
    clicks = []

    counts = {"A": 0, "B": 0}
    crossed = set()  # (object_id, 'A'/'B')
    tracker = CentroidTracker(max_lost=30)

    samples = []
    last_bin_start = time.time()

    def on_mouse(event, x, y, flags, param):
        nonlocal clicks, setting_mode, lineA, lineB, calib
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x,y))
            if setting_mode == "A" and len(clicks) == 2:
                lineA = (clicks[0], clicks[1])
                clicks = []; setting_mode = None
            elif setting_mode == "B" and len(clicks) == 2:
                lineB = (clicks[0], clicks[1])
                clicks = []; setting_mode = None
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
                except:
                    print("Invalid number.")
                clicks = []; setting_mode = None

    cv2.namedWindow("SmartWay")
    cv2.setMouseCallback("SmartWay", on_mouse)

    bin_seconds = 5 * 60

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

        dets = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 500 and h > 18 and w > 18:
                dets.append((x,y,w,h))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        objects, history = tracker.update(dets, ts)

        if lineA: draw_line(frame, lineA[0], lineA[1], "Dir A")
        if lineB: draw_line(frame, lineB[0], lineB[1], "Dir B")
        if calib:
            cv2.line(frame, calib[0], calib[1], (0,0,0), 2)
            cv2.putText(frame, "Calibrated", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        speeds = {"A": [], "B": []}
        for oid, (cx, cy) in objects.items():
            cv2.putText(frame, f"ID{oid}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            if lineA:
                side_now = point_line_side((cx,cy), lineA[0], lineA[1])
                if len(history[oid]) >= 2:
                    _, px, py = history[oid][-2]
                    side_prev = point_line_side((px,py), lineA[0], lineA[1])
                    if side_now * side_prev < 0:
                        key = (oid, "A")
                        if key not in crossed:
                            counts["A"] += 1
                            crossed.add(key)

            if lineB:
                side_now = point_line_side((cx,cy), lineB[0], lineB[1])
                if len(history[oid]) >= 2:
                    _, px, py = history[oid][-2]
                    side_prev = point_line_side((px,py), lineB[0], lineB[1])
                    if side_now * side_prev < 0:
                        key = (oid, "B")
                        if key not in crossed:
                            counts["B"] += 1
                            crossed.add(key)

            if calib:
                mph = estimate_speed_mph(history[oid], calib[2])
                if mph is not None:
                    if lineA and lineB:
                        dA = abs(point_line_side((cx,cy), lineA[0], lineA[1]))
                        dB = abs(point_line_side((cx,cy), lineB[0], lineB[1]))
                        dirkey = "A" if dA < dB else "B"
                    elif lineA:
                        dirkey = "A"
                    elif lineB:
                        dirkey = "B"
                    else:
                        dirkey = "A"
                    speeds[dirkey].append(mph)

        cv2.putText(frame, f"Counts A:{counts['A']}  B:{counts['B']}", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.imshow("SmartWay", frame)

        # (omitted: 5-min binning and CSV writing for brevity in this snippet)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            if not lineA:
                print("Click two points for Direction A line.")
                setting_mode = "A"; clicks = []
            elif not lineB:
                print("Click two points for Direction B line.")
                setting_mode = "B"; clicks = []
            else:
                print("Both lines set. Press 'd' again to redefine.")
                lineA = None; lineB = None
        elif key == ord('k'):
            print("Click two points spanning a known real-world distance (feet)…")
            setting_mode = "K"; clicks = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()