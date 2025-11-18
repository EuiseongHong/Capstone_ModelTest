

import cv2
from ultralytics import YOLO
import os, glob
import numpy as np
import re
from collections import defaultdict
import json

NAME = "crash_ver2"
RAW_VIDEO = f"./videos/{NAME}.mov"              #input video
RUN_NAME  = f"{NAME}"
RUN_DIR   = f"./runs/detect/{RUN_NAME}"

VIDEO     = f"{RUN_DIR}/{NAME}.mp4"             #annotated video
LABELS_DIR = f"{RUN_DIR}/labels"
CSV_OUT   = RUN_DIR

#trained model
model = YOLO("better.pt")
results = model.track(
    source=RAW_VIDEO,
    tracker="bytetrack.yaml",
    conf=0.4,
    persist=True,
    save=True,          #save to RUN_DIR
    save_txt=True,      #saves per-frame labels
    show=False,
    stream=False,
    name=RUN_NAME
)

#open raw video to calculate homography
cap = cv2.VideoCapture(RAW_VIDEO)        
fps  = cap.get(cv2.CAP_PROP_FPS)
W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Himg = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if not fps or fps <= 0:
    raise RuntimeError("Could not read FPS from video; set fps manually.")
print(f"FPS={fps}, W={W}, H={Himg}")

#frame 1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Failed to read frame 0")
frame_path = os.path.join(RUN_DIR, "frame_1.jpg")
os.makedirs(RUN_DIR, exist_ok=True)
cv2.imwrite(frame_path, frame)
cap.release()

#click 4 images for homography -> later changed depending on the CCTV road
img = cv2.imread(frame_path)
if img is None:
    raise RuntimeError(f"Failed to load {frame_path}")

points = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("frame", img)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", img)
print("Left-click 4 ground-plane points (A,B,C,D) in order; press any key when done.")
cv2.setMouseCallback("frame", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
if len(points) < 4:
    raise ValueError("Need at least 4 points for homography.")

img_points = np.array(points[:4], dtype=np.float32)

#calculating homograhy

real_coords = np.array([
    [3.5, 0.0],
    [3.5, 8.0],
    [0.0,  0.0],
    [0.0,  8.0]
], dtype=np.float32)

H, _ = cv2.findHomography(img_points, real_coords, cv2.RANSAC, 2.0)
if H is None or H.shape != (3,3):
    raise ValueError("Homography solve failed.")
print("Homography:\n", H)

def natural_key(path):
    b = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", b)]

def img_to_ground(x_px, y_px, H):
    p = np.array([x_px, y_px, 1.0], dtype=float)
    q = H @ p
    X = q[0] / q[2]; Y = q[1] / q[2]
    return float(X), float(Y)

def parse_line_6col(line, W, Himg):
    # Expected (tracking): class cx cy w h track_id (normalized)
    parts = line.strip().split()
    if len(parts) != 6:
        return None
    cls, cx, cy, w, h, tid = parts
    cx, cy, w, h = map(float, (cx, cy, w, h))
    tid = int(float(tid))
    cx_px = cx * W; cy_px = cy * Himg; h_px = h * Himg
    # bottom center of bbox approximates contact point
    x_bc = cx_px
    y_bc = cy_px + 0.5 * h_px
    return tid, x_bc, y_bc

txt_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")), key=natural_key)
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {LABELS_DIR}")

#filters
MIN_TRACK_SECONDS = 2.0         
MAX_STEP_M = 12.0               
USE_MEDIAN_INSTANT = False

tracks_xy = defaultdict(list)
for fi, pth in enumerate(txt_files):
    t = fi / fps
    with open(pth, "r") as f:
        for line in f:
            parsed = parse_line_6col(line, W, Himg)
            if parsed is None:
                continue
            tid, x_bc, y_bc = parsed
            X, Y = img_to_ground(x_bc, y_bc, H)
            tracks_xy[tid].append((t, X, Y))

#calculating avg speed
per_track_avg_kmh = {}
per_track_inst = {}
for tid, seq in tracks_xy.items():
    if len(seq) < 2:
        continue
    seq.sort(key=lambda z: z[0])

    inst = []; dsum = 0.0; tsum = 0.0
    for (t0,X0,Y0),(t1,X1,Y1) in zip(seq[:-1], seq[1:]):
        dt = t1 - t0
        if dt <= 1e-6:
            continue
        dist = float(np.hypot(X1 - X0, Y1 - Y0))
        if dist > MAX_STEP_M:
            continue
        v = dist / dt
        inst.append(v); dsum += dist; tsum += dt

    if tsum < MIN_TRACK_SECONDS or not inst:
        continue
    v_mps = np.median(inst) if USE_MEDIAN_INSTANT else (dsum / tsum)
    per_track_avg_kmh[tid] = float(v_mps * 3.6)
    per_track_inst[tid] = inst

overall_avg_kmh = float(np.median(list(per_track_avg_kmh.values()))) if per_track_avg_kmh else 0.0
print(f"Tracks with valid speed: {len(per_track_avg_kmh)}")
print(f"Overall average speed: {overall_avg_kmh:.1f} km/h")
for tid, v in list(per_track_avg_kmh.items())[:10]:
    print(f"Track {tid}: {v:.1f} km/h")

#saving CSV
import csv, os
os.makedirs(CSV_OUT, exist_ok=True)
with open(f"{CSV_OUT}/per_track_speeds.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["track_id", "avg_kmh"])
    for tid in sorted(per_track_avg_kmh):
        w.writerow([tid, f"{per_track_avg_kmh[tid]:.3f}"])
print(f"Wrote {CSV_OUT}/per_track_speeds.csv")