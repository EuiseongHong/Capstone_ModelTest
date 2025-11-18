import cv2
from ultralytics import YOLO
import os, glob
import numpy as np
import re
from collections import defaultdict
import json

NAME = "crash_ver2"
RAW_VIDEO = f"./videos/{NAME}.mov"     
RUN_NAME  = f"{NAME}"
RUN_DIR   = f"./runs/detect/{RUN_NAME}"
VIDEO     = f"{RUN_DIR}/{NAME}.mp4"   
LABELS_DIR = f"{RUN_DIR}/labels"
CSV_OUT   = RUN_DIR

#precalculated homography here
CALIB_DIR = "./calib"
os.makedirs(CALIB_DIR, exist_ok=True)
H_FILE = os.path.join(CALIB_DIR, f"H_{RUN_NAME}.npz")


USE_PRECALC_H = True   #Use precalculated version
SAVE_AFTER_SOLVE = True

#load YOLO here
model = YOLO("better.pt")
results = model.track(
    source=RAW_VIDEO,
    tracker="bytetrack.yaml",
    conf=0.4,
    persist=True,
    save=True,          # save to RUN_DIR
    save_txt=True,      # saves per-frame labels
    show=False,
    stream=False,
    name=RUN_NAME
)

# ====== VIDEO METADATA ======
cap = cv2.VideoCapture(RAW_VIDEO)
fps  = cap.get(cv2.CAP_PROP_FPS)
W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Himg = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if not fps or fps <= 0:
    raise RuntimeError("Could not read FPS from video; set fps manually.")
print(f"FPS={fps}, W={W}, H={Himg}")

# grab frame #0 for optional calibration UI
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Failed to read frame 0")
frame_path = os.path.join(RUN_DIR, "frame_1.jpg")
os.makedirs(RUN_DIR, exist_ok=True)
cv2.imwrite(frame_path, frame)
cap.release()

# ====== HOMOGRAPHY HELPERS ======
def natural_key(path):
    b = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", b)]

def img_to_ground(x_px, y_px, H):
    """Project image pixel -> ground plane (meters) using homography H."""
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
    # bottom center of bbox approximates ground contact point
    x_bc = cx_px
    y_bc = cy_px + 0.5 * h_px
    return tid, x_bc, y_bc

def load_precalculated_H(path_npz, cur_W, cur_H):
    """
    Load homography saved as:
      np.savez(path, H=<3x3>, calib_width=<int>, calib_height=<int>)
    If the current video resolution (cur_W,cur_H) differs from calibration,
    we scale H accordingly: H_new = H_old * S, where
       S = diag(W0/W1, H0/H1, 1)
    because x_old = S * x_new.
    """
    if not os.path.exists(path_npz):
        return None
    data = np.load(path_npz)
    H = data["H"]
    if H.shape != (3,3):
        raise ValueError("Loaded homography is not 3x3.")
    W0 = int(data["calib_width"])
    H0 = int(data["calib_height"])
    if (W0 != cur_W) or (H0 != cur_H):
        sx = W0 / float(cur_W)
        sy = H0 / float(cur_H)
        S = np.array([[sx, 0, 0],
                      [0,  sy, 0],
                      [0,  0,  1]], dtype=float)
        H = H @ S
        print(f"[homography] Adjusted for resolution: calib=({W0},{H0}) -> current=({cur_W},{cur_H})")
    else:
        print("[homography] Resolution matches calibration; no scaling applied.")
    return H

def save_homography_npz(path_npz, H, calib_W, calib_H):
    np.savez(path_npz, H=H, calib_width=int(calib_W), calib_height=int(calib_H))
    print(f"[homography] Saved to {path_npz}")

def solve_H_interactively(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to load {img_path}")

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

    # === EDIT THESE to your real-world coordinates (meters) for the clicked points ===
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
    return H

# ====== GET HOMOGRAPHY (precalc or interactive) ======
H = None
if USE_PRECALC_H:
    H = load_precalculated_H(H_FILE, W, Himg)
    if H is None:
        print(f"[homography] Precalculated file not found: {H_FILE}")

if H is None:
    # Fall back to interactive solve
    H = solve_H_interactively(frame_path)
    if SAVE_AFTER_SOLVE:
        save_homography_npz(H_FILE, H, W, Himg)

# ====== READ LABELS & PROJECT TO GROUND ======
txt_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")), key=natural_key)
if not txt_files:
    raise FileNotFoundError(f"No .txt files found in {LABELS_DIR}")

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

# ====== SPEED COMPUTATION ======
per_track_avg_kmh = {}
per_track_inst = {}
for tid, seq in tracks_xy.items():
    if len(seq) < 2:
        continue
    seq.sort(key=lambda z: z[0])

    inst = []; dsum = 0.0; tsum = 0.0
    for (t0,X0,Y0),(t1,X1,Y1) in zip(seq[:-1], seq[1:]):
        dt = t1 - t0
        if dt <= 1e-6:  # skip duplicates
            continue
        dist = float(np.hypot(X1 - X0, Y1 - Y0))
        if dist > MAX_STEP_M:  # skip teleports
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

# ====== SAVE CSV ======
import csv
os.makedirs(CSV_OUT, exist_ok=True)
with open(f"{CSV_OUT}/per_track_speeds.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["track_id", "avg_kmh"])
    for tid in sorted(per_track_avg_kmh):
        w.writerow([tid, f"{per_track_avg_kmh[tid]:.3f}"])
print(f"Wrote {CSV_OUT}/per_track_speeds.csv")
