import cv2
from ultralytics import YOLO
import os, glob
import numpy as np
import re
from collections import defaultdict
import csv

NAME = "crash_ver2"
RAW_VIDEO = f"./videos/{NAME}.mov"
RUN_NAME  = f"{NAME}"
RUN_DIR   = f"./runs/detect/{RUN_NAME}"
VIDEO     = f"{RUN_DIR}/{NAME}.mp4"
LABELS_DIR = f"{RUN_DIR}/labels"
CSV_OUT   = RUN_DIR

# ================== HOMOGRAPHY SETTINGS ==================
CALIB_DIR = "./calib"
os.makedirs(CALIB_DIR, exist_ok=True)
H_FILE = os.path.join(CALIB_DIR, f"H_{RUN_NAME}.npz")

USE_PRECALC_H   = True   # Use precalculated version if exists
SAVE_AFTER_SOLVE = True  # Save new H after interactive solve

# ================== VIDEO METADATA ==================
cap = cv2.VideoCapture(RAW_VIDEO)
fps  = cap.get(cv2.CAP_PROP_FPS)
W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Himg = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if not fps or fps <= 0:
    raise RuntimeError("Could not read FPS from video; set fps manually.")
print(f"FPS={fps}, W={W}, H={Himg}")

# grab frame #0 for optional calibration UI
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ok, frame0 = cap.read()
if not ok:
    raise RuntimeError("Failed to read frame 0")
os.makedirs(RUN_DIR, exist_ok=True)
frame_path = os.path.join(RUN_DIR, "frame_1.jpg")
cv2.imwrite(frame_path, frame0)
cap.release()

# ================== HELPERS ==================
def natural_key(path):
    b = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", b)]

def img_to_ground(x_px, y_px, H):
    """Project image pixel -> ground plane (meters) using homography H."""
    p = np.array([x_px, y_px, 1.0], dtype=float)
    q = H @ p
    X = q[0] / q[2]; Y = q[1] / q[2]
    return float(X), float(Y)

def load_precalculated_H(path_npz, cur_W, cur_H):
    """
    Load homography saved as:
      np.savez(path, H=<3x3>, calib_width=<int>, calib_height=<int>)
    If the current video resolution (cur_W,cur_H) differs from calibration,
    scale H accordingly.
    """
    if not os.path.exists(path_npz):
        return None
    data = np.load(path_npz)
    H = data["H"]
    if H.shape != (3, 3):
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
    if H is None or H.shape != (3, 3):
        raise ValueError("Homography solve failed.")
    print("Homography:\n", H)
    return H

# ================== GET HOMOGRAPHY ==================
H = None
if USE_PRECALC_H:
    H = load_precalculated_H(H_FILE, W, Himg)
    if H is None:
        print(f"[homography] Precalculated file not found: {H_FILE}")

if H is None:
    #If not saved, solve homography
    H = solve_H_interactively(frame_path)
    if SAVE_AFTER_SOLVE:
        save_homography_npz(H_FILE, H, W, Himg)

# ================== SPEED SETTINGS ==================
MIN_TRACK_SECONDS = 2.0   # minimum total observed time to consider track "valid"
MAX_STEP_M        = 12.0  # reject teleport jumps
USE_MEDIAN_INSTANT = False  # if True, use median of inst speeds instead of avg

# per-track accumulators
track_history     = defaultdict(list)  # tid -> [(t, X, Y), ...]
track_dsum        = defaultdict(float) # total distance per track (m)
track_tsum        = defaultdict(float) # total time per track (s)
per_track_inst_all = defaultdict(list) # store inst v_mps if you want

per_track_avg_kmh = {}  # final

# ================== RUN YOLO + REAL-TIME SPEED + OVERLAY ==================
model = YOLO("11_09.pt")

print("[info] Starting YOLO tracking with real-time speed computation and overlay...")

results_gen = model.track(
    source=RAW_VIDEO,
    tracker="bytetrack.yaml",
    conf=0.4,
    persist=True,
    save=True,          # YOLO will still save annotated video and txts under RUN_DIR
    save_txt=True,
    show=False,         # we will handle imshow ourselves
    stream=True,        # generator, frame-by-frame
    name=RUN_NAME
)

cv2.namedWindow("Speed View", cv2.WINDOW_NORMAL)

for fi, result in enumerate(results_gen):
    t = fi / fps

    boxes = result.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        # still show frame if you want
        frame = result.orig_img
        cv2.imshow("Speed View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    ids   = boxes.id.cpu().numpy().astype(int)
    xywhn = boxes.xywhn.cpu().numpy()   # normalized cx, cy, w, h
    xyxy  = boxes.xyxy.cpu().numpy()    # pixel coords for overlay

    # for overlay speeds on this frame
    frame_inst_speed = {}  # tid -> v_kmh

    # ---- compute ground positions + inst speeds ----



    """    
    for (tid, (cx, cy, w, h)) in zip(ids, xywhn):
        cx_px = cx * W
        cy_px = cy * Himg
        h_px  = h  * Himg

        # bottom center of bbox
        x_bc = cx_px
        y_bc = cy_px + 0.5 * h_px

        # image -> ground
        X, Y = img_to_ground(x_bc, y_bc, H)
        track_history[tid].append((t, X, Y))

        seq = track_history[tid]
        if len(seq) >= 2:
            (t0, X0, Y0), (t1, X1, Y1) = seq[-2], seq[-1]
            dt = t1 - t0
            if dt > 1e-6:
                dist = float(np.hypot(X1 - X0, Y1 - Y0))
                if dist <= MAX_STEP_M:  # ignore teleports
                    v_mps = dist / dt
                    v_kmh = v_mps * 3.6
                    frame_inst_speed[tid] = v_kmh

                    track_dsum[tid] += dist
                    track_tsum[tid] += dt
                    per_track_inst_all[tid].append(v_mps)
    """
    # Only compute speed every N frames for stability
    N = 10
    if fi % N != 0:
        continue

    # Compute instantaneous speed only on every Nth frame
    for (tid, (cx, cy, w, h)) in zip(ids, xywhn):
        cx_px = cx * W
        cy_px = cy * Himg
        h_px  = h  * Himg

        x_bc = cx_px
        y_bc = cy_px + 0.5 * h_px

        # Project to ground plane
        X, Y = img_to_ground(x_bc, y_bc, H)
        track_history[tid].append((t, X, Y))

        seq = track_history[tid]
    
        if len(seq) >= N + 1:       #using N frames to calculate speed
            (t0, X0, Y0) = seq[-(N+1)]
            (t1, X1, Y1) = seq[-1]
            dt = t1 - t0
            if dt > 1e-6:
                dist = float(np.hypot(X1 - X0, Y1 - Y0))
                if dist <= MAX_STEP_M * N: 
                    v_mps = dist / dt
                    v_kmh = v_mps * 3.6
                    frame_inst_speed[tid] = v_kmh
                    track_dsum[tid] += dist
                    track_tsum[tid] += dt
                    per_track_inst_all[tid].append(v_mps)



    # ---- YOLO's annotated frame ----
    frame = result.plot()  # BGR image with YOLO boxes

    # ---- overlay speed text for each track ----
    for tid, box_xyxy in zip(ids, xyxy):
        if tid not in frame_inst_speed:
            continue
        v_kmh = frame_inst_speed[tid]
        x1, y1, x2, y2 = box_xyxy
        cx = int((x1+ x2) / 2)
        cy = int((y1 + y2) / 2)
        text = f"{v_kmh:.1f} km/h"
        org = (cx, cy - 5)  # above the box
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),  
            2,
            cv2.LINE_AA
        )

    # ---- show frame ----
    cv2.imshow("Speed View", frame)
    # press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[info] 'q' pressed, stopping early.")
        break

cv2.destroyAllWindows()

# ================== AFTER PROCESSING: FINAL AVERAGE SPEEDS ==================
for tid, tsum in track_tsum.items():
    if tsum < MIN_TRACK_SECONDS:
        continue
    if USE_MEDIAN_INSTANT:
        inst_list = per_track_inst_all[tid]
        if not inst_list:
            continue
        v_mps = float(np.median(inst_list))
    else:
        dsum = track_dsum[tid]
        v_mps = dsum / tsum
    per_track_avg_kmh[tid] = v_mps * 3.6

if per_track_avg_kmh:
    overall_avg_kmh = float(np.median(list(per_track_avg_kmh.values())))
else:
    overall_avg_kmh = 0.0

print(f"Tracks with valid speed: {len(per_track_avg_kmh)}")
print(f"Overall average speed: {overall_avg_kmh:.1f} km/h")
for tid, v in list(per_track_avg_kmh.items())[:10]:
    print(f"Track {tid}: {v:.1f} km/h")

# ================== SAVE CSV ==================
os.makedirs(CSV_OUT, exist_ok=True)
csv_path = os.path.join(CSV_OUT, "per_track_speeds.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["track_id", "avg_kmh"])
    for tid in sorted(per_track_avg_kmh):
        w.writerow([tid, f"{per_track_avg_kmh[tid]:.3f}"])
print(f"Wrote {csv_path}")