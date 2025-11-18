import os

# ---------- PATHS ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

VIDEOS_DIR   = os.path.join(PROJECT_ROOT, "videos")
CALIB_DIR    = os.path.join(PROJECT_ROOT, "calib")
RUNS_DIR     = os.path.join(PROJECT_ROOT, "runs")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

YOLO_WEIGHTS = os.path.join(MODELS_DIR, "11_09.pt")
ACCIDENT_MODEL_PATH = os.path.join(MODELS_DIR, "accident_model.pt")

# ---------- HOMOGRAPHY ----------
USE_PRECALC_H    = True
SAVE_AFTER_SOLVE = True

# ---------- SPEED ----------
MIN_TRACK_SECONDS  = 2.0
MAX_STEP_M         = 12.0
N_FRAMES_SPEED     = 10

# ---------- ACCIDENT DNN ----------
ACCIDENT_THRESH     = 0.8
MIN_ALERT_GAP_SEC   = 2.0
ENABLE_ACCIDENT_LOG = True

"""
print("FILE =", __file__)
print("DIR =", os.path.dirname(__file__))
print("PARENT =", os.path.dirname(os.path.dirname(__file__)))
"""