
"""
validate_yolos.py — robust validator for multiple Ultralytics YOLO versions on a custom dataset

What it does
------------
1) Optionally quarantines any unreadable PNG in your val split (cv2.imread check).
2) Monkey-patches Ultralytics image reader to a safe PIL-based loader (RGB->BGR).
3) Validates a list of pretrained weights on the same val set and logs metrics.
4) Writes a CSV summary next to the script.

Usage
-----
python validate_yolos.py \
  --data datasets/dataset.yaml \
  --val-dir datasets/valid/images \
  --labels-dir datasets/valid/labels \
  --weights yolov8n.pt yolov8s.pt yolo11n.pt yolo11s.pt \
  --imgsz 640 --batch 16 --device cpu --workers 0

Notes
-----
- Remove v9/v10 from --weights unless you have valid local files for them.
- Keep imgsz/batch/device/workers fixed across models for fair timing.
- If your dataset needs COCO→your-classes mapping, add `class_map:` in data.yaml.
"""

import argparse, os, sys, time, glob, shutil, traceback
import numpy as np

# --- Fast sanity check with OpenCV path-based read
def find_bad_images_cv2(img_dir, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    try:
        import cv2
    except Exception as e:
        print("[warn] OpenCV not available for scanning:", e)
        return []

    bad = []
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    paths.sort()
    for p in paths:
        img = cv2.imread(p)  # path-based read; robust on macOS
        if img is None:
            bad.append(p)
    return bad

def quarantine_files(bad_imgs, labels_dir, quarantine_dir="datasets/_quarantine"):
    os.makedirs(quarantine_dir, exist_ok=True)
    moved = []
    for p in bad_imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl = os.path.join(labels_dir, stem + ".txt")
        for q in (p, lbl):
            if os.path.exists(q):
                dst = os.path.join(quarantine_dir, os.path.basename(q))
                try:
                    shutil.move(q, dst)
                    moved.append(dst)
                except Exception as e:
                    print("[warn] fail move:", q, "->", dst, e)
    return moved

def patch_ultralytics_loader():
    from ultralytics.utils import patches
    from PIL import Image
    import numpy as np

    def pil_imread(path, flags=None):
        # Always return C-contiguous uint8 BGR ndarray
        with Image.open(path) as img:
            img = img.convert("RGB")                           # ensure 3 channels
            arr = np.asarray(img, dtype=np.uint8)              # guarantee uint8
        arr = arr[:, :, ::-1]                                  # RGB -> BGR
        return np.ascontiguousarray(arr)                       # enforce contiguity

    patches.imread = pil_imread
    print("[info] Ultralytics imread patched -> PIL (RGB) -> contiguous uint8 BGR ndarray")

def run_validation(data_yaml, weights, imgsz=640, batch=16, device='cpu', workers=0):
    from ultralytics import YOLO
    import pandas as pd

    rows = []
    for w in weights:
        print(f"\n=== Validating {w} ===", flush=True)
        try:
            t0 = time.time()
            m = YOLO(w)

            # Params (optional)
            try:
                params = sum(p.numel() for p in m.model.parameters())
            except Exception:
                params = None

            r = m.val(
                data=data_yaml,
                imgsz=imgsz,
                split="val",
                batch=batch,
                device=device,
                workers=workers,
                verbose=False,
                plots=False,
                save_json=False
            )

            # Some versions structure attrs slightly differently; be defensive
            box = getattr(r, "box", None)
            row = {
                "weights": w,
                "mAP50-95": float(getattr(box, "map", np.nan)) if box is not None else np.nan,
                "mAP50": float(getattr(box, "map50", np.nan)) if box is not None else np.nan,
                "Precision": float(getattr(box, "mp", np.nan)) if box is not None else np.nan,
                "Recall": float(getattr(box, "mr", np.nan)) if box is not None else np.nan,
                "infer_ms/img": r.speed.get("inference") if hasattr(r, "speed") else np.nan,
                "preproc_ms/img": r.speed.get("preprocess") if hasattr(r, "speed") else np.nan,
                "postproc_ms/img": r.speed.get("postprocess") if hasattr(r, "speed") else np.nan,
                "params(M)": None if params is None else round(params/1e6, 2),
                "time_s": round(time.time() - t0, 2),
                "error": ""
            }
            rows.append(row)
        except Exception as e:
            print(f"[!] {w} failed:\n{traceback.format_exc()}")
            rows.append({
                "weights": w, "mAP50-95": np.nan, "mAP50": np.nan,
                "Precision": np.nan, "Recall": np.nan,
                "infer_ms/img": np.nan, "preproc_ms/img": np.nan, "postproc_ms/img": np.nan,
                "params(M)": np.nan, "time_s": np.nan, "error": str(e)
            })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by="mAP50-95", ascending=False)
    out_csv = "yolo_version_val_summary.csv"
    df_sorted.to_csv(out_csv, index=False)
    print("\n== Summary ==")
    print(df_sorted.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    return df_sorted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--val-dir", required=True, help="validation images folder (to scan for bad files)")
    ap.add_argument("--labels-dir", required=True, help="validation labels folder (paired with images)")
    ap.add_argument("--weights", nargs="+", required=True, help="list of weight files")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--quarantine-bad", action="store_true", help="move unreadable images & labels to datasets/_quarantine")
    args = ap.parse_args()

    # 1) Find unreadable images (cv2.imread)
    print(f"[info] Scanning {args.val_dir} for unreadable images ...")
    bad = find_bad_images_cv2(args.val_dir)
    print(f"[info] Bad images (cv2.imread): {len(bad)}")
    if bad and args.quarantine_bad:
        moved = quarantine_files(bad, args.labels_dir)
        print(f"[info] Quarantined {len(moved)} files to datasets/_quarantine")
        # Clear Ultralytics caches if present
        for cache_name in ("labels.cache", "labels.npy", "labels.pkl"):
            cache_path = os.path.join(args.labels_dir, cache_name)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    print("[info] removed cache:", cache_path)
                except Exception as e:
                    print("[warn] could not remove cache:", cache_path, e)
        # Re-scan to verify
        bad2 = find_bad_images_cv2(args.val_dir)
        print(f"[info] Bad images after quarantine: {len(bad2)}")
        if bad2:
            print("[warn] Some unreadable files remain; proceeding anyway.")

    # 2) Patch Ultralytics image loader to PIL
    patch_ultralytics_loader()

    # 3) Run validation
    run_validation(
        data_yaml=args.data,
        weights=args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
