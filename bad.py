import os, yaml, glob, numpy as np, cv2

YAML_PATH = "datasets/dataset.yaml"

with open(YAML_PATH, "r") as f:
    cfg = yaml.safe_load(f)

def resolve(key):
    v = cfg[key]
    if os.path.isabs(v):
        return v
    root = cfg.get("path", "")
    return os.path.join(root, v) if root else v

val_images = resolve("val")
val_labels = val_images.replace("/images", "/labels")

print("val_images:", val_images, "exists?", os.path.isdir(val_images))
print("val_labels:", val_labels, "exists?", os.path.isdir(val_labels))

# Build image list from labels (how Ultralytics pairs them)
label_files = sorted(glob.glob(os.path.join(val_labels, "**/*.txt"), recursive=True))
bad, zero, missing_img = [], [], []

for lb in label_files:
    rel = os.path.splitext(os.path.relpath(lb, val_labels))[0]  # e.g. sub/xxx
    img = os.path.join(val_images, rel + ".png")                # PNG dataset
    if not os.path.isfile(img):
        missing_img.append((lb, img))
        continue
    size = os.path.getsize(img)
    if size == 0:
        zero.append(img); continue
    try:
        with open(img, "rb") as fh:
            data = fh.read()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            bad.append(img)
    except Exception:
        bad.append(img)

print(f"\nChecked {len(label_files)} label-image pairs.")
print(f"  Missing image files: {len(missing_img)}")
print(f"  Zero-byte PNGs:      {len(zero)}")
print(f"  Unreadable PNGs:     {len(bad)}")

if missing_img: print("\nMissing image examples:", *missing_img[:5], sep="\n - ")
if zero:        print("\nZero-byte examples:", *zero[:5], sep="\n - ")
if bad:         print("\nUnreadable PNGs:", *bad[:10], sep="\n - ")
