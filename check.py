import os, yaml
cfg = yaml.safe_load(open("datasets/dataset.yaml"))

def resolve(img_key):
    p = cfg.get("path", "")
    v = cfg[img_key]
    return v if os.path.isabs(v) else os.path.join(p, v)

val_images = resolve("val")
val_labels = val_images.replace("/images", "/labels")

print("val images:", val_images, "exists?", os.path.isdir(val_images))
print("val labels:", val_labels, "exists?", os.path.isdir(val_labels))


#git test