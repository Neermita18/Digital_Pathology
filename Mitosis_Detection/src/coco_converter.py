# coco_converter.py
import os
import json

IMG_W = IMG_H = 500

def build_coco(patch_dir, out_json):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for fname in sorted(os.listdir(patch_dir)):
        if not fname.endswith(".png"):
            continue

        base = fname.replace(".png", "")
        json_path = os.path.join(patch_dir, base + ".json")
        if not os.path.exists(json_path):
            continue

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": IMG_W,
            "height": IMG_H
        })

        with open(json_path) as f:
            annos = json.load(f)

        for a in annos:
            x, y, w, h = a["bbox"]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": a["class_id"],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    categories = [{"id": i, "name": str(i)} for i in range(7)]

    with open(out_json, "w") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f, indent=2)

    print(f"COCO saved → {out_json}")
