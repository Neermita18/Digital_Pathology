# coco_to_yolo.py
import os, json

IMG_W = IMG_H = 500

def coco_to_yolo(coco_json, label_out):
    os.makedirs(label_out, exist_ok=True)

    with open(coco_json) as f:
        coco = json.load(f)

    id2name = {i["id"]: i["file_name"] for i in coco["images"]}
    annos = {}

    for a in coco["annotations"]:
        annos.setdefault(a["image_id"], []).append(a)

    for img_id, fname in id2name.items():
        lines = []
        for a in annos.get(img_id, []):
            cls = a["category_id"]
            x, y, w, h = a["bbox"]

            xc = (x + w / 2) / IMG_W
            yc = (y + h / 2) / IMG_H
            nw = w / IMG_W
            nh = h / IMG_H

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        with open(os.path.join(label_out, fname.replace(".png", ".txt")), "w") as f:
            f.write("\n".join(lines))

    print(f"YOLO labels → {label_out}")
