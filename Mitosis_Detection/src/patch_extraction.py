# patch_extraction.py
import os
import json
from collections import defaultdict
import numpy as np
from PIL import Image
from read_dicom_wsi import TileWSI
from extract_sqlite_annotations import load_annotations

PATCH_SIZE = 500
BOX_SIZE = 32

def is_white_patch(patch, mean_thresh=240, std_thresh=10):
    gray = patch.mean(axis=2)
    return gray.mean() > mean_thresh and gray.std() < std_thresh

def clip_bbox(x, y, w, h):
    x = max(0, x)
    y = max(0, y)
    w = min(PATCH_SIZE - x, w)
    h = min(PATCH_SIZE - y, h)
    return [x, y, w, h]

def extract_split(db_path, slides_dir, out_dir):
    slides, coords, annos, classes = load_annotations(db_path)
    os.makedirs(out_dir, exist_ok=True)

    for _, slide in slides.iterrows():
        slide_path = os.path.join(slides_dir, slide["filename"])
        if not os.path.exists(slide_path):
            continue

        slide_uid = slide["uid"]
        slide_coords = coords[coords["slide"] == slide_uid]
        if slide_coords.empty:
            continue

        try:
            wsi = TileWSI(slide_path)
        except Exception as e:
            print(f"Failed to load {slide_path}: {e}")
            continue

        patch_map = defaultdict(list)
        for _, r in slide_coords.iterrows():
            px = int(r["coordinateX"] // PATCH_SIZE) * PATCH_SIZE
            py = int(r["coordinateY"] // PATCH_SIZE) * PATCH_SIZE

            patch_map[(px, py)].append(r)

        for (px, py), rows in patch_map.items():
            patch = wsi.read_region((px, py), (PATCH_SIZE, PATCH_SIZE))
            patch = np.array(patch)

            if is_white_patch(patch):
                continue

            ann_list = []
            for r in rows:
                anno = annos[annos["uid"] == r["annoId"]]
                if anno.empty:
                    continue

                cls = int(anno.iloc[0]["agreedClass"]) - 1

                cx = int(r["coordinateX"] - px)
                cy = int(r["coordinateY"] - py)

                x = cx - BOX_SIZE // 2
                y = cy - BOX_SIZE // 2
                bbox = clip_bbox(x, y, BOX_SIZE, BOX_SIZE)

                ann_list.append({
                    "bbox": bbox,
                    "class_id": cls
                })

            if not ann_list:
                continue

            base = slide["filename"].replace(".dcm", "")
            fname = f"{base}_{px}_{py}"

            Image.fromarray(patch).save(os.path.join(out_dir, fname + ".png"))
            with open(os.path.join(out_dir, fname + ".json"), "w") as f:
                json.dump(ann_list, f)

def extract_train_val(db_path, raw_root, out_root):
    print("Extracting TRAIN...")
    extract_split(db_path,
                  os.path.join(raw_root, "train"),
                  os.path.join(out_root, "train"))

    print("Extracting VAL...")
    extract_split(db_path,
                  os.path.join(raw_root, "val"),
                  os.path.join(out_root, "val"))
