# coco_converter.py
import os
import json
from extract_sqlite_annotations import load_annotations


def patches_to_coco(patch_dir, output_json):
    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    images = []
    annotations = []
    img_id_counter = 1
    anno_id_counter = 1
    categories = {}

    # Loop through all images in patch directory
    for fname in os.listdir(patch_dir):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(patch_dir, fname)
        json_path = img_path.replace(".png", ".json")

        # Create image entry
        images.append({
            "id": img_id_counter,
            "file_name": fname,
            "height": 500,
            "width": 500
        })

        # Load annotations if exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                annos_list = json.load(f)

            for a in annos_list:
                x, y, w, h = a["bbox"]
                class_id = int(a["class_id"])

                categories[class_id] = True  # track all class ids

                annotations.append({
                    "id": anno_id_counter,
                    "image_id": img_id_counter,
                    "category_id": class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })

                anno_id_counter += 1

        img_id_counter += 1

    # Create COCO-style categories
    coco_categories = [
        {"id": cid, "name": str(cid)} for cid in sorted(categories.keys())
    ]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"COCO JSON saved → {output_json}")
