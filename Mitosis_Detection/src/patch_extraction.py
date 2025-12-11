import os
import json
from collections import defaultdict
from read_dicom_wsi import TileWSI
from extract_sqlite_annotations import load_annotations
from PIL import Image
import numpy as np

PATCH_SIZE = 500
BBOX_SIZE = 16

def is_white_patch(patch, brightness_thresh=240, std_thresh=10):
    gray = patch.mean(axis=2)
    return gray.mean() > brightness_thresh and gray.std() < std_thresh

def extract_patches_with_classes(db_path, slides_dir, output_dir):
    slides, coords, annos, classes = load_annotations(db_path)
    os.makedirs(output_dir, exist_ok=True)


    for _, slide_row in slides.iterrows():
        slide_path = os.path.join(slides_dir, slide_row['filename'])
        slide_uid = slide_row['uid']
        
        # 1. Get annotations for this slide
        slide_coords_all = coords[coords['slide'] == slide_uid]

        if slide_coords_all.empty:
            continue

        try:
            slide = TileWSI(slide_path)
        except Exception as e:
            print(f"Error loading {slide_path}: {e}")
            continue

        # 2. INSTANT GROUPING
        # We build a list of ONLY the patches that have data.
        # We will ignoring the empty space entirely.
        print(f"Grouping annotations for {slide_row['filename']}...")
        patch_lookup = defaultdict(list)
        
        for _, row in slide_coords_all.iterrows():
            px = (int(row['coordinateX']) // PATCH_SIZE) * PATCH_SIZE
            py = (int(row['coordinateY']) // PATCH_SIZE) * PATCH_SIZE
            patch_lookup[(px, py)].append(row)

        print(f"  -> Processing {len(patch_lookup)} patches...")

        # 3. LOOP ONLY THE VALID PATCHES
        # Instead of looping x and y (which scans empty space),
        # we only loop through the keys we just found.
        for (x, y), rows_in_patch in patch_lookup.items():
            
            # Read image
            patch = slide.read_region((x, y), (PATCH_SIZE, PATCH_SIZE))
            if not isinstance(patch, np.ndarray):
                patch = np.array(patch)

            # Check visuals
            if is_white_patch(patch):
                continue  

            anno_list = []
            for row in rows_in_patch:
                anno_id = row['annoId']
                class_row = annos[annos['uid'] == anno_id]
                if class_row.empty:
                    continue
                
                # Convert to int to prevent JSON crash
                class_id = int(class_row['agreedClass'])
                
                cx = int(row['coordinateX'] - x)
                cy = int(row['coordinateY'] - y)
                
                bbox = [
                    int(cx - BBOX_SIZE//2), 
                    int(cy - BBOX_SIZE//2), 
                    int(BBOX_SIZE), 
                    int(BBOX_SIZE)
                ]
                
                anno_list.append({'bbox': bbox, 'class_id': class_id})

            if not anno_list:
                continue

            # Save
            patch_fname = f"{slide_row['filename'].replace('.dcm','')}_{x}_{y}.png"
            patch_path = os.path.join(output_dir, patch_fname)
            
            Image.fromarray(patch).save(patch_path)

            with open(patch_path.replace('.png','.json'), 'w') as f:
                json.dump(anno_list, f)