# coco_converter.py
import os
import json
from extract_sqlite_annotations import load_annotations

def patches_to_coco(patch_dir, db_path, output_json):
    slides, coords, annos, classes = load_annotations(db_path)
    images = []
    annotations = []
    img_id_counter = 1
    anno_id_counter = 1

    class_map = {row['uid']:i+1 for i,row in classes.iterrows()}
    coco_categories = [{'id':v,'name':k} for k,v in class_map.items()]

    for fname in os.listdir(patch_dir):
        if not fname.endswith('.png'):
            continue
        img_id = img_id_counter
        images.append({'id': img_id, 'file_name': fname, 'height': 500, 'width': 500})
        img_id_counter += 1

        json_path = os.path.join(patch_dir, fname.replace('.png','.json'))
        if os.path.exists(json_path):
            with open(json_path) as f:
                annos_list = json.load(f)
            for a in annos_list:
                x,y,w,h = a['bbox']
                class_id = class_map.get(a['class_id'],1)
                annotations.append({'id': anno_id_counter,
                                    'image_id': img_id,
                                    'category_id': class_id,
                                    'bbox':[x,y,w,h],
                                    'area':w*h,
                                    'iscrowd':0})
                anno_id_counter += 1

    coco_dict = {'images': images, 'annotations': annotations, 'categories': coco_categories}
    with open(output_json,'w') as f:
        json.dump(coco_dict,f)
