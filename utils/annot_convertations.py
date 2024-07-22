import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil


def animalpose2yolo(json_annot_path: str,
                    dest_path: str,
                    cat_ids: tuple,
                    keypoints_count: int = 20,
                    keypoints_dim: int = 3,
                    rel_imgs_path: str = '../images/') -> None:
    
    # 5 = 1 class + 4 bbox dims 
    structure_annot_count = keypoints_count * keypoints_dim + 5   
    
    img_dest_path = os.path.join(dest_path, 'images')
    labels_path = os.path.join(dest_path, 'labels')

    # os.makedirs(dest_path, exist_ok=True)
    os.makedirs(img_dest_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    img_path = os.path.join(os.sep.join(json_annot_path.split(os.sep)[:-1]),
                            rel_imgs_path)
    
    with open(json_annot_path) as f:
        json_data = json.load(f)

    
    annotations = json_data['annotations']

    annots_map = dict()
    
    formater = lambda x: f'{x:.6f}'
    
    for annot_idx, annot in enumerate(annotations):
        img_id = annot['image_id']
        cat_id = annot['category_id']

        if cat_id in cat_ids:
                
            if img_id in annots_map:
                annots_map[img_id].append(annot_idx)
            else:
                annots_map[img_id] = [annot_idx]

    
    for img_id, annot_idxs in tqdm(annots_map.items(), desc="Annots generation"):

        img_name = json_data['images'][str(img_id)]
        annot_name = '.'.join(img_name.split('.')[:-1]) + '.txt'
        annot_img_path = img_path + img_name
        
        img = Image.open(annot_img_path)
        width, height = img.size
        
        str_annots = []
        
        for annot_idx in annot_idxs:       
            
            annot = annotations[annot_idx]
            # map ids 
            cat_id = annot['category_id'] - min(cat_ids)
            
            bbox = np.array(annot['bbox'],
                            dtype=np.float16)
            
            bbox[: 2] += bbox[2:] / 2  
            bbox[[0, 2]] /= width  
            bbox[[1, 3]] /= height
            
            if np.any(bbox > 1)  or np.any(bbox < 0): 
                continue
            
            keypoints = np.array(annot['keypoints'], 
                                 dtype=np.float16)
            keypoints[:, 0] /= width 
            keypoints[:, 1] /= height
            
            # keypoints with label 0 - not visible, 1 - partial visible, 2 - visible 
            # so, recalc
            keypoints[keypoints[:, 2] == 1, 2] = 2
            
            # keypoints norm  > 1 handle
            if np.any(keypoints[:, :2] > 1) or np.any(keypoints[:, :2] < 0):
                continue
            
            annot_str = list(map(formater, bbox.tolist() + keypoints.reshape(-1).tolist()))
            annot_str.insert(0, str(cat_id))
            
            if len(annot_str) != structure_annot_count:
                continue 
            
            annot_str = " ".join(annot_str) + '\n'
            str_annots.append(annot_str)

        if str_annots:
            
            with open(os.path.join(labels_path,annot_name),'w') as annot_f:
                annot_f.writelines(str_annots)
                    
            shutil.copy(annot_img_path,
                        os.path.join(img_dest_path,img_name))

            
            


