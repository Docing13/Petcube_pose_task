import cv2 
import os
import numpy as np
from utils.yolo_helpers import parse_annotation
import glob


def draw_annotation(annot_path: str, 
                    classes: tuple[str],
                    rel_imgs_path: str = 'images') -> np.ndarray:
    
    annot = parse_annotation(annot_path)
    
    path_ = os.sep.join(annot_path.split(os.sep)[:-2])
    
    name = annot_path.split(os.sep)[-1]
    name = '.'.join(name.split('.')[:-1])
    
    image_path = glob.glob(os.path.join(path_, 
                                        rel_imgs_path,
                                        name + '.*'))[0]
    image = cv2.imread(image_path)
    
    h, w, _ = image.shape
    
    for annot_item in annot:
        
        x, y, width, height = annot_item['bbox']
        
        x *= w
        y *= h
        
        width *= w
        height *= h

        x_min = int(x - width / 2)
        y_min = int(y - height / 2)
        x_max = int(x + width / 2) 
        y_max = int(y + height / 2)
        
        text_label = classes[annot_item['label']]
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

        text_size, _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(image, (x_min, y_min - text_h - 5), (x_min + text_w, y_min), (0, 0, 255), -1)
        cv2.putText(image, text_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for keypoint in annot_item['keypoints']:
            px, py, visibility = keypoint
            
            visibility = int(visibility)
            px, py = int(px * w), int(py * h)

            if visibility > 0:  
                cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
                
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    return image


