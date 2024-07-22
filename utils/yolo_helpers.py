import os
import glob 
import random
import shutil
from tqdm import tqdm


def parse_annotation(annot_path: str) -> list[dict[str, int | list[float] | list[list[float]]]]:
    
    items = []
    
    with open(annot_path,'r') as ann_f:
        annots = ann_f.readlines()
        
        for annot in annots:
            
            data = annot.split(' ')
            label = int(data.pop(0))
            
            data = list(map(float, data))
            
            bbox = data[:4]
            keypoints = [data[idx: idx + 3] for idx in range(4, len(data)-1, 3)]
            
            parsed = {
                'label':label,
                'bbox':bbox,
                'keypoints':keypoints
            }
            
            items.append(parsed)

    return items            
            
    
def annotations_labels(directory_path: str) -> list[int]:
    annotations = []

    for file_path in glob.glob(os.path.join(directory_path, '*.txt')):
        with open(file_path, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                if not parts:
                    continue  # Skip empty lines
                try:
                    # Extract class index, the first value in the line
                    class_label = int(parts[0])
                    annotations.append(class_label)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue
                
    return annotations


def count_classes(annotations: list[dict[str, int | list[float] | list[list[float]]]]) -> dict[int, int]:

    class_counts = dict()
    
    for class_label in annotations:
        
        if class_label in class_counts:
            class_counts[class_label] += 1
        else:
            class_counts[class_label] = 1
    
    return class_counts


def split_yolo_dataset(src_folder: str,
                       output_folder: str,
                       train_ratio: float = 0.8,
                       seed: int = 42):

    images_folder = os.path.join(src_folder,'images')
    labels_folder = os.path.join(src_folder,'labels')

    random.seed(seed)

    train_images_folder = os.path.join(output_folder, 'images', 'train')
    val_images_folder = os.path.join(output_folder, 'images', 'val')
    train_labels_folder = os.path.join(output_folder, 'labels', 'train')
    val_labels_folder = os.path.join(output_folder, 'labels', 'val')

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    image_files =[img.split(os.sep)[-1] for img in glob.glob(os.path.join(images_folder,"*"))]
    random.shuffle(image_files)

    train_count = int(train_ratio * len(image_files))

    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    def copy_(file, img_folder, label_folder): 
        shutil.copy(os.path.join(images_folder, file), img_folder)
        label_file = ('.').join(file.split('.')[:-1]) + '.txt'
        shutil.copy(os.path.join(labels_folder, label_file), label_folder)
    
    for file in tqdm(train_files, 
                     desc="Train generation"):
        copy_(file,
              train_images_folder,
              train_labels_folder)


    for file in tqdm(val_files,
                     desc="Val generation"):
        copy_(file,
              val_images_folder,
              val_labels_folder)