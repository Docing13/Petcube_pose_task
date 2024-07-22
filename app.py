from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os 
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.getcwd(),'models','cat_dog_model.pt')

model = YOLO(model_path).to(device)

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    with torch.no_grad():
        results = model(img)[0]
    
    predictions = []
    
    bboxes = results.boxes.xywh
    cat_ids = results.boxes.cls
    scores = results.boxes.conf   
    instances_keypoints = results.keypoints
    
    for bbox, cat_id, score, instance_keypoints in zip(
        bboxes, cat_ids, scores, instances_keypoints):
        
        bbox = bbox.tolist()
        cat_id = int(cat_id.item())
        score = score.item()
        cat_name = results.names[cat_id]
        instance_keypoints = instance_keypoints.data[0].tolist()
        
        prediction = {
            'cat_id': cat_id,
            'score': score,
            'cat_name': cat_name,
            'bbox': bbox,
            'keypoints': instance_keypoints
        }

        predictions.append(prediction)
    
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
