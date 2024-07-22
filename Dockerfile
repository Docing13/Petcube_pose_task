FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y wget libgl1-mesa-glx libglib2.0-0
RUN mkdir -p /app/models

COPY app.py /app/

RUN wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1J6fLYmxoc0nfTTIhPrF3SfcomIM52iYP' -O /app/models/cat_dog_model.pt

RUN pip install --no-cache-dir Flask ultralytics opencv-python-headless 

EXPOSE 5000

ENV NAME Cat-Dog-Pose-API

CMD ["python", "app.py"]