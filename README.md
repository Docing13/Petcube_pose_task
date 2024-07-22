Pet pose estimation task

Build:
docker build -t cat-dog-pose .

Run:
docker run --gpus device=0 -p 5000:5000 cat-dog-pose

Testing:
curl -X POST -F 'image=@path/to/your/test_image.jpg' http://localhost:5000/predict
Or use Postman to send a POST request with the image file to http://localhost:5000/predict

