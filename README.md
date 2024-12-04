# YOLOv5(or YOLOv7) object detection model deployment using flask
This repo contains example apps for exposing the [yolo5](https://github.com/ultralytics/yolov5) object detection model from 

The web is intended to quickly and accurately measure the amount of fruit on an apple.

An object detection deep learning model is used to predict the number of apples, and then a polynomial regression equation is used to output a more accurate count of apples.

[pytorch hub](https://pytorch.org/hub/ultralytics_yolov5/) via a [flask](https://flask.palletsprojects.com/en/1.1.x/) api/app.

## Flowchart
<p align="center"><img src="https://github.com/user-attachments/assets/d386be9e-2c1e-4b3a-9778-38d084d65f60" width="400" height="550" /></p>



## Web app
Simple app consisting of a form where you can upload an image, and see the inference result of the model in the browser. 

Run:

`$ python3 webapp.py --port 5000`

then visit [http://localhost:5000/](http://localhost:5000/) in your browser:

<p align="center">
  <img src="https://user-images.githubusercontent.com/51011169/235388468-77ba4fc3-02b4-414a-ba6c-e5452b33a2c5.png" width="400" height="550" />
  <img src="https://user-images.githubusercontent.com/51011169/235388476-5b8d9da2-4afd-4d82-9111-8ef3c823091f.png" width="400" height="550" />
</p>

Processed images are saved in the `static` directory with a datetime for the filename.




## Reference
- https://github.com/ultralytics/yolov5
- https://github.com/jzhang533/yolov5-flask (this repo was forked from here)
- https://github.com/avinassh/pytorch-flask-api-heroku

## Paper
Gwak, H. J., Jeong, Y., Chun, I. J., & Lee, C. H. (2024). Estimation of fruit number of apple tree based on YOLOv5 and regression model. Journal of IKEEE, 28(2), 150-157.
[KCI_FI003097618.pdf](https://github.com/user-attachments/files/18003027/KCI_FI003097618.pdf)
