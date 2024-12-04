import argparse
import io
import os
from PIL import Image
import datetime
import pandas as pd

import torch
from flask import Flask, render_template, request, redirect

#import lin_reg

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M"


#YOLOv5
@app.route("/", methods=["GET", "POST"]) #YOLOv5
def predict():
    if request.method == "POST": 
        if "file1" not in request.files:
            return redirect(request.url)
        file1 = request.files["file1"]
        if not file1:
            return
        
        if "file2" not in request.files:  
            return redirect(request.url)
        file2 = request.files["file2"]
        if not file2:
            return

        front_bytes = file1.read()
        back_bytes = file2.read()

        f_img = Image.open(io.BytesIO(front_bytes))
        f_results = model([f_img])

        b_img = Image.open(io.BytesIO(back_bytes))
        b_results = model([b_img])

        print(f"front: {f_results}") #image 1/1: 613x960 60 apples
                                     #Speed: 23.9ms pre-process, 667.5ms inference, 1.3ms NMS per image at shape (1, 3, 416, 640)
        
        f_count =  str(f_results).split()[-17] #Count number of apples
        b_count =  str(b_results).split()[-17]

        all_count =  int((int(f_count) + int(b_count)))
        all_count = 46.2 + 0.168 * all_count + 0.008254 * all_count**2 - 0.00004897 * all_count**3 + 0.0000001042 * all_count**4
        #polynomial regression

        print(int(all_count))
        
        f_results.render()
        real_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        f_img_savename = f"result/yolov5/front/{real_time}.png"
        Image.fromarray(f_results.ims[0]).save(f_img_savename)

        b_results.render() 
        b_img_savename = f"result/yolov5/back/{real_time}.png"
        Image.fromarray(b_results.ims[0]).save(b_img_savename)

        return render_template('index.html', all_count = all_count)
        
    return render_template("index.html")




'''
# YOLOv7
import os
from werkzeug.utils import secure_filename

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return redirect(request.url)

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return redirect(request.url)

        front_filename = secure_filename(file1.filename)
        back_filename = secure_filename(file2.filename)

        front_bytes = file1.read()
        back_bytes = file2.read()

        f_img = Image.open(io.BytesIO(front_bytes))
        f_result = model([f_img])
        f_result.save(os.path.join('result/yolov7/front/', front_filename))

        b_img = Image.open(io.BytesIO(back_bytes))
        b_result = model([b_img])
        b_result.save(os.path.join('result/yolov7/back', back_filename))

        f_count = f_result.class_count()
        b_count = b_result.class_count()
        
        all_count = int(f_count) + int(b_count)
        all_count = 46.2 + 0.168 * all_count + 0.008254 * all_count**2 - 0.00004897 * all_count**3 + 0.0000001042 * all_count**4

        return render_template('index.html', all_count = all_count)

    return render_template('index.html')
'''




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolo models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # Load the model
    #YOLOv7.ver
    #model = torch.hub.load("/home/hj/yolov7", 'custom', path_or_model='/home/hj/yolov7/runs/train/exp2/weights/best.pt', source='local')
    
    #YOLOv5.ver
    model = torch.hub.load("/home/hj/yolov5", 'custom', path='/home/hj/yolov5/runs/train/exp2/weights/best.pt', source='local')
    model.eval()
    
    app.run(host="0.0.0.0", port=args.port)
