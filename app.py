from flask import Flask, Response
app = Flask(__name__)
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from detect_number_plate import detect
from flask_pymongo import PyMongo, pymongo
from datetime import datetime

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['mp4','MOV','webm'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config["MONGO_URI"] = "mongodb+srv://hunnurjirao:hunnur000@clusterone.kazmc.mongodb.net/VehiclesDB?retryWrites=true&w=majority"
mongo = PyMongo(app,tls=True, tlsAllowInvalidCertificates=True)

plates_db = mongo.db.numberPlates

def allowed_file(filename):
    	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def home():
	return render_template('upload.html')

def gen_frames(vid): 
    pat = "D:/Projects/16. Moving Vehicles Number Plate Recognition using Deep Learning/Updated Code/static/uploads/"+str(vid)
    camera = cv2.VideoCapture(pat)
    i=0
    plate_numbers=[]
    print("===========>START<============")
    veh = plates_db.find_one({'filename': str(vid)})
    if not veh:
        try:
            plates_db.insert_one({
                "filename":str(vid)
                # "plate_numbers": plate_numbers
            })
        except :
            print("Something went wrong")
    u_veh = plates_db.find_one({'filename': str(vid)})
        
    while True:
        i+=1
        success, frame = camera.read()  
        if not success:
            print("===========>DONE<============")
            break
        else:
            if i%2==0:
                frame,plate_num = detect(frame)
                if plate_num != '' and plate_num not in plate_numbers:
                    cur_time = datetime.now().strftime("%H:%M:%S")
                    a={"time":cur_time,"plate_number":plate_num}
                    
                    plates_db.update_one(u_veh,{"$push" : {"plate_numbers":a }})  
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
    
        

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(gen_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		return redirect(request.url)


if __name__ == "__main__":
    app.run(port=5000,debug=True)