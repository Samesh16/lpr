# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2,time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import splitext,basename
from keras.models import model_from_json
# from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from utils import detect_lp

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        
wpod_net_path = "models/wpod-net.json"
wpod_net = load_model(wpod_net_path)


# Load model architecture, weight and labels
json_file = open('models/MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("models/License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('models/license_character_classes.npy')
print("[INFO] Labels loaded successfully...")



def preprocess_image(img,resize=False):
    # img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

# test_image_path = "img1.jpg"
# vehicle, LpImg,cor = get_plate(test_image_path)

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
"""
if (len(LpImg)): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)


# Create sort_contours() function to grab the contour of each digit from left to right


cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 1<=ratio<=3.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)

print("Detect {} letters...".format(len(crop_characters)))


# fig = plt.figure(figsize=(15,3))
# cols = len(crop_characters)

final_string = ''
for i,character in enumerate(crop_characters):
    title = np.array2string(predict_from_model(character,model,labels))
    final_string+=title.strip("'[]")

print(final_string)

"""
def draw_box(image_path, cor, thickness=3,text=""): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    # vehicle_image = preprocess_image(image_path)
    cv2.polylines(image_path,[pts],True,(0,255,0),thickness)
    cv2.putText(image_path,text, (pts[0][0][0],pts[0][0][1]), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 2, (0,0,255), 2, cv2.LINE_AA)
    
    return image_path
# [[[-30 375]]

#  [[ 71 370]]

#  [[ 81 406]]

#  [[-20 412]]]
cap = cv2.VideoCapture("vid15.MOV")

i=0
start_time = time.time()
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")

while (cap.isOpened()):
    ret, frame = cap.read()     
    if ret == False:
        break
    try:
        # i+=1
        # if i%2==0:
        # """
            h, w, c = frame.shape
            vehicle, LpImg,cor = get_plate(frame)
            
            if (len(LpImg)): #check if there is at least one license image
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
                
                # convert to grayscale and blur the image
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(7,7),0)
                
                # Applied inversed thresh_binary 
                binary = cv2.threshold(blur, 180, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            
            cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append charater image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 30, 60

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1<=ratio<=3.5: # Only select contour with defined ratio
                    if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                        # Sperate number and gibe prediction
                        curr_num = thre_mor[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)
            final_string = ''
            if len(crop_characters)>=5:
                for i,character in enumerate(crop_characters):
                    title = np.array2string(predict_from_model(character,model,labels))
                    final_string+=title.strip("'[]")
                print(final_string)
                
           
            
            # if final_string:
            #     cv2.putText(frame, "Detected Number Plate: "+ final_string + "- Time:{}".format(round((time.time() - start_time)),2), (00,530), 
            #                     cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0,0,255), 1, cv2.LINE_AA)
                
            # """
            frame = draw_box(frame,cor,text=final_string)
            frame = cv2.resize(frame, (960, 540))
            cv2.putText(frame, "Detected Number Plate: "+ final_string, (00,530), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(frame,"Time: "+datetime.now().strftime("%H:%M:%S") , (10,30), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow('Vehicle plate Recognition',frame)
                    
    except:
        pass    
        frame = cv2.resize(frame, (960, 540))
        cv2.putText(frame, "Detected Number Plate: "+ "", (00,530), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0,0,255), 1, cv2.LINE_AA) 
        cv2.putText(frame,"Time: "+datetime.now().strftime("%H:%M:%S") , (10,30), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('Vehicle plate Recognition',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

cap.release()
cv2.destroyAllWindows()  
