import os
import cv2
import sys
# import time
from time import time
import csv
import datetime
import pandas as pd
# import matplotlib.pyplot as plt
import face_recognition
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream #to access either our vediofile on disk
from imutils.video import VideoStream #or built in webcam/USB camera
from imutils import face_utils
import argparse
import imutils
import dlib




import numpy as np
import cv2 as cv
from keras.models import load_model
from operator import add
from tkinter import Tk, mainloop, TOP
from tkinter import messagebox

from tkinter import Tk, mainloop, TOP 
from tkinter.ttk import Button 
from time import time
from tkinter import * 
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.ttk import *

import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.animation as animation
from matplotlib import style


fields1 = ['concentration', 'time']
filename1 = "eye_blink.csv"
with open(filename1, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields1)
fields = ['concentration', 'time']
filename = "head_movement.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
# helper modules
from drawFace import draw
import reference_world as world
counter=0
count=0
# PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
args={'shape_predictor' :'shape_predictor_68_face_landmarks.dat'}
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
eye_ar_consec_frames1=15
eye_blinking_threshold=17
eye_blinking_threshold1=3
# initialize the frame counters and the total number of blinks
is_attentive=False

PREDICTOR_PATH='shape_predictor_68_face_landmarks.dat'
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--focal",
                    type=float, default=1,
                    help="Callibrated Focal Length of the camera")
parser.add_argument("-s", "--camsource", type=int, default=0,
	help="Enter the camera source")

args = vars(parser.parse_args())

face3Dmodel = world.ref3DModel()



def convert_dtype(x):
    x_float = x.astype('float32')
    return x_float

def normalize(x):
    x_n = (x - 0)/(255)
    return x_n


def reshape(x):
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r
#Defining colours for each mood
colors = {'neutral':(255, 255, 255), 'angry':(0, 0, 255), 'disgust':(0, 139, 139), 'fear':(125, 125, 125), 'happy':(0, 255, 255), 'sad':(255, 0, 0), 'surprised':(255, 245, 0)}

#giving value to each mood
imotions = {0:'angry', 1:'fear', 2:'happy', 3:'sad',
               4:'surprised', 5:'neutral'}


file = open("example.txt","r+")
file. truncate(0)
file. close()


style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    xs=xs[-30:]
    ys=ys[-30:]
    ax1.set_ylim([0,100])
    ax1.plot(xs, ys,color='green', linewidth = 2,marker='o', markerfacecolor='blue', markersize=4)
    plt.xlabel('Time ----->',color='blue')
    plt.ylabel('Attentiveness',color='blue')
    plt.tight_layout()
    

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.tight_layout()
plt.show()


model = load_model('epoch_75.hdf5')
#Video capturing
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# PREDICTOR_PATH='shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# def main():
COUNTER = 0
TOTAL = 0
TOTAL_1 = 0
bool_1 = True
condition_1 = False
condition_2 = False
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(args["camsource"])
face_found=True
not_there = False
bool1=True
wwe=1
finalArr=[0,0,0,0,0,0]
x_axis=[]
y_axis=[]
ix_axis=[]
iy_axis=[]
pie_arr=[0,0,0,0,0,0]
pie_count=0
lovy=0
p=0
cham=0
champs=0
while True:
        # ret,img = cam.read()
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        ret, img = cam.read()
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        rects = detector(gray, 0)
        # for face in rects:
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            roi_gray = cv.resize(roi_gray, (48, 48), interpolation = cv.INTER_AREA)
            roi_gray = convert_dtype(np.array([roi_gray]))
            roi_gray = normalize(roi_gray)
            roi_gray = reshape(roi_gray)
            pr = model.predict(roi_gray)[0]
           # print(pr)
            max_emo = np.argmax(pr)
            cv.rectangle(img,(x,y),(x+w,y+h), colors[imotions[max_emo]], 2)
            cv.rectangle(img,(x,y+h),(x+w,y+h+170),(128,128,128), -1)
            cv.rectangle(img,(x,y),(x+w,y+h+130), colors[imotions[max_emo]], 2)
            cv.rectangle(img,(x,y),(x+w,y+h+170), colors[imotions[max_emo]], 2)

            counter = 0

            lovy=lovy+1

            pie_count=pie_count+1
            pie_arr = list(map(add, pr , pie_arr))

            finalArr = list(map(add, pr , finalArr))


        
            if lovy%10==0:
                x_axis.append(p)
                ix_axis.append(p)  
                y_axis.append(champs)
                iy_axis.append(cham)
                p=p+1



            for i in range(len(pr)+1):


                if i!=6:
                    cv.rectangle(img, (x, y+h+counter+7), (x + int(w * pr[i]), y+h+counter+28), colors[imotions[i]], -2)    
                    counter += 20
                    cv.putText(img, str(int(pr[i]*100)), (x + int(w * pr[i]), (y + h +counter+5)), cv.FONT_HERSHEY_SIMPLEX, 0.50,(51,0,0) , 1)
                    if i != 5:
                        cv.putText(img, imotions[i], (x, (y + h +counter+5)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,0) , 1)
                    else:
                        cv.putText(img, imotions[i], (x, (y + h +counter+5)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 0) , 1)


                if i==6:
                    counter +=20;

                    if (pr[2] >0.4 or pr[5] >0.4) and (pr[0] < 0.35 and pr[3] <0.35):
                        cv.rectangle(img, (x, y+h+counter-4), (x + w, y+h+counter+24),(0,255,0), -2)
                        cv.putText(img,'Attentive', (x+10, (y + h +counter+20)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 0) , 2)
                        if pr[2]>0.5:
                            champs=int(pr[2]*100)
                        elif pr[5]>0.5:
                            champs=int(pr[5]*100)
                        else:
                            champs=int(((pr[2]+pr[5])/2)*100)


                        file1 = open("example.txt","a")
                        file1. write("\n") 
                        file1.write(str(wwe)+","+str(champs)) 
                        wwe=wwe+1
                        file1.close()
                    
                    
                        cv.putText(img,str(champs)+'%', (x+120, (y + h +counter+20)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 0) , 2)

                    else:
                        cv.rectangle(img, (x, y+h+counter-4), (x + w, y+h+counter+24),(0,0,255), -2)
                        cv.putText(img,'In-Attentive', (x+5, (y + h +counter+20)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 0) , 2)
                        if pr[0]>0.5:
                            cham=int(pr[0]*100)
                        elif pr[3]>0.5:
                            cham=int(pr[3]*100)
                        elif pr[4]>0.5:
                            cham=int(pr[4]*100)
                        else:
                            cham=int(((pr[0]+pr[3])/2)*100)

                        cv.putText(img,str(cham)+'%', (x+150, (y + h +counter+20)), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 0) , 2)

            #cv.circle(img, ((x + w//2), (y + h//2)), int(((h*h + w*w)**0.5)//2), colors[imotions[pr]], 2)
            #cv.putText(img, imotions[pr], ((x + w//2), (y + h//2) - int(((h*h + w*w)**0.5)//2)), cv.FONT_HERSHEY_SIMPLEX, 1, colors[imotions[pr]], 1)
        
        
        #HEAD_EYE CODE
        
        
        if (bool_1):
            a = time()
            bool_1 = False
        GAZE="Face Not Found"
        # ret, img = cap.read()
        # img = imutils.resize(img, width=450)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # rects = detector(gray, 0)
        if not ret:
            print(f"[ERROR - System]Cannot read from source: {args['camsource']}")
            break
        for face in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, face)
            shape1 = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape1[lStart:lEnd]
            rightEye = shape1[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= eye_ar_consec_frames1:
                    TOTAL_1 += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                # if COUNTER >= eye_ar_consec_frames1:
                #     TOTAL_1 +=1

                # reset the eye frame counter
                COUNTER = 0
                b = time()
                diff = b - a
                if (diff >= 60):
                    condition_2 = False
                    condition_1 = False
                    TOTAL = 0
                    TOTAL_1 = 0
                    bool_1 = True
                    with open(filename1, 'a') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        if is_attentive == True:
                            rows = [['ATTENTIVE', datetime.datetime.now()]]
                        else:
                            rows = [['INATTENTIVE', datetime.datetime.now()]]
                        csvwriter.writerows(rows)
                # if(TOTAL>eye_blinking_threshold):
                #     cv2.putText(frame, "INATTENTIVE", (10, 60),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # else:
                #     cv2.putText(frame, "ATTENTIVE", (10, 60),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            if (TOTAL > eye_blinking_threshold):
                condition_1 = True
            if (TOTAL_1 > eye_blinking_threshold1):
                condition_2 = True
                # cv2.putText(frame, "INATTENTIVE", (10, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # else:
            #     cv2.putText(frame, "ATTENTIVE", (10, 60),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if (condition_1 or condition_2):
                is_attentive = False
                cv2.putText(img, "INATTENTIVE", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                is_attentive = True
                cv2.putText(img, "ATTENTIVE", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Blinks: {}".format(TOTAL), (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Larger_Blinks: {}".format(TOTAL_1), (250, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        # popular feature extraction technique(It is a simplified representation of the image that
        # contains only the most important information about the image.)
        # for images – Histogram of Oriented Gradients, or HOG as its commonly known
        # faces = face_recognition.face_locations(img, model="hog")
        # print(faces)
        if not rects:
            face_found=False
        else:
            face_found=True
        if face_found == False:
        #     counter1+=1
            cv2.putText(img, "INATTENTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with open(filename, 'a') as csvfile:
                rows = [['INATTENTIVE', datetime.datetime.now()]]
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(rows)
        # Returns an array of bounding boxes of human faces in a image
        # A list of tuples of found face locations in css(top, right, bottom, left) order
        for face in rects:
            #Extracting the co cordinates to convert them into dlib rectangle object
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            u = face.right()
            v = face.bottom()

            # newrect = dlib.rectangle(x,y,u,v)
            # cv2.rectangle(img, (x, y), (x+w, y+h),
            # (0, 255, 0), 2)
            # shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(img, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            focalLength = args["focal"] * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(img, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            if angles[1] < -15:
                GAZE = "Looking: Left"
            elif angles[1] > 30:
                GAZE = "Looking: Right"
            else:
                GAZE = "Forward"
            #counting the consecutive frames for which person is not looking forward
            if GAZE=="Looking: Right" or GAZE=="Looking: Left":
                cv2.putText(img, "INATTENTIVE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                with open(filename, 'a') as csvfile:
                    rows=[['INATTENTIVE' , datetime.datetime.now()]]
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(rows)
            else:
                with open(filename, 'a') as csvfile:
                    rows=[['ATTENTIVE' ,datetime.datetime.now()]]
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(rows)

        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv.imshow('img',img)
        keypress = cv.waitKey(1)
        if keypress == ord('q'):

            fig = plt.figure()
            plt.plot(x_axis, y_axis,color='green', linewidth = 5,marker='o', markerfacecolor='white', markersize=4,label='Attentiveness') 
            plt.tight_layout()
            plt.plot(ix_axis, iy_axis,color='red', linewidth = 5,marker='o', markerfacecolor='white', markersize=4,label='In-Attentiveness') 
            plt.tight_layout()
            plt.xlabel('TIME(in sec) ------> ',color='blue')
            plt.title(' Attentiveness vs In-Attentiveness',color='blue')
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper right")
            plt.tight_layout()
            plt.show()
        
        
        
            fig = plt.figure()
            #plt.title("Proportionate %age\n" + "of Mood Patterns", bbox={'facecolor':'0.8', 'pad':5})
            ax = fig.add_axes([0,0,1,1])
            ax.axis('equal')
            langs = ['angry','fear','happy','sad','surprised','neutral']
            students = [x / pie_count for x in pie_arr]
            students = [x *100 for x in students]
            ax.pie(students, labels = langs,autopct='%1.2f%%')

            plt.show()

           # fig = plt.figure()
            plt.figure(figsize=(9,4))
            objects = ('angry','fear','happy','sad','surprised','neutral')
            y_pos = np.arange(len(objects))
            performance = pie_arr
            plt.bar(y_pos, performance, align='center', alpha=1,color=['yellow', 'red', 'green', 'blue', 'black','cyan'])
            plt.xticks(y_pos, objects)
            plt.ylabel('Mood Levels',color='red')
            plt.title('Overall Mood Patterns',color='red')
            plt.tight_layout()
            plt.show()

            break


# Close the window 
df = pd.read_csv("head_movement.csv")
INATTENTIVE_HEAD_COUNT = len(df[df['concentration'] == 'INATTENTIVE'])
ATTENTIVE_HEAD_COUNT = len(df[df['concentration'] == 'ATTENTIVE'])
TOTAL_HEAD_COUNT = len(df['concentration'])
label = ['ATTENTIVE', 'INATTENTIVE']
ATTENTIVE_PERCENTAGE = (ATTENTIVE_HEAD_COUNT / TOTAL_HEAD_COUNT) * 100
INATTENTIVE_PERCENTAGE = (INATTENTIVE_HEAD_COUNT / TOTAL_HEAD_COUNT) * 100
data = [ATTENTIVE_PERCENTAGE, INATTENTIVE_PERCENTAGE]
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=label, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
plt.title("Based on Head Movement", bbox={'facecolor': '0.8', 'pad': 5})
plt.savefig('head_movement.png')
df1 = pd.read_csv("eye_blink.csv")
INATTENTIVE_blink_COUNT = len(df1[df1['concentration'] == 'INATTENTIVE'])
ATTENTIVE_blink_COUNT = len(df1[df1['concentration'] == 'ATTENTIVE'])
TOTAL_blink_COUNT = len(df1['concentration'])
label1 = ['ATTENTIVE', 'INATTENTIVE']
ATTENTIVE_blink_PERCENTAGE = (ATTENTIVE_blink_COUNT / TOTAL_blink_COUNT) * 100
INATTENTIVE_blink_PERCENTAGE = (INATTENTIVE_blink_COUNT / TOTAL_blink_COUNT) * 100
data = [ATTENTIVE_blink_PERCENTAGE, INATTENTIVE_blink_PERCENTAGE]
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=label1, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
plt.title("Based on eye blinking", bbox={'facecolor': '0.8', 'pad': 5})
plt.savefig('eye_blink.png')
Final_attentive_percent=(ATTENTIVE_blink_PERCENTAGE+ATTENTIVE_PERCENTAGE)/2
Final_inattentive_percent=(INATTENTIVE_blink_PERCENTAGE+INATTENTIVE_PERCENTAGE)/2
data = [Final_attentive_percent, Final_inattentive_percent]
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=label1, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
plt.title("Combined graph of eye and head", bbox={'facecolor': '0.8', 'pad': 5})
plt.savefig('combined_graph_eye_head.png')
df=pd.read_csv("example.txt",header=None)
df.columns=['index','attentiveness']
avg_att=df['attentiveness'].mean()
# print(avg)
label1 = ['ATTENTIVE', 'INATTENTIVE']
avg_inatt=100-avg_att
Final_attentive_percent=60
Final_inattentive_percent=40
total_att=(Final_attentive_percent+avg_att)/2
total_inatt=(Final_inattentive_percent+avg_inatt)/2
data1=[total_att,total_inatt]
fig = plt.figure(figsize=(10, 7))
plt.pie(data1, labels=label1, explode=(0.07, 0), colors=('green', 'red'), shadow=True, autopct='%1.1f%%')
plt.title("Combined graph", bbox={'facecolor': '0.8', 'pad': 5})
# plt.show()
plt.savefig('combined_graph.png')
cam.release()
cv.destroyAllWindows()


plt.figure(figsize=(10,5))
plt.plot(x_axis, y_axis,color='orange', linewidth = 4,marker='o', markerfacecolor='blue', markersize=4)
plt.xlabel('TIME(in sec) ------> ',color='blue') 
plt.ylabel('ATTENTIVENESS',color='blue') 
plt.title('Conclusion(Analysis of Attentiveness)',color='blue')
plt.tight_layout()
plt.show()

# if __name__ == "__main__":
#     # path to your video file or camera serial
#     main()