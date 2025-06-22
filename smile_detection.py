import cv2
import numpy as np
#to detect faces in an image using OpenCV we need some data,so we took pre-trained data
trained_face_data =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_data= cv2.CascadeClassifier('haarcascade_smile.xml')
#choose an image to detect face in it
#img= cv2.imread('face.png')
webcam = cv2.VideoCapture(0)
#iterate forever over frame
while True:
   #read the current frame
   successful_frame_read,frame= webcam.read()

     #convert the image to grayscale
   gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       #detect faces 
   face_coordinates = trained_face_data.detectMultiScale(gray_img)
   
#print(face_coordinates)
#draw rectangles around the faces
   for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#run smile in each of those faces
#numpy used here for splitting
    the_face = frame[y:y+h, x:x+w]
    gray_img = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
    smile_coordinates = smile_data.detectMultiScale(gray_img,scaleFactor=1.7, minNeighbors=20)
    #for (x_, y_, w_, h_) in smile_coordinates:
        #cv2.rectangle(the_face, (x_, y_), (x_+ w_, y_ + h_), (50, 50, 200), 2)
        
    if len(smile_coordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y +h+40),fontScale=3,
            fontFace= cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

#cv2.rectangle(img, (face_coordinates[0][0], face_coordinates[0][1]),
#              (face_coordinates[0][0] + face_coordinates[0][2], face_coordinates[0][1] + face_coordinates[0][3]), 
 #               (0, 255, 0), 2)

   cv2.imshow('Original Image', frame)
   key=cv2.waitKey(1)
   if key==81 or key==113:  #q or Q
         break
#release the webcam
webcam.release()

print("code completed")
