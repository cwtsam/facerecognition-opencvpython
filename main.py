import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/elon_musk.jpeg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB) # convert to RGB
imgTest = face_recognition.load_image_file('ImagesBasic/bill_gates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) # convert to RGB

## Finding faces in image and encodings
faceLoc = face_recognition.face_locations(imgElon)[0] #only sending single image, getting just the first element
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # drawing bounding box

faceLocTest = face_recognition.face_locations(imgTest)[0] #only sending single image, getting just the first element
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2) # drawing bounding box

## now that we have encodings for each image, we will compare them

results = face_recognition.compare_faces([encodeElon],encodeTest) #gives true/false if encodings match
faceDis = face_recognition.face_distance([encodeElon],encodeTest) #gives value of how close the match is
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)
