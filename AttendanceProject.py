import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

## Encode Known List of Images
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path) # get a list of file names in the path
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # splits the file name to ignore the extensions
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

## Capture and Encode Live Camera Frames
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25, 0.25) # scale down image size by 4
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #lowest distance would be the best match
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        ## labelling unknown faces
        if faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = "Unknown"
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 # rescale resulting bounding box
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        ## labelling known faces in set
        # if matches[matchIndex]:
        #     name = classNames[matchIndex].upper()
        #     y1,x2,y2,x1 = faceLoc
        #     y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 # rescale resulting bounding box
        #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #     cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        #     cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #     markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)