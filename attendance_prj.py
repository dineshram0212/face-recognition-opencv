import cv2
import os
import numpy as np
import face_recognition as fr

path = "imgs"

images = []    # Appending images after being read

className = []   # Name of the Images(filename without extension)

myList = os.listdir(path)   # Filename

for cls in myList:
    currentImg = cv2.imread(f'{path}/{cls}')
    images.append(currentImg)
    className.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeKnownFaces = findEncodings(images)
print(len(encodeKnownFaces))


#### Capturing through Webcam
cap = cv2.VideoCapture(0)

while True: # To get each frame one by one
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = fr.face_locations(imgS)
    encodeCurrFrame = fr.face_encodings(imgS, facesCurrentFrame)

    for encodeFace, faceLoc in zip(facesCurrentFrame, encodeKnownFaces):
        matches = fr.compare_faces(encodeKnownFaces, encodeFace)
        faceDist = fr.face_distance(encodeKnownFaces, encodeFace) 
        matchIndex = np.min(faceDist)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # Labelling Detected Image            
            cv2.rectangle(img, (x1,x2), (y1,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,x2-35), (y1,y2), (0,255,0), 2, cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)

        cv2.imshow('WEBCAM', img)
        cv2.waitKey(1)