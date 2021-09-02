import cv2
import numpy as np
import face_recognition

########### STEP 1 - Importing image and converting it to RGB ###########
imgElon = face_recognition.load_image_file('elontrain.jpg')
# Converting BGR image to RGB image
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgElonTest = face_recognition.load_image_file('elontest1.jpg') # Recognize/Identify the face
# Converting BGR image to RGB image
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)


########### STEP 2 - Finding Faces in Image ###########
faceLoc = face_recognition.face_locations(imgElon)[0]   # Coordinates of the faces
encodeElon = face_recognition.face_encodings(imgElon)[0]    # Face encoding  
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeElonTest)  # Comparing Train and Test encodings
faceDis = face_recognition.face_distance([encodeElon], encodeElonTest)

print(results)
print(faceDis)
cv2.putText(imgElonTest, f'{results} {round(faceDis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255),2)

#############################################
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Musk Test', imgElonTest)

cv2.waitKey(0)