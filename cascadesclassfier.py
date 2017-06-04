import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('casscades/face.xml')
eye_cascade = cv2.CascadeClassifier('casscades/eye.xml')
smile_cascade = cv2.CascadeClassifier('casscades/smile.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smile = smile_cascade.detectMultiScale(roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 255, 0), 1)
    
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
