import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)
    res = cv2.bitwise_and(frame,frame, mask= fgmask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()