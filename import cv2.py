import cv2
import numpy as np
lower_red = np.array([0,0,100])
upper_red = np.array([45,5,255])
cap = cv2.VideoCapture(0)
while True:
    ret , frame = cap.read()
    
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    mask=cv2.inRange(hsv, lower_red,upper_red)
    
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    
    
    contours, _ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    
    if len(contours)>0:
        largest_contour=max(contours, key=cv2.contourArea)
        
        x , y,w,h= cv2.boundingRect(largest_contour)
        
        cv2.rectangle(frame,(x,y),(x+y,y+h),(0,255,0),2)
        
    cv2.imshow('LED Detection',frame)
    
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        
