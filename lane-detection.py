import numpy as np
import cv2
import RPi.GPIO as GPIO
import time

cap = cv2.VideoCapture('CVFootage.mp4')
#cap = cv2.VideoCapture('lanes2.mp4')
#cap = cv2.VideoCapture(0)

while(True):
    rho = 1
    theta = np.pi/180
    threshold = 50
    minLineLength = 20
    maxLineGap = 60
    
    ret, frame = cap.read()
    frame = frame[25:200, 0:1280]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),1)
#    blur = cv2.bilateralFilter(gray,5,75,75)
    edges = cv2.Canny(blur, 200, 100,apertureSize = 3)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
    #        print(x1, y1, x2, y2)
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
#    if lines is not None:
#        a,b,c = lines.shape
#        for i in range(a):
#            cv2.line(blur, (lines[i][0][0], lines[i][0][1]), (0,0,225), 2, cv2.CV_AA)

    cv2.imshow('edges', edges)
    cv2.imshow('hough', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
