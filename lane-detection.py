import numpy as np
import cv2

cap = cv2.VideoCapture('CVFootage.mp4')
#cap = cv2.VideoCapture('lanes2.mp4')
#cap = cv2.VideoCapture(0)

rho = 1
theta = np.pi / 180
threshold = 50
minLineLength = 20
maxLineGap = 60

while(True):

    ret, frame = cap.read()
    frame = frame[25:200, 0:1280]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),1)
    edges = cv2.Canny(blur, 200, 100,apertureSize = 3)

    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    cv2.line(frame, (width / 2, height / 6), (width / 2, height), (0, 255, 255), 3)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

    for x1,y1,x2,y2 in lines[0]:
        if abs(y2 - y1) > 20:
            cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)

    cv2.imshow('edges', edges)
    cv2.imshow('hough', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
