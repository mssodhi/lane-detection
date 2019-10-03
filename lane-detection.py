import numpy as np
import cv2

def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, frame, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[0,255,0]
    leftColor=[255,0,0]
    
    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2 - y1) > 20:
                slope = (y1-y2)/(x1-x2)
                # print(slope, x1)
                if slope > 0.3:
                    if x1 > 250 :
                        yintercept = y2 - (slope*x2)
                        rightSlope.append(slope)
                        rightIntercept.append(yintercept)
                    else: None
                elif slope < -0.3:
                    if x1 < 250:
                        yintercept = y2 - (slope*x2)
                        leftSlope.append(slope)
                        leftIntercept.append(yintercept)
    #We use slicing operators and np.mean() to find the averages of the 15 previous frames
    #This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-5:])
    leftavgIntercept = np.mean(leftIntercept[-5:])
    rightavgSlope = np.mean(rightSlope[-5:])
    rightavgIntercept = np.mean(rightIntercept[-5:])
    # print(rightavgSlope, rightavgIntercept)
    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.2*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
        right_line_x1 = int((0.2*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)
        pts = np.array([[left_line_x1, int(0.2*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.2*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,0,255))
        cv2.line(img, (left_line_x1, int(0.2*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(0.2*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
    #I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass

def hough_lines(img, frame, rho, theta, threshold, minLineLength, maxLineGap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, frame, lines)
    return line_img

cap = cv2.VideoCapture('trimmed.mp4')
#cap = cv2.VideoCapture('lanes2.mp4')
#cap = cv2.VideoCapture(0)

rho = 6
theta = np.pi / 60
threshold = 120
minLineLength = 40
maxLineGap = 25

while(True):

    ret, frame = cap.read()
    frame = frame[20:200]
    frame_color_filter = color_filter(frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges_blur = cv2.Canny(cv2.GaussianBlur(frame_color_filter,(7,7),1), 50, 120)
    edges_blur_3 = cv2.cvtColor(edges_blur, cv2.COLOR_GRAY2RGB)

    lines = cv2.HoughLinesP(edges_blur, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

    for item in lines:
        for x1,y1,x2,y2 in item:
            if abs(y2 - y1) > 20:
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),3)

    newFrame = hough_lines(edges_blur, frame, rho, theta, threshold, minLineLength, maxLineGap)
    vstack1 = np.vstack((frame, newFrame))
    vstack2 = np.vstack((frame_color_filter, edges_blur_3))
    numpy_horizontal = np.hstack((vstack1, vstack2))
    cv2.imshow("Monitor", numpy_horizontal)

    # cv2.imshow('edges', edges)
    # cv2.imshow('hough', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
