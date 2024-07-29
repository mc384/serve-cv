import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math

counter = 0
pos_reached = 0
cap = cv2.VideoCapture('rec_serve.mp4') # create a VideoCapture object
pd = PoseDetector(trackCon=0.70,detectionCon=0.70) # tracking and detection confidence

def movements(lmlist,p1,p2,p3,p4,p5,p6,p7,p8,p9):
        # global variables
        global counter
        global pos_reached

        # check if there is a nonempty landmark list
        if len(lmlist)!= 0:
            point1 = lmlist[p1]
            point2 = lmlist[p2]
            point3 = lmlist[p3] # left hand
            point4 = lmlist[p4]
            point5 = lmlist[p5]
            point6 = lmlist[p6] # right hand
            point7 = lmlist[p7]
            point8 = lmlist[p8] # right hip
            point9 = lmlist[p9] # left hip

            x1,y1 = point1[:2] # second and second last element
            x2, y2 = point2[:2]
            x3, y3 = point3[:2]
            x4, y4 = point4[:2]
            x5, y5 = point5[:2]
            x6, y6 = point6[:2]
            x7, y7 = point7[:2]
            x8, y8 = point8[:2]
            x9, y9 = point9[:2]

            leftelbowangle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                         math.atan2(y1 - y2, x1 - x2))

            rightelbowangle = math.degrees(math.atan2(y6 - y5, x6 - x5) -
                                          math.atan2(y4 - y5, x4 - x5))
            
            leftshoulderangle = math.degrees(math.atan2(y5 - y2, x5 - x2) -
                                          math.atan2(y9 - y1, x9 - x1))
            
            rightshoulderangle = math.degrees(math.atan2(y5 - y4, x5 - x4) -
                                          math.atan2(y8 - y4, x8 - x4))
            
            
            leftElbowangle = int(np.interp(leftelbowangle, [-210, 200], [100, 0]))
            rightElbowangle = int(np.interp(rightelbowangle, [-280, 300], [100, 0]))
            leftShoulderangle = int(np.interp(leftshoulderangle, [-280, 110], [100, 0]))
            rightShoulderangle = int(np.interp(rightshoulderangle, [-280, 110], [100, 0]))

            left, right = leftElbowangle, rightElbowangle
            rshoulder = rightShoulderangle
            lshoulder = leftShoulderangle

            if left >= 70 and right >= 70:
                if pos_reached == 0:
                    counter += 0.5
                    pos_reached = 1
            if left <= 70 and right <= 70:
                if pos_reached == 1:
                    counter += 0.5 
                    pos_reached = 0
                    
            # Count
            cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
            cv2.putText(img, str(int(counter)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 7)

            # Bars
            leftval  = np.interp(left,[0,100],[250,50]) # dictates height of rectangle
            rightval = np.interp(right, [0, 100], [250, 50])
            sval = np.interp(rshoulder, [0, 100], [250, 50])
            sval2 = np.interp(lshoulder, [0, 100], [250, 50])

            # Bar for right hand angle
            cv2.putText(img,'R', (1110, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
            cv2.rectangle(img,(1100,50),(1140,250),(0,255,0),5)
            cv2.rectangle(img, (1100, int(rightval)), (1140, 250), (255,0, 0), -1)

            # Bar for left hand angle
            cv2.putText(img, 'L', (1030, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
            cv2.rectangle(img, (1020, 50), (1060, 250), (0, 255, 0), 5)
            cv2.rectangle(img, (1020, int(leftval)), (1060, 250), (255, 0, 0), -1)

            # Bar for right shoulder angle
            cv2.putText(img, 'RS', (942, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
            cv2.rectangle(img, (940, 50), (980, 250), (0, 255, 0), 5)
            cv2.rectangle(img, (940, int(sval)), (980, 250), (255, 0, 0), -1)

            # Bar for left shoulder angle
            cv2.putText(img, 'LS', (862, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
            cv2.rectangle(img, (860, 50), (900, 250), (0, 255, 0), 5)
            cv2.rectangle(img, (860, int(sval2)), (900, 250), (255, 0, 0), -1)

while True:
    ret, img = cap.read()
    if not ret: # in absence of a video frame, reload the video
        cap = cv2.VideoCapture('rec_serve.mp4') 
        continue

    img = cv2.resize(img, (1300, 800))

    pd.findPose(img, draw=1)
    lmlist, bbox = pd.findPosition(img, draw=0, bboxWithHands=0)

    movements(lmlist,11,13,15,12,14,16,1,24,23)

    cv2.imshow('frame', img)
    cv2.waitKey(1)