import time

import PoseEstimation.PoseEstimationModule as pem
import cv2
import numpy as np

################################################################
webcam_height, webcam_width = 720, 480
pose_detection = pem.poseDetector()
pTime = 0
count = 0
dir = 0
################################################################
cap = cv2.VideoCapture(0)
cap.set(3, webcam_height)
cap.set(4, webcam_width)
while cap.isOpened():
    sucess, image = cap.read()
    image = cv2.resize(image, (720, 480))
    if not sucess:
        print("Ignoring Empty frame")
        # If loading a video, then we used 'break' instead of 'continue'.
        continue

    image = pose_detection.findPose(image)
    lm_marks = pose_detection.findPosition(image, False)

    if len(lm_marks) != 0:
        print(lm_marks)
        # Right arm
        angle = pose_detection.calculate_angle(image, 12, 14, 16)
        bar = np.interp(image, (210, 310), (650, 100))
        per = np.interp(image, (220, 310), (0, 100))
        color = (255, 0, 255)
        if per == 100:
            color = (255, 0, 255)
            if dir == 0:
                color = (0, 0, 255)
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 0, 255)
            if dir == 1:
                color = (255, 0, 255)
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        cv2.rectangle(image, (1100, 100), (1175, 650), (255, 0, 0), 2)
        cv2.rectangle(image, (1100, int(bar)), (1175, 650), (255,0,0), cv2.FILLED)
        cv2.putText(image, f'{int(per)}%', (1100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,255), 4)

        # Draw curl count
        cv2.rectangle(image, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, str(int(count)), (45, 670),
                    cv2.FONT_HERSHEY_SIMPLEX, 15, (255, 0, 0), 25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, "FPS: " + str(int(fps)), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("curl countint: ", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
