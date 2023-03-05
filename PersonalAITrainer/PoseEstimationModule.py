import cv2
import mediapipe as mp
import time
import math
class poseDetector():
        global mp_drawing_utils ,mp_pose,pose
        mp_drawing_utils = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5,
                            model_complexity=1, static_image_mode= False,
                            smooth_landmarks=True, enable_segmentation=False
                            )


        def findPose(self, image, draw=True):
            global results
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                if draw:
                    mp_drawing_utils.draw_landmarks(image, results.pose_landmarks,
                                                    mp_pose.POSE_CONNECTIONS,
                                                    mp_drawing_utils.DrawingSpec(color=(255 ,0, 255),thickness=2, circle_radius=2),
                                                    mp_drawing_utils.DrawingSpec(color=(0 ,0, 255),thickness=2, circle_radius=2))
            return image
        def findPosition(self, image, draw=True):
            global results ,lmList
            lmList = []
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = image.shape
                    # print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return lmList

        def calculate_angle(self, image, p1,p2,p3 , draw = True):
            global lmList
            # GET LAND MARSK   --> (_,) this sign means we escaping the first element
            x1,y1 =  lmList[p1][1:]
            x2,y2 =  lmList[p2][1:]
            x3,y3 =  lmList[p3][1:]

            # calculate angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

            # Draw
            if draw:
                cv2.line(image, (x1,y1), (x2,y2),(255,255,255),3)
                cv2.line(image, (x2,y2), (x3,y3),(255,255,255),3)
                cv2.circle(image,(x1,y1), 10, (0,0,255),cv2.FILLED)
                cv2.circle(image,(x1,y1), 15, (0,0,255),2)
                cv2.circle(image,(x2,y2), 10, (0,0,255),cv2.FILLED)
                cv2.circle(image, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(image,(x3,y3), 10, (0,0,255),cv2.FILLED)
                cv2.circle(image, (x3, y3), 15, (0, 0, 255), 2)
            return angle


def main():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("C:\\Users\HP\Downloads\pexels-vlada-karpovich-8045821.mp4")
    pTime = 0
    detector = poseDetector()
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue
        image = detector.findPose(image)
        lmList = detector.findPosition(image, draw=True)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(image, (lmList[14][1], lmList[14][1]), 10, (0,255,0) ,cv2.FILLED)

            # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()


