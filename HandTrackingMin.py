import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands      #detect hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #function to draw the points

pTime = 0
cTime = 0

while True:       #code to run the webcam
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #convert the img into backgroundcolor rgb
    results = hands.process(imgRGB)  #SENDING THIS IMGRGB into hands

    #print(results.multi_hand_landmarks)  #check if something isdetected or not
    #check if we have multihands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #extract the information of each hand
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape   #high width channel
                cx, cy = int(lm.x*w), int(lm.y*h)   #positions
                print(id, cx, cy)
                #if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  #draw the img

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)