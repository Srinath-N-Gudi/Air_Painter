import cv2
import mediapipe as mp
import time
import os
import numpy as np
xp, yp = 0, 0

def chech_fingers(i, m):
    if m > i:
        
        return 0
    else:
        return 1
    
current_color = (0, 0, 255)

canvas = np.zeros((480, 640, 3), np.uint8)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
index_pos = ()
middle_pos = ()
folder_path = "Images"
images_list = os.listdir(folder_path)
images = []
for i in images_list:
    images.append(cv2.imread(folder_path+"\\"+i))
images_new = [] 
for i in images:
    images_new.append(cv2.cv2.flip(i, 1))
images = images_new
header_image = images[-1]
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (640, 480))
    ######################### Setting header image####################
    image[0:120, 0:640] = header_image
    ###################################################################
    if not success:
      print("Ignoring empty camera frame.")
      continue

    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      for id, lm in enumerate(hand_landmarks.landmark):
          h,w,c = image.shape
          cx , cy = int(lm.x *w), int(lm.y*h)
          if id == 8:
            index_pos = (cx, cy)
          if id == 12:
            middle_pos = (cx, cy)
    if index_pos != () and middle_pos != ():
        if chech_fingers(index_pos[1], middle_pos[1]) == 0:
            cv2.circle(image, index_pos, 25, current_color, 1)
            if xp == 0 and yp == 0:
                xp = index_pos[0] 
                yp = index_pos[1]
            if current_color == (0, 0, 0):
                cv2.line(canvas, (xp, yp), (index_pos[0] ,index_pos[1]), current_color, 50)
            else:
                cv2.line(canvas, (xp, yp), (index_pos[0] ,index_pos[1]), current_color, 20)
            xp = index_pos[0]
            yp = index_pos[1]
        elif chech_fingers(index_pos[1], middle_pos[1]) == 1:
            xp, yp = 0,0
            cv2.rectangle(image, (index_pos[0], index_pos[1]-15), (middle_pos[0], middle_pos[1]+15), current_color, cv2.FILLED)
            if index_pos[1] < 150:
                if 125<index_pos[0]<280:
                    current_color = (0, 0, 255)
                    header_image = images[-1]
                elif 285<index_pos[0]<385:
                    current_color = (0, 255, 0)
                    header_image = images[-2]
                elif 390<index_pos[0]<515:
                    header_image = images[0]
                    current_color = (255, 0, 0)
                elif 520<index_pos[0]<640:
                    header_image = images[1]
                    current_color = (0,0,0)
        index_pos = ()
        middle_pos = ()
    else:
        xp, yp=0,0

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, canvas)



    cv2.imshow('Virtual Paint', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()
