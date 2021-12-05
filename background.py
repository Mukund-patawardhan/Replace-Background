import cv2
import time
import numpy as np
import keyboard

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

time.sleep(2)
image = 0

for i in range(60):
    ret, image = cap.read()
image = np.flip(image, axis=1)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.flip(frame, axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([104, 153,70])
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)

    mask_2 = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = mask_1 + mask_2

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    res_1 = cv2.bitwise_and(frame, frame, mask=mask_2)

    res_2 = cv2.bitwise_and(image, image, mask=mask_1)

    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)

    cv2.imshow("magic", final_output)
    cv2.waitKey(1)
    k = cv2.waitKey(1) & 0xFF
    
    if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
        break

cap.release()
cv2.destroyAllWindows()