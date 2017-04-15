import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui

font = cv2.FONT_HERSHEY_SIMPLEX
startTime = lastTime = currentTime = time.time()
avgFps = 0

while True:
    # Capture the screen
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(8, 30, 808, 630))), cv2.COLOR_RGB2BGR)
    # Calculate the frame rate
    currentTime = time.time()
    avgFps = 0.95 * avgFps + 0.05 / (currentTime - lastTime)
    lastTime = currentTime
    # Process the image
    screenGray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    # Display the frame rate on the image
    cv2.putText(screenGray, "%4.1f" % avgFps, (730, 595), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("window", screenGray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()        
