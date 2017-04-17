import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image[np.nonzero(image)])

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def mask_roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(image, mask)


font = cv2.FONT_HERSHEY_SIMPLEX
startTime = lastTime = currentTime = time.time()
avgFps = 0
roi_vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [790, 300], [790, 500]])

while True:
    # Capture the screen
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(8, 30, 808, 630))), cv2.COLOR_RGB2BGR)

    # Process the image
    screenGray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screenFilter = cv2.GaussianBlur(screenGray, (5, 5), 0)
    screenEdge = auto_canny(screenFilter, 0.66)
    screenMask = mask_roi(screenEdge, roi_vertices)

    # Find lines on the image
    lines = cv2.HoughLinesP(screenMask, 2, np.pi/90, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)


    # Display the processed data
    cv2.imshow("Gray", screenGray)
    cv2.imshow("Filter", screenFilter)
    cv2.imshow("Edge Detected", screenEdge)
    cv2.imshow("Edge Detected + Mask", screenMask)
    cv2.imshow("Hough Lines", screen)

    # Calculate the frame rate
    currentTime = time.time()
    avgFps = 0.95 * avgFps + 0.05 / (currentTime - lastTime)
    lastTime = currentTime

    # Respond to input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Average FPS: %0.1f" % avgFps)
        break

cv2.destroyAllWindows()
