import numpy as np
from PIL import ImageGrab
import cv2
import time
import matplotlib.pyplot as plt


def auto_canny(image, sigma=0.33, center=None):
    # compute the median of the single channel pixel intensities
    if center is None:
        v = np.median(image[np.nonzero(image)])
    else:
        v = center

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


def mask_roi_noise(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    image_masked = cv2.bitwise_and(image, mask)
    inv_mask = 255 - mask
    noise = inv_mask.astype(np.double) * np.random.rand(np.size(image, 0), np.size(image, 1))
    return image_masked + noise.astype(np.uint8)


font = cv2.FONT_HERSHEY_SIMPLEX
startTime = lastTime = currentTime = time.time()
avgFps = 0
roi_vertices = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [790, 300], [790, 500]])
avg_threshold = 127
avg_lines = np.zeros((600, 800), dtype=np.uint8)
avg_line_angle = 0


while True:
    # Capture the screen
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(8, 30, 808, 630))), cv2.COLOR_RGB2BGR)

    # Process the image
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_filter = cv2.GaussianBlur(screen_gray, (7, 7), 0)
    # screen_roi_noise = mask_roi_noise(screen_filter, roi_vertices)
    otsu_threshold = cv2.threshold(screen_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    avg_threshold = 0.6 * avg_threshold + 0.3 * otsu_threshold
    # screenThreshold = cv2.threshold(screen_filter, avg_threshold, 255, cv2.THRESH_BINARY)[1]
    screen_canny = auto_canny(255 - screen_filter, 0.5, avg_threshold)
    screenMask = mask_roi(screen_canny, roi_vertices)

    # Find lines on the image
    lines = cv2.HoughLinesP(screenMask, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=20)
    current_lines = np.zeros_like(screen_gray)
    current_line_angle = 0

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(current_lines, (x1, y1), (x2, y2), 255, 2)
            cv2.line(screen, (x1, y1), (x2, y2), (255, 0, 0), 2)
            current_line_angle += np.arctan(np.float64(x2 - x1)/np.float64(y1 - y2))
        current_line_angle /= np.size(lines, 0)

    avg_lines = 0.5 * avg_lines + 0.5 * current_lines
    avg_line_angle = 0.7 * avg_line_angle + 0.3 * current_line_angle
    cv2.line(screen, (400, 600), (int(600*np.tan(avg_line_angle) + 400), 0), (0, 255, 0), 2)

    # Display the processed data
    cv2.imshow("Edge Detected", screen_canny)
    cv2.imshow("Edge Detected + Mask", screenMask)
    cv2.imshow("Hough Lines", avg_lines)
    cv2.imshow("Lanes", screen)

    # Calculate the frame rate
    currentTime = time.time()
    avgFps = 0.95 * avgFps + 0.05 / (currentTime - lastTime)
    lastTime = currentTime

    # Respond to input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Average FPS: %0.1f" % avgFps)
        break

cv2.destroyAllWindows()
