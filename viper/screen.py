import numpy as np
from PIL import ImageGrab
import cv2
import time
from viper import direct_keys as dk

# TODO Modularize the code
# TODO Apply some perspective warping to make lane detection easier
# TODO Come up with a kernel that can find lanes consistently


# Shared variables used in mouse callback
points_clicked = 0
x_roi = []
y_roi = []
waiting_for_mouse = False


def mouse_callback(event, x, y, flags, param):
    # Shared variables
    global x_roi, y_roi, points_clicked, waiting_for_mouse

    # Stop once we've decided not to wait any longer
    if not waiting_for_mouse:
        return

    # Whenever we receive a left click, process it
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add this (x, y) position to the list of points
        x_roi.append(x)
        y_roi.append(y)
        points_clicked = points_clicked + 1

        # If we just clicked near the first point, stop
        if points_clicked > 1:
            dist = np.sqrt((x - x_roi[0]) ** 2 + (y - y_roi[0]) ** 2)
            if dist < 100:
                waiting_for_mouse = False


def get_roi_from_mouse(im):
    # Get a copy of the image so that we can draw on it
    im_clone = np.copy(im)

    # Set up our globals
    global x_roi, y_roi, points_clicked, waiting_for_mouse
    x_roi = []
    y_roi = []
    points_clicked = 0
    points_drawn = 0
    waiting_for_mouse = True

    # Show the image to the user
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI", mouse_callback)
    cv2.imshow("ROI", im_clone)

    # Wait until we're done picking the ROI
    while waiting_for_mouse:
        # Draw any new ROI edges on the image
        while points_drawn < points_clicked:
            pt = (x_roi[points_drawn], y_roi[points_drawn])
            cv2.circle(im_clone, pt, 5, (0, 0, 255), -1)
            if points_drawn > 0:
                pt2 = (x_roi[points_drawn - 1], y_roi[points_drawn - 1])
                cv2.line(im_clone, pt, pt2, (0, 0, 255), 5)
            points_drawn = points_drawn + 1
            cv2.imshow("ROI", im_clone)
        cv2.waitKey(1)
    cv2.destroyWindow("ROI")

    # Return a list of points
    return np.vstack((x_roi, y_roi)).T


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


def mask_roi(image, vertices, mask=None):
    if mask is not None:
        return cv2.bitwise_and(image, mask)
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
avg_threshold = 127
avg_lines = np.zeros((600, 800), dtype=np.uint8)
avg_line_angle = 0

for t in range(5, 0, -1):
    print("Starting in {}...".format(t))
    time.sleep(1)

# Capture the screen
roi_file = 'truck3.npy'
try:
    roi_vertices = np.load(roi_file)
except IOError:
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(8, 30, 808, 630))), cv2.COLOR_RGB2BGR)
    roi_vertices = get_roi_from_mouse(screen)
    np.save(roi_file, roi_vertices)

print(roi_vertices)
forward_control_avg = 0
right_control_avg = 0
left_control_avg = 0
forward_thread = dk.KeyThread(dk.Key.W, 2)
left_thread = dk.KeyThread(dk.Key.A, 0.5)
right_thread = dk.KeyThread(dk.Key.D, 0.5)
forward_thread.start()
left_thread.start()
right_thread.start()

while True:
    # Capture the screen
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(8, 30, 808, 630))), cv2.COLOR_RGB2BGR)

    # Process the image
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_filter = cv2.GaussianBlur(screen_gray, (7, 7), 0)
    # screen_roi_noise = mask_roi_noise(screen_filter, roi_vertices)
    otsu_threshold = cv2.threshold(screen_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    avg_threshold = 0.6 * avg_threshold + 0.4 * otsu_threshold
    screen_canny = auto_canny(255 - screen_filter, 0.5, avg_threshold)
    screenMask = mask_roi(screen_canny, roi_vertices)

    # Find lines on the image
    lines = cv2.HoughLinesP(screenMask, 1, np.pi / 180, 75, minLineLength=75, maxLineGap=30)
    current_lines = np.zeros_like(screen_gray)
    current_line_angle = 0

    if lines is not None:
        total_length = 0
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if np.abs(y1 - y2) > 0:
                angle = np.arctan((x2 - x1)/(y1 - y2))
                if np.rad2deg(np.abs(angle)) < 70:
                    cv2.line(current_lines, (x1, y1), (x2, y2), 255, 2)
                    cv2.line(screen, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    current_line_angle += np.arctan((x2 - x1)/(y1 - y2))
                    total_length += np.hypot(x2 - x1, y2 - y1) / 100
        if total_length > 0:
            current_line_angle /= total_length
        else:
            current_line_angle = 0

    avg_lines = 0.5 * avg_lines + 0.5 * current_lines
    avg_line_angle = 0.8 * avg_line_angle + 0.2 * current_line_angle
    cv2.line(screen, (400, 600), (int(600*np.tan(avg_line_angle) + 400), 0), (0, 255, 0), 2)
    cv2.putText(screen, "%6.1f" % np.rad2deg(avg_line_angle), (690, 595), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Update controls
    forward_control = 0.5 * np.cos(avg_line_angle) + 0.6
    forward_control_avg = 0.6 * forward_control_avg + 0.4 * forward_control
    if avg_line_angle > 0:
        right_control = 0.5 / (1 + np.exp(-3 * avg_line_angle + 3))
        right_control_avg = 0.6 * right_control_avg + 0.4 * right_control
        left_control_avg = 0
    else:
        left_control = 0.5 / (1 + np.exp(3 * avg_line_angle + 3))
        left_control_avg = 0.6 * left_control_avg + 0.4 * left_control
        right_control_avg = 0
    cv2.putText(screen, "%0.2f" % forward_control_avg, (720, 395), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(screen, "%0.2f" % right_control_avg, (720, 445), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(screen, "%0.2f" % left_control_avg, (720, 495), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    forward_thread.set_duty_cycle(forward_control_avg)
    right_thread.set_duty_cycle(right_control_avg)
    left_thread.set_duty_cycle(left_control_avg)

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
    key_command = cv2.waitKey(1) & 0xFF
    if key_command & 0xFF == ord('q'):
        print("Average FPS: %0.1f" % avgFps)
        break
    elif key_command == ord('s'):
        forward_thread.disable()
        right_thread.disable()
        left_thread.disable()
    elif key_command == ord('g'):
        forward_thread.enable()
        right_thread.enable()
        left_thread.enable()

cv2.destroyAllWindows()
