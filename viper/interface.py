import cv2
import numpy as np

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