import cv2
import numpy as np

# TODO Use a class instead of globals

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


def get_shape_from_mouse(im, window_title):
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
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, mouse_callback)
    cv2.imshow(window_title, im_clone)

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
            cv2.imshow(window_title, im_clone)
        cv2.waitKey(1)
    cv2.destroyWindow(window_title)

    # Return a list of points
    return np.vstack((x_roi, y_roi)).T


def get_roi(im):
    return get_shape_from_mouse(im, "Select the ROI")


def get_perspective_transform(im):
    start_points = get_shape_from_mouse(im, "Select the start points")
    final_points = get_shape_from_mouse(im, "Select the final points")
    start_points = np.float32(start_points[:4])
    final_points = np.float32(final_points[:4])
    transform_matrix = cv2.getPerspectiveTransform(start_points, final_points)
    final_size = (2*np.size(im, 1), 2*np.size(im, 0))
    return transform_matrix, final_size


if __name__ == '__main__':
    from PIL import ImageGrab
    screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(8, 30, 808, 630))), cv2.COLOR_RGB2BGR)
    #rv = get_roi(screen)
    tm, fs = get_perspective_transform(screen)
    #print(rv)
    print(tm)
    print(fs)
    cv2.imshow("Transformed", cv2.warpPerspective(screen, tm, fs))
    cv2.imshow("Original", screen)
    cv2.waitKey()

