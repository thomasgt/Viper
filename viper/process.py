import cv2
import numpy as np

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