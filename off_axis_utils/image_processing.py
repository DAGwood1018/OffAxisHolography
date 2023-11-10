import cv2
import numpy as np


def format_img(arr):
    diff = arr.max() - arr.min()
    img = cv2.convertScaleAbs(arr, alpha=255.0 / diff, beta=-arr.min() * 255.0 / diff)
    return img.astype('uint8', copy=False)


def binarize(img):
    assert img.ndim == 2, "Image must be grayscale."
    img_rescaled = (img - img.min()) / (img.max() - img.min()) * 255
    img_rescaled = img_rescaled.astype(np.uint8)
    _, bi_img = cv2.threshold(img_rescaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bi_img
