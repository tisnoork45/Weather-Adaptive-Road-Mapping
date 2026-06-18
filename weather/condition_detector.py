import cv2
import numpy as np


def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 50


def detect_weather_condition(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    if is_blurry(image):
        return "blur"

    if brightness < 50:
        return "night"

    if brightness < 90:
        return "dark"

    if brightness > 160:
        return "clear"

    return "normal"