import cv2
import numpy as np

def enhance_image(img, condition):

    if condition in ("dark", "night"):

        gamma = 2.0

        lut = np.array([
            ((i / 255.0) ** (1.0 / gamma)) * 255
            for i in range(256)
        ]).astype("uint8")

        img = cv2.LUT(img, lut)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        lab = cv2.merge((clahe.apply(l), a, b))

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    elif condition == "blur":

        gaussian = cv2.GaussianBlur(img, (0, 0), 3)

        img = cv2.addWeighted(
            img,
            1.8,
            gaussian,
            -0.8,
            0
        )

    ycrcb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    ycrcb = cv2.merge(
        (
            clahe.apply(y),
            cr,
            cb
        )
    )

    return cv2.cvtColor(
        ycrcb,
        cv2.COLOR_YCrCb2BGR
    )