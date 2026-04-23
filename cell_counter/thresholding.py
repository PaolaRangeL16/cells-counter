import numpy as np
import cv2

def threshold(img, T):
    filas, columnas = img.shape
    imgBN = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(filas):
        for j in range(columnas):
            if img[i, j] > T:
                imgBN[i, j] = 255
            else:
                imgBN[i, j] = 0

    return imgBN


def otsu_threshold(img):
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    T, imgOtsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgOtsu


def harris_corners(img):
    img = np.float32(img)
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    return dst
