# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:53:08 2026

@author: paor9
"""

# edge_detection.py
import numpy as np
from scipy import signal


def sobel(img):
   
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    kernel_x = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[ 1,  2,  1],
                          [ 0,  0,  0],
                          [-1, -2, -1]], dtype=np.float32)

    dx = signal.convolve2d(img.astype(np.float32), kernel_x, mode='same')
    dy = signal.convolve2d(img.astype(np.float32), kernel_y, mode='same')
    magnitude = np.sqrt(dx**2 + dy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def prewitt(img):
    
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    kernel_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[ 1,  1,  1],
                          [ 0,  0,  0],
                          [-1, -1, -1]], dtype=np.float32)

    dx = signal.convolve2d(img.astype(np.float32), kernel_x, mode='same')
    dy = signal.convolve2d(img.astype(np.float32), kernel_y, mode='same')
    magnitude = np.sqrt(dx**2 + dy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def laplacian(img):
    
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    kernel = np.array([[0,  1, 0],
                        [1, -4, 1],
                        [0,  1, 0]], dtype=np.float32)

    edges = signal.convolve2d(img.astype(np.float32), kernel, mode='same')
    return np.clip(np.abs(edges), 0, 255).astype(np.uint8)


def hough_circles(img, min_radius=20, max_radius=90, min_dist=40):
    """
    Regresa la imagen con círculos marcados y el conteo.
    """
    import cv2
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=100,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    count = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        count = len(circles)
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

    return result, count
