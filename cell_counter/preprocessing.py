# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:20:39 2026

@author: paor9
"""
# preprocessing.py
import numpy as np

def to_gray(img):
    if len(img.shape) == 2:
        return img.copy()
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    gray = 0.299*r + 0.587*g + 0.114*b
    return np.clip(gray, 0, 255).astype(np.uint8)


def extract_purple_channel(img):
    """
    Para imágenes microscópicas con tinción púrpura (H&E o Giemsa),
    extrae la información de color que mejor separa células del fondo.
    Usa el canal R invertido ya que las células púrpuras tienen R bajo.
    """
    if len(img.shape) == 2:
        return img.copy()
    r = img[:, :, 2].astype(np.float32)  # BGR → canal R
    g = img[:, :, 1].astype(np.float32)
    # Las células púrpuras tienen R bajo y G bajo vs fondo blanco
    # Invertimos para que células sean brillantes
    purple = 255 - ((r * 0.5 + g * 0.5)).astype(np.uint8)
    return purple


def gaussian_filter(img, kernel_size=5, sigma=1.5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    centro = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - centro
            y = j - centro
            kernel[i, j] = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel /= np.sum(kernel)
    filas, columnas = img.shape
    filtered = np.zeros((filas, columnas), dtype=np.uint8)
    padding = kernel_size // 2
    img_padded = np.pad(img, padding, mode='edge')
    for i in range(filas):
        for j in range(columnas):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            valor = np.sum(region * kernel)
            filtered[i, j] = np.clip(valor, 0, 255)
    return filtered


def gamma_correction(img, gamma=1.2):
    img_norm = img.astype(np.float32) / 255.0
    img_gamma = np.power(img_norm, gamma)
    return (img_gamma * 255).astype(np.uint8)


def preprocess_image(img):
    # 1. Extraer canal púrpura en lugar de gris estándar
    purple = extract_purple_channel(img)
    # 2. Filtro gaussiano más suave (kernel 5 en lugar de 7)
    filtered = gaussian_filter(purple, kernel_size=5, sigma=1.5)
    # 3. Gamma para realzar contraste
    enhanced = gamma_correction(filtered, gamma=0.8)
    return enhanced
