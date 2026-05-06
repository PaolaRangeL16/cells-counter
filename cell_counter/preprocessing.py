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
  
    if len(img.shape) == 2:
        return img.copy()
    r = img[:, :, 2].astype(np.float32)  # BGR → canal R
    g = img[:, :, 1].astype(np.float32)
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


def frequency_filter(img, cutoff=30, mode='low'):
   
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)

    filas, columnas = img.shape
    crow, ccol = filas // 2, columnas // 2

    # Máscara circular centrada en las frecuencias bajas
    mask = np.zeros((filas, columnas), dtype=np.float32)
    for i in range(filas):
        for j in range(columnas):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff:
                mask[i, j] = 1

    if mode == 'high':
        mask = 1 - mask

    # Aplicar máscara y regresar al dominio espacial
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)

    return np.clip(img_filtered, 0, 255).astype(np.uint8)


def gamma_correction(img, gamma=1.2):
    
    img_norm = img.astype(np.float32) / 255.0
    img_gamma = np.power(img_norm, gamma)
    return (img_gamma * 255).astype(np.uint8)


def preprocess_image(img):
    # 1. Canal púrpura
    purple = extract_purple_channel(img)
    # 2. Filtrado en frecuencia (pasa bajos) este elimina ruido de alta frecuencia
    freq_filtered = frequency_filter(purple, cutoff=30, mode='low')
    # 3. Filtro gaussiano espacial ayuda para el suavizado fino
    filtered = gaussian_filter(freq_filtered, kernel_size=5, sigma=1.5)
    # 4. Corrección gamma para  realzar contraste
    enhanced = gamma_correction(filtered, gamma=0.8)
    return enhanced
