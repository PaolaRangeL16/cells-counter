import numpy as np
import cv2

def to_gray(img):
    # Si ya está en gris, regresar igual
    if len(img.shape) == 2:
        return img.copy()
    
    filas, columnas = img.shape[:2]
    gray = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(filas):
        for j in range(columnas):
            b, g, r = img[i, j]   # ⚠️ OpenCV usa BGR
            gray[i, j] = int(0.299*r + 0.587*g + 0.114*b)

    return gray

def gaussian_filter(img, kernel_size=5, sigma=1.0):
    
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
    
    # 1. gris
    gray = to_gray(img)

    # 2. filtro
    filtered = gaussian_filter(gray)

    # 3. contraste
    enhanced = gamma_correction(filtered)

    return enhanced
  
