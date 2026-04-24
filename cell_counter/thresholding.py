import numpy as np

def threshold(img, T):
    """Umbralización manual con valor fijo."""
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")
    return ((img > T) * 255).astype(np.uint8)


def otsu_threshold(img):
    """
    Calcula el umbral óptimo de Otsu y regresa (T, imagen binaria).
    Implementación propia usando solo NumPy.
    """
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    # Histograma normalizado (np solo como apoyo)
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    total = img.size
    prob = hist / total

    mejor_T = 0
    mejor_varianza = 0.0

    for T in range(1, 256):
        w0 = prob[:T].sum()
        w1 = prob[T:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(T) * prob[:T]) / w0
        mu1 = np.sum(np.arange(T, 256) * prob[T:]) / w1
        varianza_entre = w0 * w1 * (mu0 - mu1) ** 2
        if varianza_entre > mejor_varianza:
            mejor_varianza = varianza_entre
            mejor_T = T

    binaria = ((img > mejor_T) * 255).astype(np.uint8)
    return mejor_T, binaria


def adaptive_threshold(img, block_size=31, C=5):
    """
    Umbralización adaptativa: el umbral se calcula localmente
    para cada píxel usando la media de su vecindad.
    Implementación propia, útil para imágenes con iluminación no uniforme.
    """
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")
    if block_size % 2 == 0:
        block_size += 1

    filas, columnas = img.shape
    resultado = np.zeros((filas, columnas), dtype=np.uint8)
    padding = block_size // 2
    img_padded = np.pad(img, padding, mode='reflect')

    for i in range(filas):
        for j in range(columnas):
            bloque = img_padded[i:i+block_size, j:j+block_size]
            umbral_local = bloque.mean() - C
            resultado[i, j] = 255 if img[i, j] > umbral_local else 0

    return resultado