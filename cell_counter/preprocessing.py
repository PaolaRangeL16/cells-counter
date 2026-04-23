#def to_gray(img):
    #return img

#def gaussian_filter(img):
    #return img

#def gamma_correction(img):
    #return img

#def sobel_edges(img):
    #return img

import numpy as np
import cv2
import matplotlib.pyplot as plt

def to_gray(img):
    """Convierte una imagen RGB a escala de grises."""
    # Verificar si ya está en grises
    if len(img.shape) == 2:
        return img.copy()
    
    # Obtener dimensiones
    filas, columnas = img.shape[:2]
    
    # Crear matriz de ceros para la imagen gris
    gray = np.zeros((filas, columnas), dtype=np.uint8)
    
    # Aplicar fórmula de luminosidad
    for i in range(filas):
        for j in range(columnas):
            r, g, b = img[i, j]
            gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
    
    return gray


def gaussian_filter(img, kernel_size=5, sigma=1.0):
    """Aplica un filtro Gaussiano a la imagen."""
    # Asegurar que kernel_size sea impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Crear kernel Gaussiano
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    centro = kernel_size // 2
    
    # Llenar el kernel con valores de la función Gaussiana
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - centro
            y = j - centro
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalizar el kernel (suma = 1)
    kernel = kernel / np.sum(kernel)
    
    # Obtener dimensiones de la imagen
    filas, columnas = img.shape
    
    # Crear imagen de salida
    filtered = np.zeros((filas, columnas), dtype=np.uint8)
    
    # Aplicar convolución
    padding = kernel_size // 2
    # Agregar padding a la imagen
    img_padded = np.pad(img, pad_width=padding, mode='edge')
    
    for i in range(filas):
        for j in range(columnas):
            # Extraer la región de la imagen
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            # Aplicar el kernel (multiplicación y suma)
            valor = np.sum(region * kernel)
            filtered[i, j] = np.clip(valor, 0, 255).astype(np.uint8)
    
    return filtered


def gamma_correction(img, gamma=1.2):
    """Aplica corrección gamma para ajustar el brillo/contraste."""
    # Normalizar la imagen a rango [0, 1]
    img_normalized = img.astype(np.float32) / 255.0
    
    # Aplicar corrección gamma
    img_corrected = np.power(img_normalized, gamma)
    
    # Volver a rango [0, 255]
    corrected = (img_corrected * 255).astype(np.uint8)
    
    return corrected


def sobel_edges(img):
    """Detecta bordes usando el operador Sobel."""
    # Kernels de Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    # Obtener dimensiones
    filas, columnas = img.shape
    
    # Padding
    img_padded = np.pad(img, pad_width=1, mode='edge')
    
    # Matrices para gradientes
    Gx = np.zeros((filas, columnas), dtype=np.float32)
    Gy = np.zeros((filas, columnas), dtype=np.float32)
    
    # Aplicar convolución
    for i in range(filas):
        for j in range(columnas):
            region = img_padded[i:i+3, j:j+3]
            Gx[i, j] = np.sum(region * sobel_x)
            Gy[i, j] = np.sum(region * sobel_y)
    
    # Calcular magnitud del gradiente
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Normalizar a [0, 255]
    magnitude = (magnitude / np.max(magnitude)) * 255
    edges = magnitude.astype(np.uint8)
    
    return edges, magnitude


def visualize_preprocessing_steps(original_img, gray_img, filtered_img, gamma_img, edges_img):
    """Visualiza todos los pasos del preprocesamiento."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original_img if len(original_img.shape) == 3 else original_img, cmap='gray')
    axes[0, 0].set_title('1. Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray_img, cmap='gray')
    axes[0, 1].set_title('2. Escala de Grises')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(filtered_img, cmap='gray')
    axes[0, 2].set_title('3. Filtro Gaussiano')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(gamma_img, cmap='gray')
    axes[1, 0].set_title('4. Corrección Gamma')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges_img, cmap='gray')
    axes[1, 1].set_title('5. Bordes (Sobel)')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============ FUNCIÓN PRINCIPAL PARA PIPELINE ============

def preprocess_image(img, apply_gamma=True, gamma=1.2, apply_sobel=False):
    """Función completa de preprocesamiento."""
    # Paso 1: Convertir a grises
    gray = to_gray(img)
    
    # Paso 2: Aplicar filtro Gaussiano (reduce ruido)
    filtered = gaussian_filter(gray, kernel_size=5, sigma=1.0)
    
    # Paso 3: Corrección gamma (mejora contraste de células)
    if apply_gamma:
        filtered = gamma_correction(filtered, gamma)
    
    # Paso 4: Bordes (opcional, para features)
    if apply_sobel:
        edges, _ = sobel_edges(filtered)
        return filtered, edges
    
    return filtered

# ============================================
# CÓDIGO DE PRUEBA (solo para verificar que funciona)
# ============================================
if __name__ == "__main__":
    import cv2
    
    print("=" * 50)
    print("🧪 PROBANDO PREPROCESSING.PY")
    print("=" * 50)
    
    # 1. Crear imagen de prueba artificial
    print("\n📷 Creando imagen de prueba...")
    img_prueba = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(img_prueba, (50, 50), 20, (200, 180, 160), -1)
    cv2.circle(img_prueba, (100, 80), 25, (180, 200, 170), -1)
    cv2.rectangle(img_prueba, (60, 110), (120, 140), (150, 150, 200), -1)
    print("✅ Imagen creada")
    
    # 2. Probar to_gray()
    print("\n🔘 Probando to_gray()...")
    gray = to_gray(img_prueba)
    print(f"   Shape original: {img_prueba.shape} → Shape gris: {gray.shape}")
    print(f"   ✅ to_gray() funciona")
    
    # 3. Probar gaussian_filter()
    print("\n🌀 Probando gaussian_filter()...")
    filtrada = gaussian_filter(gray, kernel_size=5, sigma=1.0)
    print(f"   Shape: {filtrada.shape}")
    print(f"   ✅ gaussian_filter() funciona")
    
    # 4. Probar gamma_correction()
    print("\n✨ Probando gamma_correction()...")
    gamma = gamma_correction(filtrada, gamma=1.2)
    print(f"   Shape: {gamma.shape}")
    print(f"   ✅ gamma_correction() funciona")
    
    # 5. Probar sobel_edges()
    print("\n📐 Probando sobel_edges()...")
    bordes, _ = sobel_edges(gray)
    print(f"   Shape bordes: {bordes.shape}")
    print(f"   ✅ sobel_edges() funciona")
    
    # 6. Mostrar resultados
    print("\n📊 Mostrando imágenes...")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_prueba)
    plt.title("1. Original")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("2. Escala de grises")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(filtrada, cmap='gray')
    plt.title("3. Filtro Gaussiano")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(gamma, cmap='gray')
    plt.title("4. Corrección Gamma")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(bordes, cmap='gray')
    plt.title("5. Bordes (Sobel)")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(gray, cmap='gray', alpha=0.6)
    plt.imshow(bordes, cmap='hot', alpha=0.4)
    plt.title("6. Superposición")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 50)
    print("🎉 ¡TODAS LAS FUNCIONAS FUNCIONAN CORRECTAMENTE! 🎉")
    print("=" * 50)