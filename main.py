import cv2
import matplotlib.pyplot as plt

from cell_counter.preprocessing import to_gray, preprocess_image
from cell_counter.thresholding import otsu_threshold
from cell_counter.segmentation import connected_components, remove_small_regions, visualize_labels

# =========================
# 1. CARGAR IMAGEN
# =========================
img = cv2.imread("img.jpg")

if img is None:
    print(" No se pudo cargar la imagen")
    exit()

# =========================
# 2. ORIGINAL
# =========================
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("1. Imagen original")
plt.axis('off')
plt.show()

# =========================
# 3. ESCALA DE GRISES
# =========================
gray = to_gray(img)

plt.imshow(gray, cmap='gray')
plt.title("2. Escala de grises")
plt.axis('off')
plt.show()

# =========================
# 4. PREPROCESAMIENTO
# =========================
processed = preprocess_image(img)

plt.imshow(processed, cmap='gray')
plt.title("3. Preprocesada (filtro + gamma)")
plt.axis('off')
plt.show()

# =========================
# 5. BINARIZACIÓN (CLAVE)
# =========================
import cv2
import numpy as np
# threshold
binary = otsu_threshold(processed)

# 1. cerrar huecos
kernel = np.ones((5,5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 2. eliminar ruido
kernel = np.ones((3,3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 3. separar células pegadas
binary = cv2.erode(binary, kernel, iterations=1)


plt.imshow(binary, cmap='gray')
plt.title("4. Imagen binaria")
plt.axis('off')
plt.show()

# =========================
# 6. SEGMENTACIÓN
# =========================
labeled, num_cells = connected_components(binary)

print("Conteo SIN limpieza:", num_cells)

visualize_labels(labeled, title="5. Segmentación sin limpieza")

# =========================
# 7. LIMPIEZA
# =========================
labeled_clean, num_cells_clean = remove_small_regions(labeled, min_area=200)

print("Conteo CON limpieza:", num_cells_clean)

visualize_labels(labeled_clean, title="6. Segmentación con limpieza")
