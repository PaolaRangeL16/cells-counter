# main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

from cell_counter.preprocessing import to_gray, preprocess_image, extract_purple_channel
from cell_counter.thresholding import otsu_threshold
from cell_counter.morphology import morphological_close, morphological_open
from cell_counter.segmentation import (
    watershed_segment,
    remove_small_regions,
    filter_by_circularity,
    visualize_labels,
    get_cell_properties
)
from cell_counter.counting import count_cells


# 1. CARGAR
img = cv2.imread("i9.png")
if img is None:
    print("No se pudo cargar la imagen")
    exit()

# 2. ORIGINAL
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("1. Imagen original")
plt.axis('off')
plt.show()

# 3. CANAL PÚRPURA
purple = extract_purple_channel(img)
plt.imshow(purple, cmap='gray')
plt.title("2. Canal púrpura extraído")
plt.axis('off')
plt.show()

# 4. PREPROCESAMIENTO
processed = preprocess_image(img)
plt.imshow(processed, cmap='gray')
plt.title("3. Preprocesada (canal púrpura + gaussiano + gamma)")
plt.axis('off')
plt.show()

# 5. OTSU
T, binary = otsu_threshold(processed)
print(f"Umbral de Otsu: {T}")
plt.imshow(binary, cmap='gray')
plt.title(f"4. Binarización Otsu (T={T})")
plt.axis('off')
plt.show()

# 6. MORFOLOGÍA
kernel5 = np.ones((5, 5), dtype=np.uint8)
kernel3 = np.ones((3, 3), dtype=np.uint8)

binary = morphological_close(binary, kernel5)
plt.imshow(binary, cmap='gray')
plt.title("5. Cierre morfológico")
plt.axis('off')
plt.show()

binary = morphological_open(binary, kernel3)
plt.imshow(binary, cmap='gray')
plt.title("6. Apertura morfológica")
plt.axis('off')
plt.show()

# 7. WATERSHED
labeled_ws, num_ws = watershed_segment(binary, min_distance=20)
print(f"Células tras watershed: {num_ws}")
visualize_labels(labeled_ws, title="7. Watershed")

# 8. LIMPIEZA POR ÁREA
labeled_area, num_area = remove_small_regions(labeled_ws, min_area=500)
print(f"Células tras limpieza por área: {num_area}")
visualize_labels(labeled_area, title="8. Limpieza por área")

# 9. CIRCULARIDAD
labeled_final, num_final = filter_by_circularity(labeled_area, min_circularity=0.4)
print(f"Células detectadas: {num_final}")
visualize_labels(labeled_final, title="9. Resultado final")

# 10. PROPIEDADES
props = get_cell_properties(labeled_final)
print(f"\n{'='*40}")
print(f"  TOTAL DE CÉLULAS DETECTADAS: {num_final}")
print(f"{'='*40}")
for label, data in props.items():
    print(f"  Célula {label:>3} | Área: {data['area']:>6} px | "
          f"Centroide: ({data['centroid'][0]:.1f}, {data['centroid'][1]:.1f})")

# 11. VERIFICACIÓN
resultado = count_cells(img)
print(f"\n✅ count_cells() detectó: {resultado['count']} células")
print(f"   Umbral de Otsu: {resultado['threshold']}")