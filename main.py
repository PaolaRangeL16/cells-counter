import cv2
from cell_counter.counting import count_cells
from cell_counter.preprocessing import preprocess_image
from cell_counter.thresholding import otsu_threshold
from cell_counter.segmentation import connected_components, remove_small_regions, visualize_labels

# 1. Cargar imagen
img = cv2.imread("imagen.jpg")

if img is None:
    print("❌ No se pudo cargar la imagen")
    exit()

# 2. Usar pipeline completo (conteo)
num_cells = count_cells(img)

print("Número de células:", num_cells)

# Para visualización / presentación)


# Repetimos pipeline para obtener imagen etiquetada
processed = preprocess_image(img)
binary = otsu_threshold(processed)
labeled, _ = connected_components(binary)
labeled, _ = remove_small_regions(labeled, min_area=20)

# Mostrar resultado
visualize_labels(labeled, title="Resultado final")
