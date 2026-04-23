import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# 1. COMPONENTES CONECTADOS
def connected_components(binary_img):
    # Validación
    if len(binary_img.shape) != 2:
        raise ValueError("La imagen debe ser 2D (escala de grises/binaria)")

    # Asegurar 0/1
    if binary_img.max() > 1:
        binary = np.where(binary_img == 255, 1, 0)
    else:
        binary = binary_img.copy()

    filas, columnas = binary.shape
    labeled = np.zeros((filas, columnas), dtype=np.int32)

    current_label = 1

    for i in range(filas):
        for j in range(columnas):
            if binary[i, j] == 1 and labeled[i, j] == 0:
                flood_fill(binary, labeled, i, j, current_label)
                current_label += 1

    num_cells = current_label - 1
    return labeled, num_cells

# 2. FLOOD FILL (BFS)
def flood_fill(binary, labeled, start_i, start_j, label):
    filas, columnas = binary.shape

    cola = deque([(start_i, start_j)])
    labeled[start_i, start_j] = label

    vecinos = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    while cola:
        i, j = cola.popleft()

        for di, dj in vecinos:
            ni, nj = i + di, j + dj

            if 0 <= ni < filas and 0 <= nj < columnas:
                if binary[ni, nj] == 1 and labeled[ni, nj] == 0:
                    labeled[ni, nj] = label
                    cola.append((ni, nj))


# 3. ELIMINAR REGIONES PEQUEÑAS
def remove_small_regions(labeled_img, min_area=10):

    cleaned = labeled_img.copy()

    labels = np.unique(labeled_img)
    labels = labels[labels != 0]

    for label in labels:
        area = np.sum(labeled_img == label)

        if area < min_area:
            cleaned[labeled_img == label] = 0

    cleaned, num_cells = renumber_labels(cleaned)

    return cleaned, num_cells


# 4. RE-ETIQUETADO
def renumber_labels(labeled_img):

    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]

    mapping = {0: 0}

    for new, old in enumerate(unique_labels, start=1):
        mapping[old] = new

    result = np.zeros_like(labeled_img)

    for old, new in mapping.items():
        result[labeled_img == old] = new

    return result, len(unique_labels)

# 5. VISUALIZACIÓN
def visualize_labels(labeled_img, title="Células detectadas"):

    unique_labels = np.unique(labeled_img)
    num_cells = len(unique_labels) - 1

    if num_cells == 0:
        print("No se detectaron células")
        plt.imshow(labeled_img, cmap='gray')
        plt.title("No se detectaron células")
        plt.axis('off')
        plt.show()
        return None

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_cells + 1, 3))
    colors[0] = [0, 0, 0]

    colored = np.zeros((labeled_img.shape[0], labeled_img.shape[1], 3), dtype=np.uint8)

    for label in unique_labels:
        colored[labeled_img == label] = colors[label]

    plt.figure(figsize=(10, 7))
    plt.imshow(colored)
    plt.title(f"{title} - Células: {num_cells}")
    plt.axis('off')
    plt.show()

    return colored

# 6. PROPIEDADES
def get_cell_properties(labeled_img):

    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]

    properties = {}

    for label in unique_labels:
        coords = np.argwhere(labeled_img == label)

        area = len(coords)

        centroid_i = np.mean(coords[:, 0])
        centroid_j = np.mean(coords[:, 1])

        properties[label] = {
            'area': area,
            'centroid': (centroid_i, centroid_j)
        }

    return properties
