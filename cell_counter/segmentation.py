# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:13:33 2026

@author: paor9
"""
# segmentation.py
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import ndimage


# =========================
# 1. FLOOD FILL (BFS)
# =========================
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


# =========================
# 2. COMPONENTES CONECTADOS
# =========================
def connected_components(binary_img):
    """
    Etiqueta regiones conectadas en una imagen binaria usando BFS.
    Conectividad de 8 vecinos.
    """
    if len(binary_img.shape) != 2:
        raise ValueError("La imagen debe ser 2D (escala de grises/binaria)")

    binary = np.where(binary_img > 0, 1, 0)
    filas, columnas = binary.shape
    labeled = np.zeros((filas, columnas), dtype=np.int32)
    current_label = 1

    for i in range(filas):
        for j in range(columnas):
            if binary[i, j] == 1 and labeled[i, j] == 0:
                flood_fill(binary, labeled, i, j, current_label)
                current_label += 1

    return labeled, current_label - 1


# =========================
# 3. WATERSHED
# =========================
def watershed_segment(binary, min_distance=15):
    """
    Separa células pegadas usando transformada de distancia + watershed.
    Útil cuando la erosión simple no logra separar células que se tocan.
    """
    binary_bool = (binary > 0).astype(np.uint8)

    # Distancia de cada píxel al borde más cercano
    distance = ndimage.distance_transform_edt(binary_bool)

    # Máximos locales = centros de células
    local_max = (distance == ndimage.maximum_filter(distance, size=min_distance))

    # Etiquetar semillas
    seeds, _ = ndimage.label(local_max)

    # Watershed desde las semillas, enmascarado al área binaria
    labeled = ndimage.watershed_ift(
        (255 - distance).astype(np.uint8), seeds
    ) * binary_bool

    num_cells = int(labeled.max())
    return labeled.astype(np.int32), num_cells


# =========================
# 4. ELIMINAR REGIONES PEQUEÑAS
# =========================
def remove_small_regions(labeled_img, min_area=200):
    """
    Elimina regiones cuya área en píxeles sea menor a min_area.
    """
    cleaned = labeled_img.copy()
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]

    for label in labels:
        area = np.sum(labeled_img == label)
        if area < min_area:
            cleaned[labeled_img == label] = 0

    cleaned, num_cells = renumber_labels(cleaned)
    return cleaned, num_cells


# =========================
# 5. FILTRAR POR CIRCULARIDAD
# =========================
def filter_by_circularity(labeled_img, min_circularity=0.3):
    """
    Elimina regiones que no son aproximadamente circulares.
    Circularidad = 4π * área / perímetro²
    1.0 = círculo perfecto. Espacios irregulares entre células
    tienen circularidad baja y son eliminados.
    """
    result = np.zeros_like(labeled_img)
    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]
    new_label = 1

    for label in unique_labels:
        mask = (labeled_img == label).astype(np.uint8)
        area = int(mask.sum())

        # Perímetro: píxeles en el borde (diferencia entre máscara y su erosión)
        eroded = ndimage.binary_erosion(mask).astype(np.uint8)
        perimeter = int(mask.sum() - eroded.sum())

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity >= min_circularity:
            result[labeled_img == label] = new_label
            new_label += 1

    return result, new_label - 1


# =========================
# 6. RE-ETIQUETADO
# =========================
def renumber_labels(labeled_img):
    """
    Re-etiqueta las regiones de forma consecutiva tras eliminar algunas.
    """
    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]
    result = np.zeros_like(labeled_img)
    for new, old in enumerate(unique_labels, start=1):
        result[labeled_img == old] = new
    return result, len(unique_labels)


# =========================
# 7. PROPIEDADES DE CÉLULAS
# =========================
def get_cell_properties(labeled_img):
    """
    Calcula área, centroide y bounding box de cada célula detectada.
    """
    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]
    properties = {}

    for label in unique_labels:
        coords = np.argwhere(labeled_img == label)
        area = len(coords)
        centroid_i = float(np.mean(coords[:, 0]))
        centroid_j = float(np.mean(coords[:, 1]))
        bbox = (
            int(coords[:, 0].min()), int(coords[:, 1].min()),
            int(coords[:, 0].max()), int(coords[:, 1].max())
        )
        properties[label] = {
            'area':     area,
            'centroid': (centroid_i, centroid_j),
            'bbox':     bbox
        }

    return properties


# =========================
# 8. VISUALIZACIÓN
# =========================
def visualize_labels(labeled_img, title="Células detectadas", show=True):
    """
    Colorea cada región etiquetada con un color distinto.
    Regresa la imagen coloreada como array.
    """
    unique_labels = np.unique(labeled_img)
    num_cells = len(unique_labels) - 1  # excluir fondo (0)

    if num_cells == 0:
        print("No se detectaron células")
        if show:
            plt.imshow(labeled_img, cmap='gray')
            plt.title("No se detectaron células")
            plt.axis('off')
            plt.show()
        return None

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(labeled_img.max() + 1, 3))
    colors[0] = [0, 0, 0]  # fondo negro

    colored = np.zeros((*labeled_img.shape, 3), dtype=np.uint8)
    for label in unique_labels:
        colored[labeled_img == label] = colors[label]

    if show:
        plt.figure(figsize=(10, 7))
        plt.imshow(colored)
        plt.title(f"{title} — Células: {num_cells}")
        plt.axis('off')
        plt.show()

    return colored
