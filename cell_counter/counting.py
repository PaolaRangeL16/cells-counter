# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:04:43 2026

@author: paor9
"""
# counting.py
import numpy as np
from .preprocessing import preprocess_image
from .thresholding import otsu_threshold
from .segmentation import watershed_segment, remove_small_regions, filter_by_circularity
from .morphology import morphological_close, morphological_open, erode


def count_cells(image, min_area=500, min_distance=20, min_circularity=0.4):
    """
    Pipeline completo para contar células en imagen microscópica con tinción púrpura.

    Parámetros:
        image           : np.ndarray — imagen BGR cargada con cv2.imread
        min_area        : int   — área mínima en píxeles (500 para esta imagen)
        min_distance    : int   — distancia mínima entre centros (20 para esta imagen)
        min_circularity : float — circularidad mínima (0.4 para células redondas)
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Se esperaba una imagen válida como array de NumPy")

    # 1. Preprocesamiento con canal púrpura
    processed = preprocess_image(image)

    # 2. Otsu sobre canal púrpura
    T, binary = otsu_threshold(processed)

    # 3. Morfología
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    binary = morphological_close(binary, kernel5)
    binary = morphological_open(binary, kernel3)

    # 4. Watershed con distancia mayor para estas células
    labeled, _ = watershed_segment(binary, min_distance=min_distance)

    # 5. Eliminar regiones pequeñas (artefactos y manchas)
    labeled, _ = remove_small_regions(labeled, min_area=min_area)

    # 6. Filtrar por circularidad
    labeled, num_cells = filter_by_circularity(labeled, min_circularity=min_circularity)

    return {
        'count':     num_cells,
        'labeled':   labeled,
        'binary':    binary,
        'processed': processed,
        'threshold': T
    }
