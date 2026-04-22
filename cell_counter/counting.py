from .thresholding import otsu_threshold
from .segmentation import connected_components
from .preprocessing import to_gray

def count_cells(image):

    # 1. gris
    gray = to_gray(image)

    # 2. umbral
    binary = otsu_threshold(gray)

    # 3. segmentación
    labeled, num_cells = connected_components(binary)

    return num_cells
