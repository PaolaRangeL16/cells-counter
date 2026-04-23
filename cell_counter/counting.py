from .preprocessing import preprocess_image
from .thresholding import otsu_threshold
from .segmentation import connected_components, remove_small_regions

def count_cells(image):

    # 1. preprocessing completo
    processed = preprocess_image(image)

    # 2. umbralización
    binary = otsu_threshold(processed)

    # 3. segmentación
    labeled, num_cells = connected_components(binary)

    # 4. limpieza de ruido (CLAVE)
    labeled, num_cells = remove_small_regions(labeled, min_area=20)

    return num_cells
