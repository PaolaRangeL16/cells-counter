# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:24:43 2026

@author: paor9
"""
# __init__.py
from .counting import count_cells
from .preprocessing import preprocess_image, to_gray, gaussian_filter, gamma_correction, extract_purple_channel, frequency_filter
from .thresholding import threshold, otsu_threshold, adaptive_threshold
from .morphology import erode, dilate, morphological_close, morphological_open
from .edge_detection import sobel, prewitt, laplacian, hough_circles
from .segmentation import (
    connected_components,
    watershed_segment,
    remove_small_regions,
    filter_by_circularity,
    visualize_labels,
    get_cell_properties
)


__version__ = "1.0.0"
__author__ = "VMP"
