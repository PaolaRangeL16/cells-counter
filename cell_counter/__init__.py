# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:24:43 2026

@author: paor9
"""
# __init__.py
from .counting import count_cells
from .preprocessing import preprocess_image, to_gray, gaussian_filter, gamma_correction
from .thresholding import threshold, otsu_threshold, adaptive_threshold
from .segmentation import (
    connected_components,
    remove_small_regions,
    visualize_labels,
    get_cell_properties
)
from .morphology import erode, dilate, morphological_close, morphological_open

__version__ = "1.0.0"
__author__ = "VMP"
