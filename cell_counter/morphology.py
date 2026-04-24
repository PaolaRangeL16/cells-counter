# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:31:31 2026

@author: paor9
"""

# morphology.py
import numpy as np

def _apply_morphology(img, kernel, func):
    """
    Función base para erosión y dilatación.
    Aplica 'func' (np.min o np.max) sobre cada vecindad definida por el kernel.
    """
    binary = (img > 0).astype(np.uint8)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    result = np.zeros_like(binary)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = func(region[kernel == 1])
    return result * 255


def erode(img, kernel):
    """
    Erosión morfológica: reduce regiones blancas.
    Útil para separar células pegadas.
    """
    return _apply_morphology(img, kernel, np.min)


def dilate(img, kernel):
    """
    Dilatación morfológica: expande regiones blancas.
    """
    return _apply_morphology(img, kernel, np.max)


def morphological_close(img, kernel):
    """
    Cierre morfológico: dilatar → erosionar.
    Cierra huecos internos dentro de las células.
    """
    return erode(dilate(img, kernel), kernel)


def morphological_open(img, kernel):
    """
    Apertura morfológica: erosionar → dilatar.
    Elimina ruido pequeño fuera de las células.
    """
    return dilate(erode(img, kernel), kernel)