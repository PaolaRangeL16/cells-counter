# setup.py
from setuptools import setup, find_packages

setup(
    name="cell_counter",
    version="1.0.0",
    author="PaolaRangeL16",
    description="Librería para el conteo automático de células en imágenes microscópicas",
    url="https://github.com/PaolaRangeL16/cells-counter",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "opencv-python",
    ],
    python_requires=">=3.8",
)
