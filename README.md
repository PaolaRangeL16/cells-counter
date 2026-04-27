# cells-counter

Librería en Python para contar células uninucleadas en imágenes microscópicas. Desarrollada como proyecto universitario para la materia de Procesamiento Digital de Imágenes.

El sistema toma una imagen microscópica y regresa el número de células uninucleadas detectadas, pasando por etapas de preprocesamiento, umbralización, morfología y segmentación. La mayoría de las funciones están implementadas desde cero usando NumPy.

## Requisitos

- Python 3.8+
- numpy
- matplotlib
- scipy
- opencv-python

## Instalación

git clone https://github.com/PaolaRangeL16/cells-counter.git
cd cells-counter
pip install -e .

O directo desde GitHub:

pip install git+https://github.com/PaolaRangeL16/cells-counter.git


## Uso básico

```python
import cv2
from cell_counter import count_cells

img = cv2.imread("img.jpg")
resultado = count_cells(img)

print(f"Células detectadas: {resultado['count']}")
```

El diccionario que regresa `count_cells` también incluye las imágenes intermedias del pipeline por si quieres visualizarlas:

```python
resultado['processed']  # imagen preprocesada
resultado['binary']     # imagen binarizada
resultado['labeled']    # imagen segmentada con etiquetas
resultado['threshold']  # valor T calculado por Otsu
```


## Demo

Para ver todo el pipeline paso a paso con una imagen de prueba:

python main.py

Muestra cada etapa: canal de color, preprocesamiento, binarización, morfología, watershed y resultado final.


## Estructura

El paquete principal es la carpeta `cell_counter/`. Dentro de ella, `preprocessing.py` 
se encarga de preparar la imagen antes de analizarla: extrae el canal de color púrpura, 
aplica el filtro gaussiano y ajusta el contraste con corrección gamma. `thresholding.py` 
contiene los algoritmos de umbralización, incluyendo la implementación propia de Otsu y 
el umbral adaptativo. `morphology.py` tiene las operaciones morfológicas (erosión, 
dilatación, apertura y cierre) todas implementadas desde cero. `segmentation.py` es el 
módulo más complejo: aquí vive el BFS para componentes conectados, el watershed para 
separar células pegadas, el filtro de circularidad y las funciones de visualización. 
Finalmente, `counting.py` conecta todo el pipeline en una sola función. Fuera de la 
carpeta del paquete está `main.py`, que es el script de demostración que muestra cada 
etapa visualmente.


## Parámetros

`count_cells` tiene tres parámetros opcionales que puedes ajustar dependiendo de tu imagen:

```python
resultado = count_cells(
    img,
    min_area=500,         # área mínima en píxeles para considerar una célula
    min_distance=20,      # separación mínima entre centros (watershed)
    min_circularity=0.4   # qué tan redonda debe ser la región (1.0 = círculo perfecto)
)
```

Si el conteo está contando de más, sube `min_area` o `min_circularity`. Si está contando de menos, bájalos.

---

## Notas

- Funciona bien con imágenes de tinción Giemsa o H&E donde las células son púrpuras sobre fondo claro.
- Las funciones principales (Otsu, gaussiano, BFS, morfología) están implementadas desde cero sin depender de OpenCV para la lógica central.
- OpenCV se usa únicamente para cargar imágenes (`cv2.imread`) y mostrar la imagen original.
- Si las células de tu imagen son muy pequeñas o muy grandes, el parámetro que más ayuda a ajustar es `min_area`.

---

## Autoras

Proyecto desarrollado para la materia de Procesamiento Digital de Imágenes.  
Victoria Díaz de León
Mia Regina Jiménez Ruiz Esparza
Paola Marisol Rangel López
[github.com/PaolaRangeL16/cells-counter](https://github.com/PaolaRangeL16/cells-counter)
