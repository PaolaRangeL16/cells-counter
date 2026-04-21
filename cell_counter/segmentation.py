#def connected_components(img):
 #   return img
# Segmentación de regiones
# Hola mi gente nuevo toledana

import numpy as np
import matplotlib.pyplot as plt

def connected_components(binary_img):
    
    # Primero confirma que la imagen sea binaria
    # Si la imagen tiene 0 y 255, convertir a 0 y 1
    if binary_img.max() > 1:
        binary = np.where(binary_img == 255, 1, 0)
    else:
        binary = binary_img.copy()
    
    # Despues obtenemos dimensiones
    filas, columnas = binary.shape
    
    #3 Crear matriz de etiquetas (inicializada en 0)
    labeled = np.zeros((filas, columnas), dtype=np.int32)
    
    #4 Variable para llevar el conteo de etiquetas
    current_label = 1
    
    #5 Recorremos toda la imagen AHHHHHH =`(
    for i in range(filas):
        for j in range(columnas):
            # Si encontramos un píxel blanco (1) que NO ha sido etiquetado
            if binary[i, j] == 1 and labeled[i, j] == 0:
                # Iniciar una búsqueda para etiquetar toda la región conectada
                labeled = flood_fill(binary, labeled, i, j, current_label)
                current_label += 1  # Aumentar etiqueta para la siguiente célula

    #6 aqui cambia el número de células pq es current_label - 1 (porque empezó en 1)
    num_cells = current_label - 1
    
    return labeled, num_cells


def flood_fill(binary, labeled, start_i, start_j, label):  
    filas, columnas = binary.shape
    
    #usamos una lista como cola para BFS( o sea la busqueda de ancho)
    cola = [(start_i, start_j)]
    
    #punto inicial
    labeled[start_i, start_j] = label
    
    #si hay puntos por explorar
    while cola:
        #sacas el primer punto de la cola (FIFO)
        i, j = cola.pop(0)
        
        #checa la vecindad
      
        # Vecinos (fila, columna) relativos para 8-direcciones
        vecinos = [
            (-1, -1), (-1, 0), (-1, 1),  # Fila superior
            (0, -1),           (0, 1),   # Misma fila
            (1, -1),  (1, 0),  (1, 1)    # Fila inferior
        ]
        
        for di, dj in vecinos:
            ni, nj = i + di, j + dj  # coordenadas del vecino
            
            # Verificar que el vecino esté dentro de la imagen
            if 0 <= ni < filas and 0 <= nj < columnas:
                # Si el vecino es blanco (1) y no está etiquetado
                if binary[ni, nj] == 1 and labeled[ni, nj] == 0:
                    labeled[ni, nj] = label  # Etiquetar
                    cola.append((ni, nj))    # Agregar a la cola
    
    return labeled


def remove_small_regions(labeled_img, min_area=10):
    
    #eliminamos el ruido de la imagen etiquetada.
    
    #copiamos la imagen para no modificar la original
    cleaned = labeled_img.copy()
    
    #sacamos  las etiquetas únicas, no contamos el fondo 0
    labels = np.unique(labeled_img)
    labels = labels[labels != 0]  # Quitamos el fondo
    
    #calcular el área de cada etiqueta y ver si nos sirve o nel
    for label in labels:
        # Calcular área (número de píxeles con esa etiqueta)
        area = np.sum(labeled_img == label)
        
        # Si el área es menor que el mínimo, eliminar (convertir a fondo)
        if area < min_area:
            cleaned[labeled_img == label] = 0
    
    # Re-etiquetar para que las etiquetas sean consecutivas (1,2,3...)
    cleaned, num_cells = renumber_labels(cleaned)
    
    return cleaned, num_cells


def renumber_labels(labeled_img):
   #que las etiquetas sean continuas 
    # Obtener etiquetas únicas (excluyendo fondo)
    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Crear un diccionario de mapeo {etiqueta_vieja: etiqueta_nueva}
    mapping = {0: 0}  # El fondo sigue siendo 0
    for new, old in enumerate(unique_labels, start=1):
        mapping[old] = new
    
    # Aplicar el mapeo
    result = np.zeros_like(labeled_img)
    for old, new in mapping.items():
        result[labeled_img == old] = new
    
    return result, len(unique_labels)


def visualize_labels(labeled_img, title="Células detectadas"):
    #visualiza la imagen etiquetada con colores diferentes para cada célula.
    
    #sacamos el número de células
    unique_labels = np.unique(labeled_img)
    num_cells = len(unique_labels) - 1
    
    if num_cells == 0:
        print("No se detectaron células")
        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_img, cmap='gray')
        plt.title("No se detectaron células")
        plt.axis('off')
        plt.show()
        return None
    
    # Crear mapa de colores aleatorios pero fijos para reproducibilidad
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_cells + 1, 3))
    colors[0] = [0, 0, 0]  # Fondo en negro
    
    # Crear imagen RGB
    colored = np.zeros((labeled_img.shape[0], labeled_img.shape[1], 3), dtype=np.uint8)
    
    for i, label in enumerate(unique_labels):
        colored[labeled_img == label] = colors[label]
    
    # Mostrar
    plt.figure(figsize=(12, 8))
    plt.imshow(colored)
    plt.title(f"{title} - Células encontradas: {num_cells}", fontsize=14)
    plt.axis('off')
    
    # Agregar leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[label]/255, edgecolor='black', 
                             label=f'Célula {label}') 
                       for label in unique_labels if label != 0]
    # Limitar leyenda a máximo 10 elementos para no saturar
    if len(legend_elements) > 10:
        legend_elements = legend_elements[:10] + [Patch(facecolor='gray', label='...')]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return colored


def get_cell_properties(labeled_img):
    #Obtiene propiedades de cada célula (área, centroide)
    
    unique_labels = np.unique(labeled_img)
    unique_labels = unique_labels[unique_labels != 0]
    
    properties = {}
    
    for label in unique_labels:
        # Obtener coordenadas de la célula
        coords = np.argwhere(labeled_img == label)
        
        # Calcular área
        area = len(coords)
        
        # Calcular centroide (promedio de coordenadas)
        centroid_i = np.mean(coords[:, 0])
        centroid_j = np.mean(coords[:, 1])
        
        properties[label] = {
            'area': area,
            'centroid': (centroid_i, centroid_j)
        }
    
    return properties

#creo que ya payola, si no te corre o no te genera bien la imagen me dices y le cambio
