def threshold(img, T): #Función para crear imagen Blanco y Negro 

    filas, columnas = img.shape #Sacar las dimensiones 
    
    imgBN = np.zeros((filas, columnas), dtype=np.uint8) #Imagen nueva "vacia"

    for i in range(filas): #Llenado de la imagen
        for j in range(columnas):

            if img[i,j] > T: #Dependiendo del valor del pixel, se pone Blanco y Negro 
                imgBN[i,j] = 255
            else:
                imgBN[i,j] = 0

    return imgBN #Imagen en Blanco y Negro

def otsu_threshold(img):

    T, imgOtsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Formula para encontrar el umbral, mejo opcion para saber si es balnco o negro

    print("Umbral encontrado:", T) #solo es para saber que umbrañ calculo

    return imgOtsu #Regresa la función en blanci y negro 

def harris_corners(img):
    return img
