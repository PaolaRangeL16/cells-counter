import cv2
from cell_counter.counting import count_cells

img = cv2.imread("imagen.jpg")

count = count_cells(img)

print("Número de células:", count)
