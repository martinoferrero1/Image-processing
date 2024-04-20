import cv2
import matplotlib.pyplot as plt
import numpy as np
import process 
import preprocess
import os

# Directorio donde se encuentran las imágenes
directorio_imagenes = "./images"
histogramas = []

# Tamaño deseado para las imágenes redimensionadas
nuevo_ancho = 500
nuevo_alto = 500

# for filename in os.listdir(directorio_imagenes):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         path_imagen = os.path.join(directorio_imagenes, filename)
#         img_redimensionada = preprocess.ajustar_imagen(path_imagen, nuevo_ancho, nuevo_alto)
#         cv2.imshow(filename, img_redimensionada)
#         cv2.waitKey(0)  # Esperar a que se presione una tecla para cerrar la ventana
#         imagen_with_gaussian = cv2.GaussianBlur(img_redimensionada, (7, 7), 2)
#         result = preprocess.resaltar_limon(imagen_with_gaussian)
#         result = cv2.resize(result, (img_redimensionada.shape[1], img_redimensionada.shape[0]))
#         result = cv2.bitwise_not(result)
        
#         kernel = np.ones((5,5),np.uint8)
#         result = cv2.dilate(result, kernel, iterations=5)
#         imagen_negra = np.zeros_like(img_redimensionada)
#         imagen_negra[result == 0] = img_redimensionada[result == 0]
        
#         result = process.process(imagen_negra)
#         cv2.imshow(filename, result)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

path_imagen = os.path.join(directorio_imagenes, "limones10.jpg")
img_redimensionada = preprocess.ajustar_imagen(path_imagen, nuevo_ancho, nuevo_alto)
process.view_image("inicio",img_redimensionada)
imagen_with_gaussian = cv2.GaussianBlur(img_redimensionada, (7, 7), 2)
result = preprocess.resaltar_limon(imagen_with_gaussian)
result = cv2.resize(result, (img_redimensionada.shape[1], img_redimensionada.shape[0]))
result = cv2.bitwise_not(result)

kernel = np.ones((5,5),np.uint8)
result = cv2.dilate(result, kernel, iterations=5)
imagen_negra = np.zeros_like(img_redimensionada)
imagen_negra[result == 0] = img_redimensionada[result == 0]

result = process.process(imagen_negra)
process.view_image("fin",result)
cv2.waitKey(0)
cv2.destroyAllWindows()