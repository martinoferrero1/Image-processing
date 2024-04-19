import cv2
import os
import numpy as np

def ajustar_imagen(imagen, nuevo_ancho, nuevo_alto):
    img = cv2.imread(imagen)
    img_redimensionada = cv2.resize(img, (nuevo_ancho, nuevo_alto))
    return img_redimensionada

def calcular_histograma(imagen):
    # Leer la imagen
    img = cv2.imread(imagen)
    
    # Calcular histograma para cada canal de color (BGR)
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
    
    return hist_b, hist_g, hist_r

def resaltar_limon(image):
    #image = cv2.imread(imagen)

    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para los limones en HSV
    #lower_lemon = np.array([0, 30, 30])
    #upper_lemon = np.array([180, 255, 255])
    lower_lemon = np.array([0, 10, 10])  # Umbral inferior para H, S y V
    upper_lemon = np.array([252, 255, 255])  # Umbral superior para H, S y V

    # Crear una máscara para los limones
    mask = cv2.inRange(hsv, lower_lemon, upper_lemon)

    # Invertir la máscara
    mask = cv2.bitwise_not(mask)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(image, image, mask=mask)

    # Definir el kernel para la operación de erosión
    kernel = np.ones((5, 5), np.uint8)

    # Aplicar la operación de erosión
    eroded_image = cv2.erode(result, kernel, iterations=3)

    final_result = cv2.bitwise_not(eroded_image)

    return final_result

