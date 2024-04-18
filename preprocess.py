import cv2
import os

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

