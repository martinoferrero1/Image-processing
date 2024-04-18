import cv2
import numpy as np
import matplotlib.pyplot as plt
import process as p 

def convertir_escala_grises(imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    return imagen_gris

def detectar_bordes(imagen_gris):
    bordes = cv2.Canny(imagen_gris, threshold1=30, threshold2=100)
    return bordes

def crecimiento_regiones(imagen_gris, semilla):
    h, w = imagen_gris.shape[:2]
    visitado = np.zeros_like(imagen_gris)
    cola = []
    cola.append(semilla)
    region = []
    
    while cola:
        punto = cola.pop(0)
        if visitado[punto] == 0:
            visitado[punto] = 1
            region.append(punto)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = punto[0] + dx, punto[1] + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        if visitado[nx, ny] == 0:
                            if abs(int(imagen_gris[nx, ny]) - int(imagen_gris[punto])) < 30:
                                cola.append((nx, ny))
    
    return region

def segmentar_imagen(imagen):
    imagen_gris = convertir_escala_grises(imagen)
    bordes = detectar_bordes(imagen_gris)
    p.view_image('Bordes: ',bordes)
    # Obtener dimensiones de la imagen
    h, w = imagen_gris.shape[:2]
    
    # Encontrar semilla para el crecimiento de regiones (por ejemplo, centroide del limón)
    semilla = (h//2, w//2)  # Usando centroide como semilla
    
    # Realizar crecimiento de regiones
    region_limones = crecimiento_regiones(imagen_gris, semilla)
    
    # Crear máscara para la región segmentada
    mascara = np.zeros_like(imagen_gris)
    for punto in region_limones:
        mascara[punto] = 255
    
    # Aplicar máscara a la imagen original
    imagen_segmentada = cv2.bitwise_and(imagen, imagen, mask=mascara)
    
    p.view_image('Segmentation: ',imagen_segmentada)
    
    return 
