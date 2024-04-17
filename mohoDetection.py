import cv2
import matplotlib.pyplot as plt
import numpy as np
import process as p 

#CARGA DE IMAGEN
nombre_archivo = r'.\\images\\limones1.jpg'
imagen = cv2.imread(nombre_archivo)

# Comprueba si la imagen se cargó correctamente
if imagen is not None:
    
    p.view_image('Imagen original',imagen)
    
    #APLICA FILTRO GAUSSIANO (Disminuir ruido y eliminar detalles finos)
    imagen_with_gaussian = cv2.GaussianBlur(imagen, (7, 7), 2)
    p.view_image('Gaussiano',imagen_with_gaussian)
    
    #CAMBIA FORMATO BGR A HSV (Resalta color amarillo, lo hace homogeneo)
    imagen_hsv = cv2.cvtColor(imagen_with_gaussian, cv2.COLOR_BGR2HSV)
    p.view_image('RGB A HSV',imagen_hsv)
    
    #APLICAR FILTRO DE DETECCION AMARILLO
    imagen_amarillo= p.yellow_detect(imagen_hsv)
    p.view_image('FILTRO AMARILLO',imagen_amarillo)
    
    #QUITAR AMARILLO DE ORIGINAL
    imagen_sin_amarillo = p.delete_yellow(imagen_hsv,imagen_amarillo)
    p.view_image('SIN AMARILLO',imagen_sin_amarillo)

    #APLICAR FILTRO DE MOHO
    imagen_moho = p.moho_detect(imagen_sin_amarillo,imagen_with_gaussian)
    p.view_image('FILTRO MOHO',imagen_moho)
    
    #RESALTAR MOHO EN LA ORIGINAL GRIS
    imagen_resultante = p.paint(imagen,imagen_moho)

    #VER RESULTADOS
    p.view_image('Imagen resultado',imagen_resultante)


else:
    print(f'Error: No se pudo abrir la imagen "{nombre_archivo}"')
