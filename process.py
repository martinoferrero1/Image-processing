import cv2
import matplotlib.pyplot as plt
import numpy as np

def yellow_detect(imagen):
    
    # Crear una máscara 
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([33, 255, 255])
    mask_yellow = cv2.inRange(imagen, lower_yellow, upper_yellow)

    # Aplicar la máscara 
    yellow_detected = cv2.bitwise_and(imagen, imagen, mask=mask_yellow)
    
    return yellow_detected

def delete_yellow(imagen,amarillo):

    # return imagen_sin_amarillo
    mask_umbral = (amarillo > 0).all(axis=2)  # True donde el color es amarillo

    # Invertir la máscara para obtener píxeles que no son amarillos
    mask_no_amarillo = np.logical_not(mask_umbral)

    # Convertir la máscara a tipo uint8 para usar con bitwise_and
    mask_no_amarillo = mask_no_amarillo.astype(np.uint8) * 255

    # Aplicar la máscara para eliminar píxeles amarillos de la imagen
    imagen_sin_amarillo = cv2.bitwise_and(imagen, imagen, mask=mask_no_amarillo)

    return imagen_sin_amarillo

def moho_detect(imagen_hsv,imagen_with_gaussian):
    # Definir rangos de color para el moho (verde amarillento)
    lower_verde = np.array([25, 20, 30])   # Umbral inferior en HSV
    upper_verde = np.array([80, 255, 255])  # Umbral superior en HSV

    # Crear una máscara para identificar el color del moho
    mask_moho = cv2.inRange(imagen_hsv, lower_verde, upper_verde)

    # Aplicar la máscara para segmentar las regiones de moho
    regiones_moho = cv2.bitwise_and(imagen_with_gaussian, imagen_with_gaussian, mask=mask_moho)
    
    return regiones_moho

def paint(imagen,imagen_moho):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_rgb_gris = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    
    # Reemplazar los píxeles en la segunda imagen donde la intersección es mayor que cero
    _, mask_binaria = cv2.threshold(imagen_moho, 1, 255, cv2.THRESH_BINARY)
    # imagen_rgb_gris[mask_binaria > 0] = imagen_moho[mask_binaria > 0]
    coordenadas = np.argwhere(mask_binaria > 0)

# Iterar sobre cada posición (fila, columna) donde mask_binaria > 0
    for coordenada in coordenadas:
        fila, columna = coordenada[:2]  # Obtener las primeras dos posiciones (fila y columna)
        imagen_rgb_gris[fila,columna] = imagen[fila,columna]
    return imagen_rgb_gris

def view_image(title,imagen):
    if imagen is None or not isinstance(imagen, np.ndarray):
        print(f"Error: La imagen '{title}' no es válida.")
        return
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Permite redimensionar la ventana
    cv2.resizeWindow(title, 600, 400)  
    cv2.imshow(title, imagen) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()  

def process():
    #CARGA DE IMAGEN
    nombre_archivo = r'.\\images\\limones1.jpg'
    imagen = cv2.imread(nombre_archivo)

    # Comprueba si la imagen se cargó correctamente
    if imagen is not None:
        
        view_image('Imagen original',imagen)
        
        #APLICA FILTRO GAUSSIANO (Disminuir ruido y eliminar detalles finos)
        imagen_with_gaussian = cv2.GaussianBlur(imagen, (7, 7), 2)
        view_image('Gaussiano',imagen_with_gaussian)
        
        #CAMBIA FORMATO BGR A HSV (Resalta color amarillo, lo hace homogeneo)
        imagen_hsv = cv2.cvtColor(imagen_with_gaussian, cv2.COLOR_BGR2HSV)
        view_image('RGB A HSV',imagen_hsv)
        
        #APLICAR FILTRO DE DETECCION AMARILLO
        imagen_amarillo= yellow_detect(imagen_hsv)
        view_image('FILTRO AMARILLO',imagen_amarillo)
        
        #QUITAR AMARILLO DE ORIGINAL
        imagen_sin_amarillo = delete_yellow(imagen_hsv,imagen_amarillo)
        view_image('SIN AMARILLO',imagen_sin_amarillo)

        #APLICAR FILTRO DE MOHO
        imagen_moho = moho_detect(imagen_sin_amarillo,imagen_with_gaussian)
        view_image('FILTRO MOHO',imagen_moho)
        
        #RESALTAR MOHO EN LA ORIGINAL GRIS
        imagen_resultante = paint(imagen,imagen_moho)

        #VER RESULTADOS
        view_image('Imagen resultado',imagen_resultante)


    else:
        print(f'Error: No se pudo abrir la imagen "{nombre_archivo}"')