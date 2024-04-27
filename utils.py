import cv2
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------PREPROCESAMIENTO----------------------------------------------------#

def ajustar_imagen(imagen, nuevo_ancho, nuevo_alto):
    img_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
    return img_redimensionada

def preprocess(imagen):
    nuevo_ancho = 1000
    nuevo_alto = 1000
    img_redimensionada = ajustar_imagen(imagen, nuevo_ancho, nuevo_alto)
    result = cv2.GaussianBlur(img_redimensionada, (7, 7), 2)
    return result

#-----------------------------------------------------PROCESAMIENTO------------------------------------------------------#

def segmentar_limon(imagen):
    #image = cv2.imread(imagen)
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    # Definir el rango de color para los limones en HSV
    lower_lemon = np.array([0, 10, 10])  # Umbral inferior para H, S y V
    upper_lemon = np.array([252, 255, 255])  # Umbral superior para H, S y V
    # Crear una máscara para los limones
    mask = cv2.inRange(hsv, lower_lemon, upper_lemon)
    #view_image("(1)", mask)
    # Invertir la máscara
    mask = cv2.bitwise_not(mask)
    #view_image("(2)", mask)
    # Aplicar la máscara a la imagen original
    result_parcial = cv2.bitwise_and(imagen, imagen, mask=mask)
    #view_image("(3)", result_parcial)
    # Definir el kernel para la operación de erosión
    kernel = np.ones((5, 5), np.uint8)
    # Aplicar la operación de erosión
    eroded_image = cv2.erode(result_parcial, kernel, iterations=3) #opening alternativa
    result = cv2.bitwise_not(eroded_image)
    #view_image("(4)", result)
    result = cv2.resize(result, (result_parcial.shape[1], result_parcial.shape[0]))
    final_result = cv2.bitwise_not(result)
    #view_image("(5)", final_result)
    result = cv2.dilate(final_result, kernel, iterations=5)
    imagen_segmentada = np.zeros_like(imagen)
    imagen_segmentada[result == 0] = imagen[result == 0]
    #view_image("(6)", imagen_negra)
    return imagen_segmentada

def color_detect(imagen,lower_yellow,upper_yellow):
    
    mask_yellow = cv2.inRange(imagen, lower_yellow, upper_yellow)
    # Aplicar la máscara 
    yellow_detected = cv2.bitwise_and(imagen, imagen, mask=mask_yellow)
    return yellow_detected

def delete_color(imagen,amarillo):

    # return imagen_sin_amarillo
    mask_umbral = (amarillo > 0).all(axis=2)  # True donde el color es amarillo

    # Invertir la máscara para obtener píxeles que no son amarillos
    mask_no_amarillo = np.logical_not(mask_umbral)

    # Convertir la máscara a tipo uint8 para usar con bitwise_and
    mask_no_amarillo = mask_no_amarillo.astype(np.uint8) * 255

    # Aplicar la máscara para eliminar píxeles amarillos de la imagen
    imagen_sin_amarillo = cv2.bitwise_and(imagen, imagen, mask=mask_no_amarillo)

    return imagen_sin_amarillo

def paint(imagen,imagen_moho, color):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_rgb_gris = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    
    # Reemplazar los píxeles en la segunda imagen donde la intersección es mayor que cero
    _, mask_binaria = cv2.threshold(imagen_moho, 1, 255, cv2.THRESH_BINARY)
    coordenadas = np.argwhere(mask_binaria > 0)

# Iterar sobre cada posición (fila, columna) donde mask_binaria > 0
    for coordenada in coordenadas:
        fila, columna = coordenada[:2]  # Obtener las primeras dos posiciones (fila y columna)
        imagen_rgb_gris[fila,columna] = color
    return imagen_rgb_gris
    
def add_color_pixels(image_dest, image_source, color_to_add):
    # Convertir el color deseado a formato BGR (sin transparencia)
    color_to_add_bgr = np.array([color_to_add], dtype=np.uint8)
    # Encontrar los píxeles en la imagen de origen que coinciden con el color deseado
    mask_color = np.all(image_source == color_to_add_bgr, axis=-1)
    # Copiar los píxeles del color deseado desde la imagen de origen a la imagen de destino
    image_dest[mask_color] = color_to_add
    return image_dest

def process(imagen):

    imagen = segmentar_limon(imagen)
    #view_image('Imagen original',imagen)
    
    #APLICA FILTRO GAUSSIANO (Disminuir ruido y eliminar detalles finos)
    imagen_with_gaussian = cv2.GaussianBlur(imagen, (7, 7), 2)
    #view_image('Gaussiano',imagen_with_gaussian)
    
    #CAMBIA FORMATO BGR A HSV (Resalta color amarillo, lo hace homogeneo)
    imagen_hsv = cv2.cvtColor(imagen_with_gaussian, cv2.COLOR_BGR2HSV)
    #view_image('RGB A HSV',imagen_hsv)
    
    #APLICAR FILTRO DE VERDE
    # Ajustar el rango para el filtro de detección de verde en HSV
    lower_verde = np.array([30, 18, 0])   # Umbral inferior en HSV para tonos verdes oscuros (tonalidad, saturación, luminosidad)
    upper_verde = np.array([180, 255, 255])  # Umbral superior en HSV para tonos verdes (tonalidad, saturación, luminosidad)
    imagen_verde = color_detect(imagen_hsv,lower_verde,upper_verde)
    #view_image('FILTRO MOHO',imagen_moho)
    color_verde = np.array([50, 200, 50])
    #RESALTAR MOHO EN LA ORIGINAL GRIS
    imagen_con_verde = paint(imagen,imagen_verde,color_verde)
    view_image("Parte verde detectada", imagen_con_verde)
    imagen_sin_verde = delete_color(imagen_hsv,imagen_verde)
    view_image("Imagen con extraccion de verde detectado", imagen_sin_verde)
    
    #APLICAR FILTRO DE DETECCION BLANCO

    # Ajustar el rango inferior para el filtro de detección de blanco en HSV
    lower_white = np.array([0, 0, 20])   # Umbral inferior en HSV para tonos blancos y suaves (tonalidad, saturación, luminosidad)
    upper_white = np.array([255, 105, 255])  # Umbral superior en HSV para blancos y grises (tonalidad, saturación, luminosidad)
    imagen_blanco = color_detect(imagen_sin_verde,lower_white,upper_white)
    #view_image('FILTRO AMARILLO',imagen_amarillo)
    color_blanco = np.array([255, 255, 255])
    imagen_con_blanco = paint(imagen,imagen_blanco,color_blanco)
    view_image("Parte blanca detectada", imagen_con_blanco)
    imagen_sin_blanco = delete_color(imagen_sin_verde,imagen_blanco)
    view_image("Imagen con extraccion de blanco detectado", imagen_sin_blanco)
    
    #APLICAR FILTRO DE DETECCION AMARILLO
        # Crear una máscara 
    lower_yellow = np.array([0, 106, 0])   # Umbral inferior en HSV para tonos amarillos (tonalidad, saturación, luminosidad)
    upper_yellow = np.array([29, 255, 255])   # Umbral superior en HSV para tonos amarillos (tonalidad, saturación, luminosidad)
    imagen_amarillo= color_detect(imagen_sin_blanco,lower_yellow,upper_yellow)
    #view_image('FILTRO AMARILLO',imagen_amarillo)
    color_amarillo = np.array([51, 255, 255])
    imagen_con_amarillo = paint(imagen,imagen_amarillo,color_amarillo)
    view_image("Parte amarilla detectada", imagen_con_amarillo)
    #QUITAR AMARILLO DE ORIGINAL

    result = add_color_pixels(imagen_with_gaussian,imagen_con_verde,color_verde)   
    result = add_color_pixels(result,imagen_con_blanco,color_blanco)
    result = add_color_pixels(result,imagen_con_amarillo,color_amarillo)
          
    return result
    
#---------------------------------------------------POSTPROCESAMIENTO----------------------------------------------------#


def calculate_color_percentage(image,colors):
    color_percentages = []
    total_pixels = image.shape[0] * image.shape[1]  # Total de píxeles en la imagen
    
    for color in colors:
        color_bgr = np.array([color], dtype=np.uint8)
        mask_color = np.all(image == color_bgr, axis=-1)
        num_pixels_color = np.sum(mask_color)
        percentage_color = (num_pixels_color / total_pixels) * 100
        color_percentages.append(percentage_color)
    
    return color_percentages

def plot_color_percentage_bars(colors, percentages):
    plt.figure(figsize=(8, 6), facecolor='lightgrey')
    colors_rgb = [(color[2]/255, color[1]/255, color[0]/255) for color in colors]  # Convertir a formato RGB
    labels = ["Hongo Aspergillus\no región inmadura", "Hongo Penicillium", "Limón apto"]
    plt.pie(percentages, labels=labels,
            colors=colors_rgb, autopct='%1.1f%%', startangle=140)
    plt.title('Estado del limón')
    plt.show()

def postprocess(imagen):
    colors = [np.array([50, 200, 50]),np.array([255, 255, 255]),np.array([51, 255, 255])]
    porcentajes = calculate_color_percentage(imagen,colors)
    plot_color_percentage_bars(colors,porcentajes)
    

#---------------------------------------------------VISUALIZACION----------------------------------------------------#

def view_image(title,imagen):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 600)  
    cv2.imshow(title, imagen) 
