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
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    # Definir el rango de color para los limones en HSV
    lower_lemon = np.array([0, 10, 10])  # Umbral inferior para H, S y V
    upper_lemon = np.array([252, 255, 255])  # Umbral superior para H, S y V
    # Crear una máscara para los limones
    mask = cv2.inRange(hsv, lower_lemon, upper_lemon)
    #view_image("Mascara binaria inicial", mask)
    # Invertir la máscara
    mask = cv2.bitwise_not(mask)
    #view_image("Mascara binaria invertida", mask)
    #seg_binaria_parcial = cv2.bitwise_and(imagen, imagen, mask=mask) # Esta operacion de hacer una composicion de la imagen original con sí misma a partir de la máscara, no se hace ya que no tiene sentido en este caso recuperar detalles extra del fondo de la imagen como marcas de agua
    #view_image("Segmentacion binaria inicial", seg_binaria_parcial)
    # Definir el kernel para la operación de erosión
    kernel = np.ones((15, 15), np.uint8)
    # Aplicar la operación de erosión a la máscara binaria obtenida
    eroded_image = cv2.erode(mask, kernel, iterations=3)
    # Definir el kernel para la operación de dilatación
    kernel = np.ones((18, 18), np.uint8)
    # Aplicar la operación de dilatación a la imagen erosionada
    seg_binaria = cv2.dilate(eroded_image, kernel, iterations=3)
    #view_image("Segmentacion binaria luego aplicar operaciones morfologicas", seg_binaria)
    seg_binaria = cv2.resize(seg_binaria, (imagen.shape[1], imagen.shape[0]))
    # Se crea una imagen completamente negra del mismo tamaño que la original, y se reemplazan los píxeles donde va el limón por los de la original usando la segmentación binaria
    imagen_segmentada = np.zeros_like(imagen)
    imagen_segmentada[seg_binaria == 0] = imagen[seg_binaria == 0]
    #view_image("Segmentacion final del limon con fondo negro", imagen_segmentada)
    return imagen_segmentada

def color_detect(imagen,lower_color, upper_color):
    mask_color = cv2.inRange(imagen, lower_color, upper_color)
    # Aplicar la máscara 
    color_detected = cv2.bitwise_and(imagen, imagen, mask=mask_color)
    return color_detected

def delete_color(imagen, color):
    # Este metodo retorna la imagen sin el color especificado
    mask_umbral = (color > 0).all(axis=2)  # True donde el color es el indicado por parametro

    # Invertir la máscara para obtener píxeles que no son de ese color
    mask_no_color = np.logical_not(mask_umbral)

    # Convertir la máscara a tipo uint8 para usar con bitwise_and
    mask_no_color = mask_no_color.astype(np.uint8) * 255

    # Aplicar la máscara para eliminar píxeles de ese color de la imagen
    imagen_sin_color = cv2.bitwise_and(imagen, imagen, mask=mask_no_color)

    return imagen_sin_color

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
    
    #APLICA FILTRO GAUSSIANO (Disminuir ruido y eliminar detalles finos)
    imagen_with_gaussian = cv2.GaussianBlur(imagen, (7, 7), 2)
    
    # Convertir la imagen a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen_with_gaussian, cv2.COLOR_BGR2HSV)
    #view_image('Imagen del limon segmentado de RGB a HSV', imagen_hsv)
    
    #APLICAR FILTRO DE DETECCION VERDE

    # Ajustar el rango para el filtro de detección de verde en HSV
    lower_verde = np.array([30, 18, 0])
    upper_verde = np.array([180, 255, 255])
    imagen_verde = color_detect(imagen_hsv,lower_verde,upper_verde)
    color_verde = np.array([50, 200, 50])
    # Resaltar regiones de moho verde o azulado o zonas inmaduras en la original gris
    imagen_con_verde = paint(imagen,imagen_verde,color_verde)
    view_image("Region detectada de hongo Aspergillus o inmadura", imagen_con_verde)
    # Eliminar el verde detectado para continuar detectando las otras regiones sin tener en cuenta los píxeles que ya se consideraron para este caso
    imagen_sin_verde = delete_color(imagen_hsv,imagen_verde)
    #view_image("Imagen con extraccion de verde detectado", imagen_sin_verde)
    
    #APLICAR FILTRO DE DETECCION BLANCO

    lower_white = np.array([0, 0, 20])
    upper_white = np.array([255, 105, 255])
    imagen_blanco = color_detect(imagen_sin_verde,lower_white,upper_white)
    color_blanco = np.array([255, 255, 255])
    imagen_con_blanco = paint(imagen,imagen_blanco,color_blanco)
    view_image("Region detectada de hongo Penicillium", imagen_con_blanco)
    imagen_sin_blanco = delete_color(imagen_sin_verde,imagen_blanco)
    #view_image("Imagen con extraccion de blanco detectado", imagen_sin_blanco)
    
    #APLICAR FILTRO DE DETECCION AMARILLO

    lower_yellow = np.array([0, 106, 0])
    upper_yellow = np.array([29, 255, 255])
    imagen_amarillo= color_detect(imagen_sin_blanco,lower_yellow,upper_yellow)
    color_amarillo = np.array([51, 255, 255])
    imagen_con_amarillo = paint(imagen,imagen_amarillo,color_amarillo)
    view_image("Region detectada de limon sano", imagen_con_amarillo)

    #SE SEGMENTA INTERNAMENTE EL LIMON A PARTIR DE LAS IMAGENES BGR QUE SE OBTUVIERON CON CADA UNA DE LAS ZONAS DE INTERES DETECTADAS INDIVIDUALMENTE

    result = add_color_pixels(imagen_with_gaussian,imagen_con_verde,color_verde)
    #view_image("Segmentacion interna del limon a partir de la primera region detectada", result)
    result = add_color_pixels(result,imagen_con_blanco,color_blanco)
    #view_image("Segmentacion interna del limon a partir de la primera y segunda region detectada", result)
    result = add_color_pixels(result,imagen_con_amarillo,color_amarillo)
    #view_image("Segmentacion interna compoleta del limon, a partir de las tres regiones detectadas", result)
    # Se retorna una imagen en formato BGR donde se unifican las segmentaciones parciales de cada region interna del limon en una sola
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
    # La imagen se redimensiona con un largo de 600 y un ancho de 800 aprovechando la carcateristica de que en general los limones tienen un mayor ancho que alto 
    cv2.resizeWindow(title, 800, 600)
    cv2.imshow(title, imagen) 
