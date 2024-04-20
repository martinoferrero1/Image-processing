import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    # imagen_rgb_gris[mask_binaria > 0] = imagen_moho[mask_binaria > 0]
    coordenadas = np.argwhere(mask_binaria > 0)

# Iterar sobre cada posición (fila, columna) donde mask_binaria > 0
    for coordenada in coordenadas:
        fila, columna = coordenada[:2]  # Obtener las primeras dos posiciones (fila y columna)
        imagen_rgb_gris[fila,columna] = color
    return imagen_rgb_gris

def view_image(title,imagen):
    if imagen is None or not isinstance(imagen, np.ndarray):
        print(f"Error: La imagen '{title}' no es válida.")
        return
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Permite redimensionar la ventana
    cv2.resizeWindow(title, 600, 400)  
    cv2.imshow(title, imagen) 
    
def add_color_pixels(image_dest, image_source, color_to_add):
    # Convertir el color deseado a formato BGR (sin transparencia)
    color_to_add_bgr = np.array([color_to_add], dtype=np.uint8)
    
    # Encontrar los píxeles en la imagen de origen que coinciden con el color deseado
    mask_color = np.all(image_source == color_to_add_bgr, axis=-1)
    
    # Copiar los píxeles del color deseado desde la imagen de origen a la imagen de destino
    image_dest[mask_color] = color_to_add
    
    return image_dest

def process(imagen):

    # Comprueba si la imagen se cargó correctamente
    if imagen is not None:
        
        #view_image('Imagen original',imagen)
        
        #APLICA FILTRO GAUSSIANO (Disminuir ruido y eliminar detalles finos)
        imagen_with_gaussian = cv2.GaussianBlur(imagen, (7, 7), 2)
        #view_image('Gaussiano',imagen_with_gaussian)
        
        #CAMBIA FORMATO BGR A HSV (Resalta color amarillo, lo hace homogeneo)
        imagen_hsv = cv2.cvtColor(imagen_with_gaussian, cv2.COLOR_BGR2HSV)
        #view_image('RGB A HSV',imagen_hsv)
        
        #APLICAR FILTRO DE DETECCION BLANCO
        lower_white = np.array([0, 0, 200])        # Umbral inferior en HSV (para blanco)
        upper_white = np.array([180, 30, 255])      # Umbral superior en HSV (para blanco)  
        imagen_blanco = color_detect(imagen_hsv,lower_white,upper_white)
        #view_image('FILTRO AMARILLO',imagen_amarillo)
        color_blanco = np.array([255, 255, 255])
        imagen_con_blanco = paint(imagen,imagen_blanco,color_blanco)
        imagen_sin_blanco = delete_color(imagen_hsv,imagen_blanco)
        
        #APLICAR FILTRO DE DETECCION AMARILLO
            # Crear una máscara 
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([33, 255, 255])
        imagen_amarillo= color_detect(imagen_sin_blanco,lower_yellow,upper_yellow)
        #view_image('FILTRO AMARILLO',imagen_amarillo)
        color_amarillo = np.array([51, 255, 255])
        imagen_con_amarillo = paint(imagen,imagen_amarillo,color_amarillo)
        #QUITAR AMARILLO DE ORIGINAL
        imagen_sin_amarillo = delete_color(imagen_sin_blanco,imagen_amarillo)
        
        #APLICAR FILTRO DE VERDE
        lower_verde = np.array([25, 20, 20])   # Umbral inferior en HSV
        upper_verde = np.array([80, 255, 255])  # Umbral superior en HSV
        imagen_verde = color_detect(imagen_sin_amarillo,lower_verde,upper_verde)
        #view_image('FILTRO MOHO',imagen_moho)
        color_verde = np.array([50, 200, 50])
        #RESALTAR MOHO EN LA ORIGINAL GRIS
        imagen_con_verde = paint(imagen,imagen_verde,color_verde)
        
        result = add_color_pixels(imagen_with_gaussian,imagen_con_blanco,color_blanco)
        result = add_color_pixels(result,imagen_con_amarillo,color_amarillo)
        result = add_color_pixels(result,imagen_con_verde,color_verde)
        
        colors = [color_verde,color_blanco,color_amarillo]
        porcentajes = calculate_color_percentage(result,colors)
        plot_color_percentage_bars(colors,porcentajes)                
        return result
    
def calculate_color_percentage(image, colors):
    """
    Calcula el porcentaje de píxeles en la imagen que coinciden con cada color dado.
    
    Parameters:
        image (numpy.ndarray): Imagen de entrada en formato BGR.
        colors (list): Lista de tres colores en formato (B, G, R) a detectar.
    
    Returns:
        list: Lista de porcentajes correspondientes a la presencia de cada color en la imagen.
    """
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
    """
    Muestra un gráfico de barras con los porcentajes de los colores detectados.
    
    Parameters:
        colors (list): Lista de colores en formato (B, G, R) para etiquetar las barras.
        percentages (list): Lista de porcentajes correspondientes a cada color.
    """
    # plt.figure(figsize=(8, 6), facecolor='lightgrey')
    # colors_rgb = [(color[2]/255, color[1]/255, color[0]/255) for color in colors]  # Convertir a formato RGB
    
    # plt.bar(range(len(colors)), percentages, color=colors_rgb)
    # plt.xticks(range(len(colors)), ['Color {}'.format(i+1) for i in range(len(colors))])
    # plt.xlabel('Colores')
    # plt.ylabel('Porcentaje (%)')
    # plt.title('Porcentaje de colores en la imagen')
    # plt.gca().set_facecolor('lightgrey')
    # plt.ylim(0, 100)  # Establecer el rango del eje y de 0% a 100%
    # plt.show()
    plt.figure(figsize=(8, 6), facecolor='lightgrey')
    
    colors_rgb = [(color[2]/255, color[1]/255, color[0]/255) for color in colors]  # Convertir a formato RGB
    
    plt.pie(percentages, labels=['Color {}'.format(i+1) for i in range(len(colors))],
            colors=colors_rgb, autopct='%1.1f%%', startangle=140)
    
    plt.title('Porcentaje de colores en la imagen')
    
    plt.show()