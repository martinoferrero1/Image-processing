import cv2
from utils import preprocess as pre , process, postprocess as post, view_image as vi
import os

# Directorio donde se encuentran las imágenes
directorio_imagenes = 'images'

# Función para procesar una imagen
def procesar_imagen(imagen):
    preprocesada = pre(imagen)
    procesada = process(preprocesada)
    return procesada

# Obtener la lista de nombres de archivos en el directorio de imágenes
nombres_archivos = os.listdir(directorio_imagenes)

# Construir la ruta completa de la imagen
path_img = os.path.join(directorio_imagenes, "limones6.jpg")

# Leer la imagen usando OpenCV
imagen = cv2.imread(path_img)

# Verificar si la lectura fue exitosa (imagen no es None)
if imagen is not None:
    # Mostrar la imagen original
    vi("Original", imagen)

    # Procesar la imagen
    procesada = procesar_imagen(imagen)

    # Mostrar la imagen procesada
    vi("Procesada", procesada)
    post(procesada)
    
    # Esperar hasta que se presione una tecla
    cv2.waitKey(0)

    # Cerrar todas las ventanas al finalizar
cv2.destroyAllWindows()