import cv2
import matplotlib.pyplot as plt
import numpy as np
import process as p 

#CARGA DE IMAGEN
nombre_archivo = r'.\\images\\limones1.jpg'
imagen = cv2.imread(nombre_archivo)

# Comprueba si la imagen se cargó correctamente
# Verificar si la imagen se cargó correctamente
if imagen is not None:
    
    p.view_image("Resultado: ",imagen)
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar el operador Sobel para la detección de bordes en dirección X (horizontal)
    sobelx = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)  # Sobel en dirección X

    # Aplicar el operador Sobel para la detección de bordes en dirección Y (vertical)
    sobely = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)  # Sobel en dirección Y

    # Calcular la magnitud del gradiente combinando bordes en X e Y
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))

    # Escalar la magnitud del gradiente a un rango de 0 a 255
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    # Crear una máscara basada en la magnitud del gradiente (valores mayores que 0)
    _, mask = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY)

    # Convertir la máscara a tipo uint8 (asegurarse de que sea del mismo tipo que la imagen original)
    mask = mask.astype(np.uint8)

    # Aplicar la máscara a la imagen original para segmentar los píxeles de interés
    segmented_image = cv2.bitwise_and(imagen, imagen, mask=mask)
    
    img = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Definir el kernel para la operación de erosión
    kernel = np.ones((5, 5), np.uint8)  # Puedes ajustar el tamaño del kernel según sea necesario

    # Aplicar la operación de erosión
    img_erosion = cv2.erode(magnitude, kernel, iterations=1)


    p.view_image("Resultado: ",img_erosion)

else:
    print(f'Error: No se pudo abrir la imagen "{nombre_archivo}"')
