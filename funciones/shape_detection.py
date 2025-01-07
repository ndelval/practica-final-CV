import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def detect_shapes_grayscale(gray):
    """
    Detecta figuras geométricas en una imagen en escala de grises sin importar el color.

    Args:
        image (numpy.ndarray): Imagen a analizar.

    Returns:
        list: Lista de figuras detectadas (triángulo, cuadrado, círculo, etc.).
        str: Resultado textual ("Figura no detectada" o el nombre de la figura).
    """

    # Aplicar suavizado para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detectar bordes con Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Diccionario de mapeo de vértices a nombres de figuras
    vertex_to_shape = {
        3: "triangle",
        4: "square/rectangle",  
        5: "pentagon",
        6: "hexagon",
        7: "heptagon",
        8: "octagon",
        9: "nonagon"
    }

    def process_contour(contour):
        area = cv2.contourArea(contour)
        if area < 500 or area > 0.9 * gray.shape[0] * gray.shape[1]:
            return None

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        vertices = len(approx)

        return vertex_to_shape.get(vertices, "polygon")

    detected_shapes = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_contour, contour) for contour in contours]
        for future in futures:
            shape_name = future.result()
            if shape_name:
                detected_shapes.append(shape_name)

    if detected_shapes:
        return detected_shapes, ", ".join(detected_shapes)
    else:
        return [], "Figura no detectada"