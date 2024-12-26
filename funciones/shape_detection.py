import cv2
import numpy as np

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
        4: "square/rectangle",  # Se puede afinar más adelante
        5: "pentagon",
        6: "hexagon",
        7: "heptagon",
        8: "octagon",
        9: "nonagon"
    }

    detected_shapes = []

    for contour in contours:
        # Filtrar contornos muy pequeños o grandes
        area = cv2.contourArea(contour)
        if area < 500 or area > 0.9 * gray.shape[0] * gray.shape[1]:
            continue

        # Aproximar el contorno para simplificarlo
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Obtener el número de vértices
        vertices = len(approx)

        # Clasificar figura usando el diccionario
        shape_name = vertex_to_shape.get(vertices, "polygon")
        detected_shapes.append(shape_name)

    # Retornar la lista de figuras detectadas y el resultado
    if detected_shapes:
        return detected_shapes, ", ".join(detected_shapes)
    else:
        return [], "Figura no detectada"
