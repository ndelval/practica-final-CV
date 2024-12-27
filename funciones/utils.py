import os
import shutil
import numpy as np
import cv2


def clean_directory(path):
    """
    Elimina el contenido de un directorio y lo vuelve a crear.
    Args:
        path: Ruta del directorio a limpiar
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_images_from_directory(directory):
    """
    Carga imágenes en escala de grises desde un directorio.
    Args:
        directory: Ruta del directorio
    Returns:
        images: Lista de imágenes en escala de grises
        labels: Lista de etiquetas de las imágenes
    """
    images = []
    labels = []
    for label, person_name in enumerate(os.listdir(directory)):
        person_path = os.path.join(directory, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if face_img is not None:
                    images.append(face_img)
                    labels.append(label)
    return np.array(labels), images


def prepare_training_data(clean_path='./train_limpio'):
    """
    Prepara los datos de entrenamiento.
    Args:
        clean_path: Directorio con las imágenes de caras recortadas
    Returns:
        labels: Lista de etiquetas de las caras
        faces: Lista de caras recortadas
    """
    return load_images_from_directory(clean_path)

