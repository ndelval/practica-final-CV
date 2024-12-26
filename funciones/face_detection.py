import cv2
import os
import numpy as np
from funciones.utils import *


def detect_and_crop_faces(image):
    """
    Detecta y recorta las caras en una imagen.
    Args:
        image: Imagen en escala de grises
    Returns:
        cropped_faces: Lista de caras recortadas
    """
    # Cargar el clasificador pre-entrenado de OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detectar las caras
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_faces.append(image[y:y+h, x:x+w])
    
    return cropped_faces, faces


def save_cropped_faces(person_path, person_clean_path, person_name):
    """
    Detecta y guarda las caras recortadas de una persona.
    Args:
        person_path: Ruta del directorio con las imágenes originales
        person_clean_path: Ruta del directorio para guardar las caras recortadas
        person_name: Nombre de la persona
    """
    for i, image_name in enumerate(os.listdir(person_path)):
        image_path = os.path.join(person_path, image_name)
        face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if face_img is not None:
            cropped_faces, _ = detect_and_crop_faces(face_img)
            if cropped_faces:
                cropped_face = cropped_faces[0]
                clean_image_path = os.path.join(person_clean_path, f"{person_name}_face{i}.jpg")
                cv2.imwrite(clean_image_path, cropped_face)
                print(f"Cara recortada y guardada en: {clean_image_path}")


def prepare_clean_train_folder(base_path='./train', clean_path='./train_limpio'):
    """
    Prepara el directorio de entrenamiento limpio.
    Args:
        base_path: Directorio con las imágenes originales
        clean_path: Directorio para guardar las imágenes de caras recortadas
    """
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        person_clean_path = os.path.join(clean_path, person_name)
        os.makedirs(person_clean_path, exist_ok=True)

        if os.path.isdir(person_path):
            save_cropped_faces(person_path, person_clean_path, person_name)


def train_model():
    """
    Entrena el modelo LBPH para reconocimiento facial.
    """
    labels, faces = prepare_training_data()
    if len(faces) == 0 or len(labels) == 0:
        print("No se encontraron imágenes o etiquetas. Verifica la estructura del directorio.")
        return

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, labels)
    face_recognizer.save('modelo_lbphface.xml')
    print("Modelo entrenado y guardado como 'modelo_lbphface.xml'.")


def predict_faces(cropped_faces, face_recognizer):
    """
    Predice si cada cara es conocida.
    Args:
        cropped_faces: Lista de caras recortadas
        face_recognizer: Modelo entrenado de reconocimiento facial
    Returns:
        predictions: Lista de predicciones de cada cara
    """
    predictions = []
    for cropped_face in cropped_faces:
        label, confidence = face_recognizer.predict(cropped_face)
        predictions.append((label, confidence))
    return predictions
