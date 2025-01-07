import cv2
import os
import numpy as np
import shutil

# Inicialización de clasificadores para detección de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(image):
    """
    Detecta caras en una imagen.
    Args:
        image: Imagen en escala de grises
    Returns:
        faces: Lista de coordenadas (x, y, w, h) de las caras detectadas
    """
    faces = face_cascade.detectMultiScale(image, 
                                          scaleFactor=1.2, 
                                          minNeighbors=5, 
                                          minSize=(30, 30), 
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def crop_faces(image, faces):
    """
    Recorta las caras detectadas en una imagen.
    Args:
        image: Imagen en escala de grises
        faces: Lista de coordenadas (x, y, w, h) de las caras detectadas
    Returns:
        cropped_faces: Lista de caras recortadas
    """
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = cv2.resize(image[y:y+h, x:x+w], (100, 100))
        cropped_faces.append(cropped_face)
    return cropped_faces

def detect_and_crop_faces(image):
    """
    Detecta y recorta múltiples caras en la imagen.
    Args:
        image: Imagen en escala de grises
    Returns:
        cropped_faces: Lista de caras recortadas
        face_coords: Lista de coordenadas (x, y, w, h) de las caras recortadas
    """
    faces = detect_faces(image)
    cropped_faces = crop_faces(image, faces)
    return cropped_faces, faces

def clean_directory(path):
    """
    Elimina el contenido de un directorio y lo vuelve a crear.
    Args:
        path: Ruta del directorio a limpiar
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

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
    clean_directory(clean_path)
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        person_clean_path = os.path.join(clean_path, person_name)
        os.makedirs(person_clean_path, exist_ok=True)

        if os.path.isdir(person_path):
            save_cropped_faces(person_path, person_clean_path, person_name)

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

