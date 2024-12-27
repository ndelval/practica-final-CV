import cv2
import time
import glob
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from funciones.utils import *
from funciones.face_detection import *
from funciones.calibration_camara import (
    load_images, detect_corners, refine_corners, generate_chessboard_points,
    calibrate_camera, undistort_image
)
from funciones.shape_detection import detect_shapes_grayscale
from funciones.motion_detection import (
    configure_background_subtractor, configure_kalman_filter, detect_moving_objects
)

def calibrate_camera_system():
    """
    Calibra la cámara utilizando imágenes de un tablero de ajedrez.
    Returns:
        intrinsics: Matriz intrínseca de la cámara
        dist_coeffs: Coeficientes de distorsión de la cámara
    """
    pattern_size = (8, 6)  # Filas y columnas del tablero de ajedrez
    square_size = 1.0  # Tamaño de los cuadrados en unidades arbitrarias
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    image_paths = glob.glob('/Users/ndelvalalvarez/Downloads/TERECERO/Computer_VIsion/ProyectoFinal/right/*.jpg')
    images = load_images(image_paths)
    valid_images, corners = detect_corners(images, pattern_size)
    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in valid_images]
    corners_refined = refine_corners(images_gray, corners, criteria)
    chessboard_points = [generate_chessboard_points(pattern_size, square_size) for _ in valid_images]
    image_size = images_gray[0].shape[::-1]

    rms, intrinsics, dist_coeffs, _ = calibrate_camera(
        chessboard_points, [cor[1] for cor in corners if cor[0]], image_size
    )
    print(f"Error RMS: {rms}")
    print("Matriz intrínseca:", intrinsics)
    print("Coeficientes de distorsión:", dist_coeffs)

    return intrinsics, dist_coeffs

def initialize_camera():
    """
    Inicializa la cámara con parámetros optimizados de resolución y FPS.
    Returns:
        cap: Objeto VideoCapture configurado
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Cámara configurada: {width}x{height} a {fps} FPS")

    return cap

def detect_shapes_during_interval(cap, duration=1):
    """
    Detecta figuras durante un intervalo de tiempo especificado.
    Args:
        cap: Objeto VideoCapture para capturar frames.
        duration: Duración en segundos para detectar figuras.
    """
    start_time = time.time()
    detected_shapes = []

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shapes, _ = detect_shapes_grayscale(gray_frame)
        detected_shapes.extend(shapes)

    if detected_shapes:
        most_common_shape = Counter(detected_shapes).most_common(1)[0]
        print(f"Figura más detectada: {most_common_shape[0]} con {most_common_shape[1]} detecciones.")
    else:
        print("No se detectaron figuras durante el intervalo.")

def process_faces_and_motion(frame, gray_frame, mog2, kf, measurement, prediction, face_cascade, face_recognizer, names, executor, last_faces_predictions, last_face_coords, last_motion_frame, frame_count):
    # Procesar rostros
    if frame_count % 2 == 0:
        cropped_faces, face_coords = detect_and_crop_faces(gray_frame, face_cascade)
        future_faces = executor.submit(predict_faces, cropped_faces, face_recognizer)
        last_faces_predictions = future_faces.result()
        last_face_coords = face_coords

    # Dibujar rostros
    for (x, y, w, h), (label, confidence) in zip(last_face_coords, last_faces_predictions):
        color = (255, 0, 0) if label in names and confidence < 95.0 else (0, 0, 255)
        text = f"{names.get(label, 'Unknown')} ({confidence:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Procesar movimiento
    if frame_count % 3 != 0:
        future_motion = executor.submit(detect_moving_objects, frame, mog2, kf, measurement, prediction, frame_count)
        last_motion_frame = future_motion.result()
    
    if last_motion_frame is not None:
        frame = last_motion_frame

    return frame, last_faces_predictions, last_face_coords, last_motion_frame

def process_video(intrinsics, dist_coeffs, calibration=False):
    preprocess = input("Is it necessary to include new images in the model? [Y/N]: ")
    if preprocess.upper() == "Y":
        prepare_clean_train_folder()
        train_model()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('/Users/ndelvalalvarez/Downloads/TERECERO/Computer_VIsion/ProyectoFinal/modelos/modelo_lbphface.xml')

    names = {0: "Nico", 1: "Kike"}

    cap = initialize_camera()
    mog2 = configure_background_subtractor()
    kf, measurement, prediction = configure_kalman_filter()

    print("Procesando video... Presiona 'q' para salir. Presiona 'Enter' para detectar figuras durante 1 segundo.")

    frame_count = 0
    last_faces_predictions, last_face_coords, last_motion_frame = [], [], None

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('f'):  # Spacebar
                print("Detectando figuras...")
                detect_shapes_during_interval(cap)
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame, last_faces_predictions, last_face_coords, last_motion_frame = process_faces_and_motion(
                    frame, gray_frame, mog2, kf, measurement, prediction, face_cascade, face_recognizer, names, executor,
                    last_faces_predictions, last_face_coords, last_motion_frame, frame_count
                )

            cv2.imshow('Procesamiento de Video', frame)

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main(calibrate):
    intrinsics, dist_coeffs = calibrate_camera_system() if calibrate else (None, None)
    process_video(intrinsics, dist_coeffs, calibration=calibrate)

if __name__ == '__main__':
    calibrate = '--calibrate' in sys.argv
    main(calibrate)
