import cv2
import time
import glob
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
from concurrent.futures import ThreadPoolExecutor


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

    # Cargar imágenes para calibración
    image_paths = glob.glob('/Users/ndelvalalvarez/Downloads/TERECERO/Computer_VIsion/ProyectoFinal/right/*.jpg')  # Cambia a la ruta correcta
    images = load_images(image_paths)
    valid_images, corners = detect_corners(images, pattern_size)
    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in valid_images]
    corners_refined = refine_corners(images_gray, corners, criteria)
    chessboard_points = [generate_chessboard_points(pattern_size, square_size) for _ in valid_images]
    image_size = images_gray[0].shape[::-1]  # (ancho, alto)

    # Calibrar cámara
    rms, intrinsics, dist_coeffs, extrinsics = calibrate_camera(
        chessboard_points, [cor[1] for cor in corners if cor[0]], image_size
    )
    print(f"Error RMS: {rms}")
    print("Matriz intrínseca:", intrinsics)
    print("Coeficientes de distorsión:", dist_coeffs)

    return intrinsics, dist_coeffs


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara")
        exit()
    return cap

def process_video(intrinsics, dist_coeffs, calibration=False):
    preprocess = input("Is it necessary to include new images in the model? [Y/N]: ")
    if preprocess.upper() == "Y":
        prepare_clean_train_folder()
        train_model()

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('/Users/ndelvalalvarez/Downloads/TERECERO/Computer_VIsion/ProyectoFinal/modelos/modelo_lbphface.xml')  # Ruta del modelo

    names = {0: "Nico", 1: "Kike"}

    cap = initialize_camera()
    mog2 = configure_background_subtractor()
    kf, measurement, prediction = configure_kalman_filter()

    print("Procesando video... Presiona 'q' para salir.")

    last_check_time = time.time()
    frame_count = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_shapes = None
        ret, frame = cap.read()
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (frame_width, frame_height)) 
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break

            if calibration:
                frame = undistort_image(frame, intrinsics, dist_coeffs)

            # Detección de figuras geométricas
            current_time = time.time()
            if current_time - last_check_time > 5:
                future_shapes = executor.submit(detect_shapes_grayscale, gray_frame)
                last_check_time = current_time

            # Detección y predicción de rostros
            cropped_faces, face_coords = detect_and_crop_faces(gray_frame)
            future_faces = executor.submit(predict_faces, cropped_faces, face_recognizer)

            # Detección de objetos en movimiento
            future_motion = executor.submit(detect_moving_objects, frame, mog2, kf, measurement, prediction, frame_count)

            # Esperar los resultados
            if future_shapes is not None:  # Solo intentar acceder a `future_shapes` si fue definida
                detected_shapes, result_text = future_shapes.result()
                print(result_text)
            predictions = future_faces.result()
            frame = future_motion.result()

            # Detección de caras
            for (x, y, w, h), (label, confidence) in zip(face_coords, predictions):
                if label in names and confidence < 100.0:
                    color = (0, 255, 0)  # Rectángulo verde para caras conocidas
                    text = f"{names[label]} ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Rectángulo rojo para caras desconocidas
                    text = f"Unknown ({confidence:.2f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame_count += 1
            cv2.imshow('Procesamiento en Tiempo Real', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    intrinsics, dist_coeffs = calibrate_camera_system()
    process_video(intrinsics, dist_coeffs)

if __name__ == '__main__':
    main()
