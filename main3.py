import cv2
import time
import glob
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from picamera2 import Picamera2

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

SHAPE_ORDER = ["square/rectangle", "triangle", "hexagon"]
PATTERN_TIME_LIMIT = 30
ALARM_ACTIVE = True

def calibrate_camera_system():
    '''
    Calibra la cámara utilizando un tablero de ajedrez con un patrón de 9x6 esquinas.
    Carga las imágenes de la carpeta 'Calibracion/*.jpg' y realiza la calibración
    si se detectan esquinas válidas.

    Returns:
        intrinsics (np.array): Matriz intrínseca de la cámara
        dist_coeffs (np.array): Coeficientes de distorsión de la cámara
    '''
    pattern_size = (9, 6)
    square_size = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    image_paths = glob.glob('Calibracion/*.jpg')
    images = load_images(image_paths)
    valid_images, corners = detect_corners(images, pattern_size)

    if not valid_images:
        print("No se encontraron imágenes válidas para la calibración. Revisa la carpeta Calibracion/*.jpg.")
        return None, None

    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in valid_images]
    corners_refined = refine_corners(images_gray, corners, criteria)

    if not any(cor[0] for cor in corners):
        print("No se detectaron esquinas de tablero en ninguna imagen. Verifica las fotos del tablero.")
        return None, None

    chessboard_points = [generate_chessboard_points(pattern_size, square_size) for _ in valid_images]
    image_size = images_gray[0].shape[::-1]

    rms, intrinsics, dist_coeffs, _ = calibrate_camera(
        chessboard_points,
        [cor[1] for cor in corners if cor[0]],
        image_size
    )

    print(f"Error RMS: {rms}")
    print("Matriz intrínseca:", intrinsics)
    print("Coeficientes de distorsión:", dist_coeffs)

    return intrinsics, dist_coeffs

def initialize_camera():
    '''
    Inicializa la cámara usando Picamera2 con una resolución 320x240
    en formato RGB888, y la pone en marcha.

    Returns:
        picam2 (Picamera2): Objeto de cámara ya configurado.
    '''
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={
            "size": (320, 240),
            "format": "RGB888"
        }
    )
    picam2.configure(config)
    picam2.start()
    return picam2

def detect_shapes_once(frame_bgr):
    '''
    Detecta la forma más común en un frame BGR determinado, llamando a
    detect_shapes_grayscale. Devuelve un string con el nombre de la figura
    más común, o None si no se detectan figuras.

    Args:
        frame_bgr (np.array): Imagen en formato BGR.

    Returns:
        (str o None): 'square/rectangle', 'triangle', 'hexagon' u otra figura, o None.
    '''
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    shapes, _ = detect_shapes_grayscale(gray_frame)
    if not shapes:
        return None
    most_common_shape = Counter(shapes).most_common(1)[0][0]
    return most_common_shape

def process_pattern(picam2):
    '''
    Gestiona la detección secuencial de 3 figuras en un tiempo máximo de 30 seg:
    - 1er F: square/rectangle
    - 2do F: triangle
    - 3er F: hexagon

    Se acumulan las figuras detectadas cada frame en un buffer.
    Cuando el buffer llega a 10 frames, se decide la figura mayoritaria
    y se compara con la esperada.

    Muestra una ventana 'Deteccion de Patron' durante el proceso.
    Retorna True si se completa con éxito; False si se agota el tiempo
    o no se detectan las figuras en el orden correcto.

    Args:
        picam2 (Picamera2): Objeto de cámara para capturar frames.

    Returns:
        (bool): True si el patrón se completó, False en caso contrario.
    '''
    print("Iniciando detección de patrón geométrico (30 seg). Debe seguir el orden: cuadrado → triangulo → hexagono.")

    start_time = time.time()
    shape_index = 0
    frames_buffer = []
    FRAMES_PER_DETECTION = 10

    try:
        while time.time() - start_time < PATTERN_TIME_LIMIT:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            detected_shape = detect_shapes_once(frame_bgr)

            if detected_shape is not None:
                frames_buffer.append(detected_shape)

            if len(frames_buffer) >= FRAMES_PER_DETECTION:
                shape_counts = Counter(frames_buffer)
                major_shape = shape_counts.most_common(1)[0][0]
                frames_buffer.clear()

                expected_shape = SHAPE_ORDER[shape_index]
                if major_shape.lower() == expected_shape.lower():
                    print(f"Encontrado {major_shape}. Próxima figura...")
                    shape_index += 1
                    if shape_index >= len(SHAPE_ORDER):
                        print("¡Patrón completo correctamente!")
                        return True
                    time.sleep(1)

            remaining_time = PATTERN_TIME_LIMIT - int(time.time() - start_time)
            cv2.putText(
                frame_bgr,
                f"Patron: {SHAPE_ORDER[shape_index]}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.putText(
                frame_bgr,
                f"Tiempo restante: {remaining_time}s",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.imshow('Deteccion de Patron', frame_bgr)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        print("Patrón fallido: No se completó a tiempo o no era el orden correcto.")
        return False
    finally:
        cv2.destroyWindow('Deteccion de Patron')

def process_faces_and_motion(
    frame, gray_frame, mog2, kf, measurement, prediction,
    face_cascade, face_recognizer, names, executor,
    last_faces_predictions, last_face_coords, last_motion_frame,
    frame_count
):
    '''
    Aplica detección de rostros y movimiento en cada frame.
    - Detección de rostros cada 2 fotogramas.
    - Detección de movimiento cada fotograma (excepto cuando frame_count % 3 == 0).

    Args:
        frame (np.array): Frame actual en BGR.
        gray_frame (np.array): Frame en escala de grises.
        mog2: Objeto de sustracción de fondo.
        kf: Filtro de Kalman.
        measurement, prediction: Arrays usados en Kalman.
        face_cascade (cv2.CascadeClassifier): Clasificador en cascada para rostros.
        face_recognizer: Modelo LBPH para reconocimiento facial.
        names (dict): Diccionario {label: nombre}.
        executor (ThreadPoolExecutor): Pool de hilos para tareas concurrentes.
        last_faces_predictions (list): Últimas predicciones de rostros.
        last_face_coords (list): Últimas coordenadas de rostros.
        last_motion_frame (np.array): Último frame de movimiento dibujado.
        frame_count (int): Contador de frames.

    Returns:
        (tuple): (frame, last_faces_predictions, last_face_coords, last_motion_frame)
            - frame: Frame resultante con rostros dibujados y/o movimiento.
            - last_faces_predictions: Lista de predicciones de rostros actualizada.
            - last_face_coords: Lista de coords de rostros actualizada.
            - last_motion_frame: Último frame de movimiento dibujado.
    '''
    if frame_count % 2 == 0:
        cropped_faces, face_coords = detect_and_crop_faces(gray_frame, face_cascade)
        future_faces = executor.submit(predict_faces, cropped_faces, face_recognizer)
        last_faces_predictions = future_faces.result()
        last_face_coords = face_coords

    for (x, y, w, h), (label, confidence) in zip(last_face_coords, last_faces_predictions):
        color = (255, 0, 0)
        text_label = names.get(label, 'Unknown')        
        if confidence > 80:
            text_label = 'Unknown'
            color = (0, 0, 255)
        text = f"{text_label} ({confidence:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if frame_count % 3 != 0:
        future_motion = executor.submit(detect_moving_objects, frame, mog2, kf, measurement, prediction, frame_count)
        last_motion_frame = future_motion.result()

    if last_motion_frame is not None:
        frame = last_motion_frame

    return frame, last_faces_predictions, last_face_coords, last_motion_frame

def process_video(intrinsics, dist_coeffs, calibration=False):
    '''
    Bucle principal que:
      1. No. Pregunta si hay que actualizar (Y/N) el modelo de reconocimiento facial.
      2. Inicia la cámara Picamera2.
      3. Configura MOG2 y el filtro de Kalman.
      4. En cada frame, detecta movimiento y, si hay, detecta rostros.
      5. Ofrece la opción de presionar 'f' para iniciar process_pattern().
      6. Si se completa el patrón o se reconoce un rostro, desactiva la alarma.
      7. Muestra la ventana con el FPS y el estado de la alarma.

    Args:
        intrinsics (np.array): Matriz intrínseca (opcional).
        dist_coeffs (np.array): Coeficientes de distorsión (opcional).
        calibration (bool): Indica si se hizo calibración.
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    preprocess = input("Is it necessary to include new images in the model? [Y/N]: ")
    if preprocess.upper() == "Y":
        prepare_clean_train_folder(face_cascade, './train', './train_limpio')
        train_model()

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('modelo_lbphface.xml')
    names = {0: "Nico", 1: "Kike"}

    picam2 = initialize_camera()
    mog2 = configure_background_subtractor()
    kf, measurement, prediction = configure_kalman_filter()

    print("Procesando video... Presiona 'q' para salir. Presiona 'f' para verificar patrón de figuras.")

    frame_count = 0
    last_faces_predictions, last_face_coords, last_motion_frame = [], [], None
    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0.0
    alarm_active = True
    motion_detected = False

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                print("No se pudo capturar frame.")
                break

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            fg_mask = mog2.apply(frame_bgr)
            white_pixels = cv2.countNonZero(fg_mask)
            threshold_movement = 500

            if white_pixels > threshold_movement:
                motion_detected = True
            else:
                motion_detected = False

            if motion_detected and alarm_active:
                frame_result, last_faces_predictions, last_face_coords, last_motion_frame = process_faces_and_motion(
                    frame_bgr, gray_frame, mog2, kf, measurement, prediction,
                    face_cascade, face_recognizer, names, executor,
                    last_faces_predictions, last_face_coords, last_motion_frame, 
                    frame_count
                )

                recognized_face = False
                for (label, confidence) in last_faces_predictions:
                    if label in names and confidence < 95.0:
                        recognized_face = True
                        break

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    print("Detección de patrón activada. Tienes 30s para completar: Cuadrado->Triangulo->Hexagono.")
                    pattern_ok = process_pattern(picam2)
                    if pattern_ok:
                        print("Patrón completado exitosamente.")
                        alarm_active = False
                    else:
                        print("El patrón no se completó correctamente.")
                        alarm_active = True

                if recognized_face:
                    alarm_active = False
            else:
                frame_result = frame_bgr
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            if not alarm_active:
                kf = None
                last_motion_frame = None

            if alarm_active:
                cv2.putText(frame_result, "ALARMA ACTIVADA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame_result, "ALARMA DESACTIVADA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_count += 1
            fps_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_start_time = time.time()

            cv2.putText(frame_result, f"FPS: {current_fps:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Procesamiento de Video', frame_result)

    cv2.destroyAllWindows()
    picam2.stop()

def main(calibrate):
    '''
    Punto de entrada principal:
    - Detecta si se incluye '--calibrate' en los argumentos de línea de comandos.
    - Si es así, llama a calibrate_camera_system() para obtener intrinsics y dist_coeffs.
    - Luego llama a process_video() para iniciar el flujo de captura y procesamiento.

    Args:
        calibrate (bool): Indica si se deben calibrar los parámetros de la cámara.
    '''
    intrinsics, dist_coeffs = (None, None)
    if calibrate:
        intrinsics, dist_coeffs = calibrate_camera_system()
    process_video(intrinsics, dist_coeffs, calibration=calibrate)

if __name__ == '__main__':
    calibrate = '--calibrate' in sys.argv
    main(calibrate)
