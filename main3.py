import cv2
import glob
from utils import prepare_clean_train_folder, train_model, detect_and_crop_faces, predict_faces
from calibration_camara import (
    load_images, detect_corners, refine_corners, generate_chessboard_points,
    calibrate_camera, undistort_image
)

def main():
    # Configuración de calibración
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

    # Reconocimiento facial
    preprocess = input("Is it necessary to include new images in the model? [Y/N]: ")
    enters = True if preprocess == "Y" else False
    if enters:
        prepare_clean_train_folder()
        train_model()
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('/Users/ndelvalalvarez/Downloads/TERECERO/Computer_VIsion/ProyectoFinal/modelo_lbphface.xml')  # Cambia la ruta

    cap = cv2.VideoCapture(0)
    names = {0: "Nico", 1: "Kike"}

    print("Grabando video... Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Corregir distorsión
        undistorted_frame = undistort_image(frame, intrinsics, dist_coeffs)

        # Procesar reconocimiento facial
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        cropped_faces, face_coords = detect_and_crop_faces(gray_frame)
        predictions = predict_faces(cropped_faces, face_recognizer)

        for (x, y, w, h), (label, confidence) in zip(face_coords, predictions):
            color = (255, 0, 0) if label in names and confidence < 85.0 else (0, 0, 255)
            text = f"{names.get(label, 'Unknown')} ({confidence:.2f})"
            cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(undistorted_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Reconocimiento Facial', undistorted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
