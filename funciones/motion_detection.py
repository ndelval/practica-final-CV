import cv2
import numpy as np

# Configuración del modelo de fondo MOG2
def configure_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=500,  
        varThreshold=16,
        detectShadows=True 
    )

# Configuración del filtro de Kalman
def configure_kalman_filter():
    kf = cv2.KalmanFilter(4, 2) 

    # Matriz de transición (posición y velocidad constantes)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    # Matriz de medición (solo mide posición)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)

    # Covarianza del ruido del proceso
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4

    return kf, np.zeros((2, 1), np.float32), np.zeros((2, 1), np.float32)

# Detección de objetos en movimiento
def detect_moving_objects(frame, mog2, kf, measurement, prediction, frame_count, update_interval=50):
    """
    Procesa un frame para detectar objetos en movimiento y predice su posición.
    Args:
        frame: Frame actual en formato BGR.
        mog2: Modelo de fondo configurado con MOG2.
        kf: Filtro de Kalman.
        measurement: Medición actual de la posición.
        prediction: Predicción actual del filtro de Kalman.
        frame_count: Número actual de frame.
        update_interval: Intervalo para actualizar el modelo de fondo.
    Returns:
        Frame procesado con marcadores de objetos en movimiento.
    """
    # Actualizar el modelo de fondo periódicamente
    if frame_count % update_interval == 0:
        mog2.apply(frame, learningRate=0.01)

    # Generar máscara binaria con MOG2
    mask = mog2.apply(frame)
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procesar el contorno más grande si existe
    if contours:
        # Filtrar contornos pequeños
        largest_contour = max((c for c in contours if cv2.contourArea(c) > 500), key=cv2.contourArea, default=None)

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2  # Centro del objeto

            # Actualizar y predecir con el filtro de Kalman
            kf.statePost[:2] = np.array([[cx], [cy]], np.float32)
            prediction = kf.predict()
            measurement = np.array([[cx], [cy]], np.float32)
            kf.correct(measurement)

            # Dibujar marcador en el frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectángulo del objeto
            cv2.circle(frame, (int(prediction[0][0]), int(prediction[1][0])), 5, (0, 0, 255), -1)  # Predicción

    return frame
