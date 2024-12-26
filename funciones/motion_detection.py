import cv2
import numpy as np

# Función para configurar el modelo de fondo MOG2
def configure_background_subtractor():
    history = 500  # Número de frames para construir el modelo de fondo
    varThreshold = 16  # Umbral para detectar el fondo
    detectShadows = True  # Si es True, el algoritmo detecta las sombras
    mog2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
    return mog2

# Función para configurar el filtro de Kalman
def configure_kalman_filter():
    state_size = 4  # Posición y velocidad en 2D (x, y, vx, vy)
    measurement_size = 2  # Solo posición (x, y)

    kf = cv2.KalmanFilter(state_size, measurement_size)

    # Inicializar las matrices del filtro de Kalman
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

    kf.processNoiseCov = np.eye(state_size, dtype=np.float32) * 1e-4
    measurement = np.array((2, 1), np.float32)
    prediction = np.zeros((2, 1), np.float32)

    return kf, measurement, prediction

# Función para procesar el video y detectar objetos en movimiento
def detect_moving_objects(frame, mog2, kf, measurement, prediction, frame_count, update_interval=10):


    # Actualizar el modelo de fondo solo cada 'update_interval' frames
    if frame_count % update_interval == 0:
        mog2.apply(frame, learningRate=0.01)  # Actualiza el modelo de fondo de manera gradual

    # Aplicar el modelo de fondo MOG2
    mask = mog2.apply(frame)

    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Filtra áreas pequeñas
            # Obtener el rectángulo que rodea el contorno
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = x + w // 2
            cy = y + h // 2

            # Usar el filtro de Kalman para predecir y corregir la posición
            kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            prediction = kf.predict()
            measurement = np.array([[cx], [cy]], np.float32)
            kf.correct(measurement)

            # Dibujar el rectángulo alrededor del objeto en movimiento
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dibujar la posición predicha
            cv2.circle(frame, (int(prediction[0][0]), int(prediction[1][0])), 5, (0, 0, 255), -1)

    return frame
