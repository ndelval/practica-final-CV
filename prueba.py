import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara predeterminada

# Verifica que la cámara se haya abierto correctamente
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

# Configuración de MOG2
history = 500  # Número de frames para construir el modelo de fondo
varThreshold = 16  # Umbral para detectar el fondo
detectShadows = True  # Si es True, el algoritmo detecta las sombras

# Crear el objeto MOG2 con los parámetros especificados
mog2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

# Inicializar el filtro de Kalman
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

# Captura en tiempo real y procesamiento de video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede obtener el frame de la cámara")
        break

    # Aplicar el modelo de fondo MOG2
    mask = mog2.apply(frame)

    # Encontrar contornos en la máscara binaria (movimiento detectado)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    # Mostrar el frame con el objeto seguido
    cv2.imshow('Frame con objeto en movimiento y Kalman', frame)

    # Esperar por una tecla para salir
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
