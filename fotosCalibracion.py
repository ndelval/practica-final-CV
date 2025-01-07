import cv2
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    
    # Configuramos tamaño y formato
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    
    # Configuramos en modo preview
    picam.configure("preview")
    picam.start()

    i = 0
    while True:
        # Capturamos frame en formato RGB
        frame_rgb = picam.capture_array()
        # Convertimos a BGR para OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Mostramos el stream en una ventana de OpenCV
        cv2.imshow("Live Feed", frame_bgr)
        
        # Esperamos 1 ms por una tecla
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Salir
            break
        elif key == ord('c'):
            # Capturar una imagen
            i += 1
            filename = f"train/Kike/picam_1280_720_{i}.jpg"
            cv2.imwrite(filename, frame_bgr)
            print(f"Imagen guardada: {filename}")

    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
