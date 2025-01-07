import cv2
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    
    
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    
    
    picam.configure("preview")
    picam.start()

    i = 0
    while True:
        
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    
        cv2.imshow("Live Feed", frame_bgr)
        
    
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            
            break
        elif key == ord('c'):
            
            i += 1
            filename = f"train/Kike/picam_1280_720_{i}.jpg"
            cv2.imwrite(filename, frame_bgr)
            print(f"Imagen guardada: {filename}")

    picam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
