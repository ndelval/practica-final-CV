import cv2
import glob
import os

# Ajusta a la carpeta Calibracion:
image_paths = glob.glob("Calibracion/*.jpg")

# Tamaño de las esquinas internas (en horizontal, vertical)
# Si tu tablero tiene 9 casillas en horizontal y 6 en vertical,
# son 8 esquinas internas por 5 esquinas internas:
pattern_size = (9, 6)

def test_chessboard_detection_all():
    if not image_paths:
        print("No se encontraron imágenes en la carpeta 'Calibracion/'.")
        return

    for i, img_path in enumerate(image_paths, start=1):
        # Carga la imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer la imagen {img_path}.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Intentamos encontrar las esquinas del tablero
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # Informamos el resultado
        if ret:
            print(f"Imagen {i}: Se encontraron esquinas en {os.path.basename(img_path)}.")
            # Refinar esquinas (opcional)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Dibujamos las esquinas en la imagen y mostramos
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        else:
            print(f"Imagen {i}: NO se encontraron esquinas en {os.path.basename(img_path)}.")

        # Descomenta estas líneas si quieres ver cada imagen en una ventana
        # cv2.imshow("Tablero", img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    test_chessboard_detection_all()
