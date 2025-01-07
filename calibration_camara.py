import cv2
import numpy as np

def load_images(image_paths):
    """Carga imágenes desde una lista de rutas."""
    return [cv2.imread(path) for path in image_paths]

def detect_corners(images, pattern_size):
    """
    Detecta las esquinas en las imágenes del tablero de ajedrez.
    
    Args:
        images: Lista de imágenes.
        pattern_size: Dimensiones del tablero de ajedrez (filas, columnas).
    
    Returns:
        valid_images: Lista de imágenes válidas donde se encontraron esquinas.
        corners: Lista de esquinas detectadas.
    """
    valid_images = []
    corners = []
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corner_points = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            valid_images.append(img)
            corners.append((ret, corner_points))
        else:
            print(f"No se encontraron esquinas en la imagen {idx + 1}.")
    return valid_images, corners

def refine_corners(images_gray, corners, criteria):
    """
    Refina las esquinas detectadas en las imágenes.

    Args:
        images_gray: Lista de imágenes en escala de grises.
        corners: Lista de esquinas detectadas.
        criteria: Criterios de refinamiento.
    
    Returns:
        corners_refined: Lista de esquinas refinadas.
    """
    return [
        cv2.cornerSubPix(img_gray, cor[1], (11, 11), (-1, -1), criteria) if cor[0] else [] 
        for img_gray, cor in zip(images_gray, corners)
    ]

def generate_chessboard_points(pattern_size, square_size):
    """
    Genera puntos en 3D del tablero de ajedrez.

    Args:
        pattern_size: Dimensiones del tablero de ajedrez (filas, columnas).
        square_size: Tamaño de cada cuadrado en unidades arbitrarias.
    
    Returns:
        objp: Puntos del tablero en 3D.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    return objp

def calibrate_camera(chessboard_points, valid_corners, image_size):
    """
    Realiza la calibración de la cámara.

    Args:
        chessboard_points: Puntos del tablero de ajedrez en 3D.
        valid_corners: Esquinas detectadas en las imágenes.
        image_size: Tamaño de las imágenes (ancho, alto).
    
    Returns:
        rms: Error RMS.
        intrinsics: Matriz intrínseca.
        dist_coeffs: Coeficientes de distorsión.
        extrinsics: Matrices extrínsecas.
    """
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points, valid_corners, image_size, None, None
    )
    extrinsics = [
        np.hstack((cv2.Rodrigues(rvec)[0], tvec))
        for rvec, tvec in zip(rvecs, tvecs)
    ]
    return rms, intrinsics, dist_coeffs, extrinsics

def undistort_image(frame, intrinsics, dist_coeffs):
    """
    Corrige la distorsión de una imagen.

    Args:
        frame: Imagen original.
        intrinsics: Matriz intrínseca de la cámara.
        dist_coeffs: Coeficientes de distorsión.
    
    Returns:
        undistorted_frame: Imagen sin distorsión.
    """
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, intrinsics, dist_coeffs, None, new_camera_matrix)
    return undistorted_frame
