# Reconocimiento Facial y Detección de Movimiento con OpenCV

## Descripción

Este proyecto implementa un sistema completo de procesamiento de video en tiempo real que combina:
- **Reconocimiento facial**: Utilizando el modelo LBPH para identificar rostros.
- **Detección de movimiento**: Implementando un modelo MOG2 para identificar objetos en movimiento.
- **Calibración de cámara**: Usando un tablero de ajedrez para calcular parámetros intrínsecos y coeficientes de distorsión.
- **Detección de figuras geométricas**: Análisis de contornos para identificar formas geométricas en imágenes.

El proyecto está diseñado para procesar video en tiempo real, combinando tareas como reconocimiento facial, detección de movimiento y análisis de figuras.

## Características principales
- **Preprocesamiento de datos de entrenamiento**: Limpieza y preparación de imágenes para entrenamiento.
- **Modelo de reconocimiento facial**: Entrenamiento y predicción con el modelo LBPH.
- **Filtro de Kalman**: Seguimiento y predicción de objetos en movimiento.
- **Calibración de cámara**: Utilizando imágenes de un tablero de ajedrez.

## Requisitos
- Python 3.8 o superior
- OpenCV 4.5.0 o superior
- Numpy
- Librerías estándar de Python

## Instalación

1. Clona este repositorio:

   ```bash
   git clone https://github.com/ndelval/practica-final-CV.git
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```


## Uso

### Calibración de cámara

Para calibrar la cámara con imágenes de un tablero de ajedrez y luego iniciar el procesamiento de video en tiempo real, ejecuta:

```bash
python main3.py --calibrate
```

Presiona:
- `q`: Salir del programa.
- `f`: Detectar figuras geométricas durante 30 segundos.

## Estructura del proyecto

```
.
├── funciones/
│   ├── utils.py                # Utilidades generales
│   ├── face_detection.py       # Funciones relacionadas con la detección facial
│   ├── calibration_camara.py   # Funciones para la calibración de cámara
│   ├── shape_detection.py      # Detección de figuras geométricas
│   ├── motion_detection.py     # Detección de movimiento con MOG2 y Kalman
├── modelo_lbphface.xml          # Modelo LBPH entrenado
├── train/                       # Imágenes originales de entrenamiento
├── train_limpio/                # Imágenes procesadas para entrenamiento limpio
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Documentación del proyecto
└── main3.py       # Script principal
```

---
Realizado por Enrique Rodríguez y Nicolás del Val.



