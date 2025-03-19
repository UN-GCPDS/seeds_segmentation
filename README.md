# Seed Detection and Segmentation Notebooks

Este repositorio contiene una colección de cuadernos Jupyter para detección y segmentación de semillas utilizando diferentes modelos de aprendizaje profundo. Los cuadernos están diseñados para ejecutarse en Kaggle con Accelerator GPU T4 x 2, pero también pueden adaptarse a otros entornos.

## Contenido

El repositorio incluye los siguientes cuadernos:

1. **EffiSegNet**:
   - `effisegnet_seeds_train.py`: Cuaderno para entrenar el modelo EffiSegNet para segmentación de semillas.
   - `effisegnet_seeds_test.py`: Cuaderno para realizar inferencia con el modelo EffiSegNet entrenado.

2. **ResUNet**:
   - `resunet_seeds_train.py`: Cuaderno para entrenar el modelo ResUNet para segmentación de semillas.
   - `resunet_seeds_test.py`: Cuaderno para realizar inferencia con el modelo ResUNet entrenado.

3. **YOLOv11**:
   - Detección de objetos:
     - `yolov11_seeds_obj_dect_train.py`: Cuaderno para entrenar YOLOv11 para detección de semillas.
     - `yolov11_seeds_obj_dect_test.py`: Cuaderno para realizar inferencia con YOLOv11 para detección de semillas.
   - Segmentación:
     - `yolov11_seeds_seg_train.py`: Cuaderno para entrenar YOLOv11 para segmentación de semillas.
     - `yolov11_seeds_seg_test.py`: Cuaderno para realizar inferencia con YOLOv11 para segmentación de semillas.

## Requisitos

Para ejecutar estos cuadernos, se requiere lo siguiente:

- Python 3.7 o superior
- TensorFlow 2.x
- PyTorch (para YOLOv11)
- OpenCV
- Matplotlib
- NumPy
- Kaggle API
- Roboflow API (para YOLOv11)
- Ultralytics (para YOLOv11)

## Cómo usar

### Entrenamiento

1. Clona este repositorio en tu máquina local o directamente en Kaggle.
2. Asegúrate de tener configurado tu entorno con las dependencias necesarias.
3. Elige el cuaderno de entrenamiento correspondiente al modelo que deseas usar.
4. Si vas a usar tus propios datos, asegúrate de estructurarlos según lo indicado en la sección "Estructura de datos".
5. Ejecuta el cuaderno de entrenamiento. Los modelos se guardarán automáticamente en la carpeta `./models/`.

### Inferencia

1. Después de entrenar un modelo o descargar uno preentrenado, usa el cuaderno de inferencia correspondiente.
2. Asegúrate de tener las imágenes de prueba en la carpeta `./datasets/Seeds/Test/images/` (para EffiSegNet y ResUNet) o en la carpeta correspondiente para YOLOv11.
3. Ejecuta el cuaderno de inferencia para obtener los resultados de detección o segmentación.

## Estructura de datos

Los cuadernos esperan que los datos estén organizados en la siguiente estructura:

```
datasets/
└── Seeds/
    ├── Train/
    │   ├── images/
    │   └── masks/
    ├── Valid/
    │   ├── images/
    │   └── masks/
    └── Test/
        ├── images/
        └── masks/
```

Para YOLOv11, la estructura es ligeramente diferente y se maneja a través del archivo `data.yaml` que se genera durante el proceso de descarga del dataset.

## Resultados

Los cuadernos incluyen visualizaciones de los resultados de entrenamiento y ejemplos de inferencia. Los modelos entrenados se guardan en la carpeta `./models/` y los resultados de evaluación se guardan en la carpeta `./results/`.

## Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras errores o tienes sugerencias para mejorar estos cuadernos, por favor abre un issue o envía una pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más información.

---

¡Listo para detectar y segmentar semillas con estos poderosos modelos! 🌱🔍
