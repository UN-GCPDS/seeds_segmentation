# Seed Detection and Segmentation Notebooks

Este repositorio contiene una colección de cuadernos Jupyter para detección y segmentación de semillas utilizando diferentes modelos de aprendizaje profundo. Los cuadernos están diseñados para ejecutarse en Kaggle con Accelerator GPU T4 x 2, pero también pueden adaptarse a otros entornos.

## Contenido

El repositorio incluye los siguientes cuadernos:

1. **UNet**:
   - `UNet - Seeds - Train.py`: Cuaderno para entrenar el modelo UNet para segmentación de semillas.
   - `UNet - Seeds - Train.py`: Cuaderno para realizar inferencia con el modelo UNet entrenado.

2. **EffiSegNet**:
   - `EffiSegNet - Seeds - Train.py`: Cuaderno para entrenar el modelo EffiSegNet para segmentación de semillas.
   - `EffiSegNet - Seeds - Test.py`: Cuaderno para realizar inferencia con el modelo EffiSegNet entrenado.

3. **ResUNet**:
   - `ResUNet - Seeds - Train.py`: Cuaderno para entrenar el modelo ResUNet para segmentación de semillas.
   - `ResUNet - Seeds - Train.py`: Cuaderno para realizar inferencia con el modelo ResUNet entrenado.

4. **YOLOv11**:
   - Detección de objetos:
     - `YOLOv11 - Seeds - Obj Dect - Train.py`: Cuaderno para entrenar YOLOv11 para detección de semillas.
     - `YOLOv11 - Seeds - Obj Dect - Test.py`: Cuaderno para realizar inferencia con YOLOv11 para detección de semillas.
   - Segmentación:
     - `YOLOv11 - Seeds - Seg - Train.py`: Cuaderno para entrenar YOLOv11 para segmentación de semillas.
     - `YOLOv11 - Seeds - Seg - Test.py`: Cuaderno para realizar inferencia con YOLOv11 para segmentación de semillas.

## Requisitos

Para ejecutar estos cuadernos, se requiere lo siguiente:

- Python 3.10 o superior
- TensorFlow 2.15 o superior
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
2. Asegúrate de tener las imágenes de prueba en la carpeta `./datasets/Seeds/Test/images/` (para UNET, EffiSegNet y ResUNet) o en la carpeta correspondiente para YOLOv11.
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

A continuación se presentan las medidas de rendimiento esperadas para cada modelo en la base de datos de semillas:

| Modelo               | Variante | Dice Coefficient | Jaccard Index | Sensitivity | Specificity | Precision (P) | Recall (R)  | mAP50 | mAP50-95 |
|----------------------|-----------|------------------|---------------|-------------|-------------|---------------|-------------|-------|----------|
| UNet                 | 2c        | 0.67455          | 0.60085       | 0.76881     | 0.76881     | -             | -           | -     | -        |
| UNet                 | 3c        | 0.59769          | 0.54479       | 0.60893     | 0.91768     | -             | -           | -     | -        |
| EffiSegNet           | 2c        | 0.82524          | 0.77986       | 0.80327     | 0.97725     | -             | -           | -     | -        |
| ResUNet              | 2c        | 0.96209          | 0.93151       | 0.95886     | 0.95886     | -             | -           | -     | -        |
| ResUNet              | 3c        | 0.80689          | 0.75730       | 0.79345     | 0.97511     | -             | -           | -     | -        |
| YOLOv11 (Obj Dect)     | default   | -                | -             | -           | -           | 0.901         | 0.903       | 0.947 | 0.755    |
| YOLOv11 (Seg)          | segmentation | -                | -             | -           | -           | 0.904         | 0.891       | 0.938 | 0.619    |

Los cuadernos incluyen visualizaciones de los resultados de entrenamiento y ejemplos de inferencia. Los modelos entrenados se guardan en la carpeta `./models/` y los resultados de evaluación se guardan en la carpeta `./results/`.

## Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras errores o tienes sugerencias para mejorar estos cuadernos, por favor abre un issue o envía una pull request.

## Licencia

Este proyecto está bajo la licencia BSD 2-Clause. Consulta el archivo LICENSE para más información.

---

¡Listo para detectar y segmentar semillas con estos poderosos modelos! 🌱🔍
