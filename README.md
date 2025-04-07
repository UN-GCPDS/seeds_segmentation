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
     - `YOLOv11 - Seeds - Seg - Test and Metrics.py`: Cuaderno para realizar inferencia con YOLOv11 para segmentación de semillas y medir metricas.

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
2. Asegúrate de tener las imágenes de prueba en la carpeta `./datasets/Seeds/Test/images/` (para UNet, EffiSegNet y ResUNet) o en la carpeta correspondiente para YOLOv11.
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

A continuación se presentan las medidas de rendimiento obtenidas para cada modelo en la base de datos de semillas de tomate del Grupo GPDS con data Aumentation:

| Modelo               | Variante | Dice Coefficient | Jaccard Index | Sensitivity | Specificity | Precision (P) | Recall (R)  | mAP50 | mAP50-95 |
|----------------------|----------|------------------|---------------|-------------|-------------|---------------|-------------|-------|----------|
| UNet                | 2c       | 0.95020          | 0.91299       | 0.94734     | 0.94734     | 0.95506     | 0.94734     | 0.99153     | 0.85805     |
| UNet                | 3c       | 0.69584          | 0.63568       | 0.74289     | 0.94567     | 0.72038     | 0.74289     | 0.66384     | 0.52034     |
| ResUNet             | 2c       | 0.94122          | 0.90118       | 0.93793     | 0.93793     | 0.94804     | 0.93793     | 0.96610     | 0.83644     |
| ResUNet             | 3c       | 0.74074          | 0.68176       | 0.74376     | 0.95489     | 0.75792     | 0.74376     | 0.72881     | 0.57090     |
| FCN                 | 2c       | 0.95105          | 0.91409       | 0.95244     | 0.95244     | 0.95094     | 0.95244     | 0.99153     | 0.85763     |
| FCN                 | 3c       | 0.82072          | 0.76505       | 0.76861     | 0.96151     | 0.78149     | 0.76861     | 0.83616     | 0.65932     |
| MoblilenetV2        | 2c       | 0.94514          | 0.90328       | 0.95115     | 0.95115     | 0.94017     | 0.95115     | 0.99153     | 0.84195     |
| MoblilenetV2        | 3c       | 0.82374          | 0.76559       | 0.77379     | 0.96243     | 0.76899     | 0.77379     | 0.85876     | 0.65565     |
| UNetMobV2           | 2c       | 0.96443          | 0.93545       | 0.96685     | 0.96685     | 0.96280     | 0.96685     | 1.00000     | 0.90254     |
| UNetMobV2           | 3c       | 0.83767          | 0.78960       | 0.79887     | 0.97337     | 0.80216     | 0.79887     | 0.87288     | 0.70198     |
| ResUNetMobV2        | 2c       | 0.96339          | 0.93345       | 0.96966     | 0.96966     | 0.95794     | 0.96966     | 1.00000     | 0.89915     |
| ResUNetMobV2        | 3c       | 0.82117          | 0.77275       | 0.80076     | 0.97651     | 0.79614     | 0.80076     | 0.86723     | 0.86723     |
| FCNMobV2            | 2c       | 0.96650          | 0.93888       | 0.96667     | 0.96667     | 0.96715     | 0.96667     | 1.00000     | 0.90975     |
| FCNMobV2            | 3c       | 0.69838          | 0.63728       | 0.76532     | 0.94403     | 0.71174     | 0.76532     | 0.71469     | 0.53446     |
| UNetVGG16           | 2c       | 0.96416          | 0.93500       | 0.96449     | 0.96449     | 0.96474     | 0.96449     | 1.00000     | 0.90297     |
| UNetVGG16           | 3c       | 0.81483          | 0.76733       | 0.79489     | 0.97231     | 0.80621     | 0.79489     | 0.85593     | 0.68192     |
| ResUNetVGG16        | 2c       | 0.96342          | 0.93353       | 0.96676     | 0.96676     | 0.96070     | 0.96676     | 1.00000     | 0.89830     |
| ResUNetVGG16        | 3c       | 0.68708          | 0.60992       | 0.69209     | 0.95733     | 0.71942     | 0.69209     | 0.67232     | 0.44831     |
| VGG16               | 2c       | 0.90330          | 0.84424       | 0.92208     | 0.92208     | 0.88786     | 0.92208     | 0.94915     | 0.73263     |
| VGG16               | 3c       | 0.74920          | 0.67403       | 0.69309     | 0.91286     | 0.71485     | 0.69309     | 0.68079     | 0.52458     |

A continuación se presentan las medidas de rendimiento esperadas para cada modelo de segmentación en la base de datos de semillas:

| Modelo               | Variante     | Dice Coefficient | Jaccard Index | Sensitivity | Specificity | 
|----------------------|--------------|------------------|---------------|-------------|-------------|
| UNet                 | 2c           | 0.67455          | 0.60085       | 0.76881     | 0.76881     | 
| UNet                 | 3c           | 0.59769          | 0.54479       | 0.60893     | 0.91768     | 
| EffiSegNet           | 2c           | 0.82524          | 0.77986       | 0.80327     | 0.97725     | 
| ResUNet              | 2c           | 0.96209          | 0.93151       | 0.95886     | 0.95886     | 
| ResUNet              | 3c           | 0.80689          | 0.75730       | 0.79345     | 0.97511     | 
| YOLOv11 (Seg)        | segmentation | 0.88057          | 0.78972       | 0.95254     | 0.99072     |

Para finalizar se presenta las medidas de deteccion del modelo YOLOv11m:

| Modelo               | Variante | Precision (P) | Recall (R)  | mAP50 | mAP50-95 |
|----------------------|----------|---------------|-------------|-------|----------|
| YOLOv11 (Obj Dect)   | default  | 0.901         | 0.903       | 0.947 | 0.755    |


Los cuadernos incluyen visualizaciones de los resultados de entrenamiento y ejemplos de inferencia. Los modelos entrenados se guardan en la carpeta `./models/` y los resultados de evaluación se guardan en la carpeta `./results/`.

## Imágenes Ilustrativas

A continuación se presentan ejemplos visuales del proceso completo:

- Imagen de Entrada:

<img src="https://github.com/user-attachments/assets/7035abaa-0cf0-4d56-b915-3d59c2285b2a" alt="original_seed" width="400"/>

- Detección:

<img src="https://github.com/user-attachments/assets/a84bd455-c067-4d2d-a556-e8bbf1ebd722" alt="detection_seed" width="400"/>

- Segmentación:

<img src="https://github.com/user-attachments/assets/3e8e014c-fd01-4a28-af9a-b6928c1e9390" alt="segmentation_seed" width="400"/>

## Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras errores o tienes sugerencias para mejorar estos cuadernos, por favor abre un issue o envía una pull request.

## Licencia

Este proyecto está bajo la licencia BSD 2-Clause. Consulta el archivo LICENSE para más información.

---

¡Listo para detectar y segmentar semillas con estos poderosos modelos! 🌱🔍
