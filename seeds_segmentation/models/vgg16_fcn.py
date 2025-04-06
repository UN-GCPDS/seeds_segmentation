import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

def fcn_vgg16(input_shape=(224, 224, 3), out_channels=1, out_ActFunction='sigmoid'):
    # Cargar VGG16 sin la parte fully connected (top), con pesos de ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar capas del encoder
    for layer in base_model.layers:
        layer.trainable = False

    # Encoder (como en MobileNetV2 FCN)
    x = base_model.output

    # Decoder estilo FCN (upsampling con Conv2DTranspose)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)  # 7x7 -> 14x14
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)  # 14x14 -> 28x28
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)   # 28x28 -> 56x56
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)   # 56x56 -> 112x112
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x)   # 112x112 -> 224x224
    x = layers.BatchNormalization()(x)

    # Capa final
    outputs = layers.Conv2D(out_channels, (1, 1), activation=out_ActFunction)(x)

    return Model(inputs=base_model.input, outputs=outputs, name='VGG16_FCN')

if __name__ == '__main__':
    model = fcn_vgg16()
    model.summary()