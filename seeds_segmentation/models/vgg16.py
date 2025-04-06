import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Activation, UpSampling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2

def vgg16(input_shape=(128, 128, 3), name='VGG16_EncoderDecoder',
                           out_channels=1, out_ActFunction='sigmoid'):
    # Cargar VGG16 sin las capas fully connected
    base_model = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)

    # Congelar el encoder si quieres transfer learning puro
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output  # Salida del encoder (VGG16)

    # Decoder (igual al estilo MobileNetV2 de tu archivo)
    x = Conv2DTranspose(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D((2, 2))(x)

    # Capa final para segmentación
    x = Conv2DTranspose(out_channels, (3, 3), strides=2, padding='same', activation=out_ActFunction)(x)

    return Model(inputs=base_model.input, outputs=x, name=name)

if __name__ == '__main__':
    model = vgg16()
    model.summary()
