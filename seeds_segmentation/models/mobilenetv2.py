from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2DTranspose, Activation, UpSampling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2

def mobilenetv2(input_shape=(128, 128, 3), name='UNET', out_channels=1, out_ActFunction='sigmoid'):
    """
    Creates a U-Net-like segmentation model using MobileNetV2 as the backbone.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        name (str): Name of the model.
        out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
        out_ActFunction (str): Activation function for the output layer.

    Returns:
        model (Model): A Keras Model instance.
    """
    # Load pre-trained MobileNetV2 without top classification layers
    base_model = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)

    # Add a custom segmentation head
    x = base_model.output
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

    # Final output layer
    x = Conv2DTranspose(out_channels, (3, 3), strides=2, padding='same', activation=out_ActFunction)(x)

    # Create the segmentation model
    model = Model(inputs=base_model.input, outputs=x, name=name)

    return model

if __name__ == '__main__':
    # Create and summarize the model
    model = mobilenetv2()
    model.summary()