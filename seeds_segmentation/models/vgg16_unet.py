import tensorflow as tf 

def upsample(filters,size,strides=2,padding="same",batchnorm=False,dropout=0):

    layer = tf.keras.Sequential()
    layer.add(
        tf.keras.layers.Conv2DTranspose(filters,size,strides,padding,use_bias = False))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    if dropout != 0:
        layer.add(tf.keras.layers.Dropout(dropout))

    layer.add(tf.keras.layers.ReLU())

    return layer

def get_encoder(input_shape=[None, None, 3], trainable=True, name="encoder"):
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.VGG16(input_tensor=Input, include_top=False)
    
    # Selecciona capas
    layer_names = [
        'block2_conv2',   # 64x64
        'block3_conv3',   # 32x32
        'block4_conv3',   # 16x16
        'block5_conv3',   # 8x8
        'block5_pool',    # 4x4 (capa de pooling en lugar de conv3)
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Crear el modelo de extracción de características
    encoder = tf.keras.Model(inputs=Input, outputs=layers, name=name)
    encoder.trainable = trainable

    return encoder

def get_decoder(skips,dropout=0):
    up_stack = [
        upsample(512, 3, batchnorm=True, dropout=dropout),  # 4x4 -> 8x8
        upsample(256, 3, batchnorm=True, dropout=dropout),  # 8x8 -> 16x16
        upsample(128, 3, batchnorm=True, dropout=dropout),  # 16x16 -> 32x32
        upsample(64, 3, batchnorm=True, dropout=dropout),   # 32x32 -> 64x64
    ]
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
    return x

def unet_vgg16(input_shape=(128, 128, 3), out_channels=1, out_ActFunction='sigmoid', trainable=False, name="uNetVGG16"):
    input = tf.keras.layers.Input(shape=input_shape)

    skips = get_encoder(input_shape=list(input.shape[1:]),  trainable = trainable)(input)

    x = get_decoder(skips, dropout=0.25)

    last = tf.keras.layers.Conv2DTranspose(
        out_channels, kernel_size=(3,3), strides=2,
        padding='same',activation=out_ActFunction)  #64x64 -> 128x128

    x = last(x)
    model = tf.keras.Model(inputs=input, outputs=x,name=name)
    return model