from tensorflow.keras import Model, layers, initializers
import tensorflow as tf
from tensorflow.keras import layers

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, units, kernel_initializer, name=None):
        super(ResBlock, self).__init__(name=name)
        self.units = units
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.conv00 = layers.Conv2D(self.units, (1, 1), (1, 1),
                                    kernel_initializer=self.kernel_initializer,
                                    padding='same', name=f'{self.name}_Conv00')
        self.bn00 = layers.BatchNormalization(name=f'{self.name}_Batch00')
        self.act00 = layers.Activation('relu', name=f'{self.name}_Act00')

        self.conv01 = layers.Conv2D(self.units, (3, 3), (1, 1),
                                    kernel_initializer=self.kernel_initializer,
                                    padding='same', name=f'{self.name}_Conv01')
        self.bn01 = layers.BatchNormalization(name=f'{self.name}_Batch01')

        self.conv02 = layers.Conv2D(self.units, (1, 1), (1, 1),
                                    kernel_initializer=self.kernel_initializer,
                                    padding='same', name=f'{self.name}_Conv02')
        self.bn02 = layers.BatchNormalization(name=f'{self.name}_Batch02')

    def call(self, x):
        x_c = x
        x = self.conv00(x)
        x = self.bn00(x)
        x = self.act00(x)

        x = self.conv01(x)
        x = self.bn01(x)

        x_c = self.conv02(x_c)
        x_c = self.bn02(x_c)

        x = layers.Add()([x, x_c])
        x = layers.Activation('relu', name=f'{self.name}_Act01')(x)
        return x

def kernel_initializer(seed):
    return initializers.GlorotUniform(seed=seed)

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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_encoder(input_shape=[None,None,3], trainable = True, name="encoder"):
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV2(input_tensor=Input, include_top=False)
    layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder  = tf.keras.Model(inputs=Input, outputs=layers,name=name)
    encoder.trainable = trainable

    return encoder
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_decoder(skips,dropout=0):
    up_stack = [
        upsample(512, 3,dropout=dropout),  # 4x4 -> 8x8
        upsample(256, 3,dropout=dropout),  # 8x8 -> 16x16
        upsample(128, 3,dropout=dropout),  # 16x16 -> 32x32
        upsample(64, 3,dropout=dropout),   # 32x32 -> 64x64
    ]
    rest1_stack = [
        ResBlock(units=64,kernel_initializer=kernel_initializer(65),name='Res10'),
        ResBlock(units=32,kernel_initializer=kernel_initializer(34),name='Res12'),
        ResBlock(units=16,kernel_initializer=kernel_initializer(32),name='Res14'),
        ResBlock(units=8,kernel_initializer=kernel_initializer(4),name='Res16'),
    ]
    rest2_stack = [
        ResBlock(units=64,kernel_initializer=kernel_initializer(87),name='Res11'),
        ResBlock(units=32,kernel_initializer=kernel_initializer(4),name='Res13'),
        ResBlock(units=16,kernel_initializer=kernel_initializer(42),name='Res15'),
        ResBlock(units=8,kernel_initializer=kernel_initializer(6),name='Res17'),
    ]
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up,skip,rest1,rest2 in zip(up_stack,skips, rest1_stack, rest2_stack):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x,skip])
        x = rest1(x)
        x = rest2(x)
    return x

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def restunet_mobilenetv2(input_shape=(128,128,3), out_channels=1, out_ActFunction='sigmoid', trainable = False, name="unetMobile"):
    input = tf.keras.layers.Input(shape=input_shape)

    skips = get_encoder(input_shape=list(input.shape[1:]),  trainable = trainable)(input)

    x = get_decoder(skips, dropout=0.25)

    last = tf.keras.layers.Conv2DTranspose(
        out_channels, kernel_size=(3,3), strides=2,
        padding='same',activation=out_ActFunction)  #64x64 -> 128x128

    x = last(x)
    model = tf.keras.Model(inputs=input, outputs=x,name=name)
    return model

if __name__ == '__main__':
    model = restunet_mobilenetv2()
    model.summary()