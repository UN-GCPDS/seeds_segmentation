import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp


def _get_random_features_initializer(initializer, shape, seed):
    print(f"[DEBUG] _get_random_features_initializer - initializer={initializer}, shape={shape}, seed={seed}")

    def _get_cauchy_samples(loc, scale, shape):
        np.random.seed(seed) 
        probs = np.random.uniform(low=0., high=1., size=shape)
        return loc + scale * np.tan(np.pi * (probs - 0.5))

    if isinstance(initializer, str):
        if initializer == "gaussian":
            print("[DEBUG] Gaussian initializer selected.")
            return tf.keras.initializers.RandomNormal(stddev=1.0, seed=seed)
        elif initializer == "laplacian":
            print("[DEBUG] Laplacian initializer selected.")
            return tf.keras.initializers.Constant(
                _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))

    raise ValueError(f'Unsupported kernel initializer {initializer}')


class ConvRFF(tf.keras.layers.Layer):
    def __init__(self,
                 output_dim,
                 kernel='gaussian',
                 scale=None,
                 seed=None,
                 trainable_W=True,
                 kernel_size=3,
                 padding="SAME",
                 trainable_scale=False,
                 mass=False,
                 kernel_regularizer=None,
                 **kwargs):
        super(ConvRFF, self).__init__(**kwargs)
        
        self.output_dim = output_dim
        self.kernel = kernel
        self.initial_scale = scale
        self.seed = seed
        self.trainable_W = trainable_W
        self.kernel_size = kernel_size
        self.padding = padding
        self.trainable_scale = trainable_scale
        self.mass = mass
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        kernel_shape = (self.kernel_size, self.kernel_size, input_dim, self.output_dim)

        self.kernel_initializer = _get_random_features_initializer(self.kernel, kernel_shape, self.seed)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=self.trainable_W,
            regularizer=self.kernel_regularizer
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )

        if self.initial_scale is not None:
            self.kernel_scale = tf.constant(self.initial_scale, dtype=tf.float32)
        else:
            self.kernel_scale = tf.constant(tf.math.sqrt(tf.cast(input_dim, tf.float32)), dtype=tf.float32)

        super(ConvRFF, self).build(input_shape)

    def call(self, inputs):
        outputs = tf.nn.conv2d(inputs, self.kernel,
                               strides=[1, 1, 1, 1],
                               padding=self.padding)

        outputs = tf.nn.bias_add(outputs, self.bias)

        output_dim = tf.cast(self.output_dim, tf.float32)

        if self.trainable_scale:
            outputs = outputs / self.kernel_scale

        outputs = tf.cos(outputs)

        if self.mass:
            outputs = outputs * tf.math.sqrt(2.0 / output_dim)

        return outputs

    def get_config(self):
        config = super(ConvRFF, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "kernel": self.kernel,
            "scale": self.initial_scale,
            "seed": self.seed,
            "trainable_W": self.trainable_W,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "trainable_scale": self.trainable_scale,
            "mass": self.mass,
            "kernel_regularizer": self.kernel_regularizer,
        })
        return config


def ConvRFF_block(x, deepth, mul_dim=3, block_id='01', trainable_W=True,
                 kernel_size=3, kernel_regularizer=None):

    phi_units = np.round(deepth*mul_dim).astype(np.uint32)
    x = ConvRFF(output_dim=phi_units, kernel_size=kernel_size,
               padding="SAME", trainable_scale=False, 
               trainable_W=trainable_W, name=f'ConvRFF_{block_id}', 
               mass=True,
               kernel_regularizer=kernel_regularizer)(x)
    return x