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

    def __init__(self, output_dim, kernel_size=3,
                 scale=None,
                 trainable_scale=False, trainable_W=False,
                 kernel='gaussian',
                 padding='VALID',
                 stride=1,
                 kernel_regularizer=None,
                 normalization=True,
                 seed=None,
                 mass=False,
                 **kwargs):
        
        super(ConvRFF, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.scale = scale 
        self.trainable_scale = trainable_scale 
        self.trainable_W = trainable_W
        self.padding = padding
        self.stride = stride
        self.rff_initializer = kernel
        self.kernel_regularizer = kernel_regularizer
        self.normalization = normalization
        self.seed = seed
        self.mass = mass

        print(f"[INIT] ConvRFF - output_dim={output_dim}, kernel={kernel}, scale={scale}, seed={seed}, trainable_W={trainable_W}")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'scale': self.scale,
            'trainable_scale': self.trainable_scale,
            'trainable_W': self.trainable_W,
            'padding': self.padding,
            'kernel': self.rff_initializer,
            'normalization': self.normalization,
            'seed': self.seed,
            'mass': self.mass
        })
        return config

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_initializer = _get_random_features_initializer(self.rff_initializer,
                                                             shape=(self.kernel_size,
                                                                   self.kernel_size,
                                                                   input_dim,
                                                                   self.output_dim),
                                                             seed=self.seed)
    
        print(f"[BUILD] kernel_initializer type: {type(kernel_initializer)}")
        print(f"[BUILD] scale: {self.scale}, initializer: {self.rff_initializer}, input_dim: {input_dim}")

        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, self.kernel_size,
                   input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=kernel_initializer,
            trainable=self.trainable_W,
            regularizer=self.kernel_regularizer,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=0.0, maxval=2*np.pi, seed=self.seed),
            trainable=self.trainable_W
        )

        if self.scale is None:
            if self.rff_initializer == 'gaussian':
                self.scale = float(np.sqrt((input_dim * self.kernel_size ** 2) / 2.0))
            elif self.rff_initializer == 'laplacian':
                self.scale = 1.0
            else:
                raise ValueError(f'Unsupported kernel initializer {self.rff_initializer}')
        print(f"[BUILD] Computed self.scale: {self.scale} (type: {type(self.scale)})")
        assert isinstance(self.scale, float), f"scale must be float, got {type(self.scale)}"

        self.kernel_scale = self.add_weight(
            name='kernel_scale',
            shape=(1,),
            dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(self.scale),
            trainable=self.trainable_scale,
            constraint=tf.keras.constraints.NonNeg(),
        )
    
    def compute_output_shape(self, input_shape):
        # Calculate output shape based on input shape, kernel size, stride, and padding
        batch_size, height, width, _ = input_shape
        
        if self.padding == 'VALID':
            out_height = (height - self.kernel_size + 1) // self.stride
            out_width = (width - self.kernel_size + 1) // self.stride
        elif self.padding == 'SAME':
            out_height = height // self.stride
            out_width = width // self.stride
        else:
            raise ValueError(f"Invalid padding: {self.padding}")
            
        return (batch_size, out_height, out_width, self.output_dim)

    def _compute_normal_probability(self, x, mean, std):
        constant = 1/(tf.math.sqrt(2*np.pi)*std)
        return constant*tf.math.exp(-0.5*(x-mean)*(x-mean)/(std*std))

    def _compute_mass(self):
        try:
            weights = tf.reshape(self.kernel, shape=(-1, self.output_dim))
            ww = tf.linalg.norm(weights, axis=0)
            ww_pos = tf.sort(ww)
            mean_pos = tf.reduce_mean(ww_pos)
            std_pos = tf.math.reduce_std(ww_pos)
            
            # Ensure std_pos is not too small to avoid numerical issues
            std_pos = tf.maximum(std_pos, 1e-6)
            
            mass_pos = self._compute_normal_probability(ww_pos, mean_pos, std_pos)
            
            # Use safe trapz implementation to avoid issues
            mass_result = tf.sqrt(tfp.math.trapz(tf.abs(mass_pos), ww_pos))
            
            # Ensure the result is a scalar tensor
            mass_result = tf.cast(mass_result, tf.float32)
            
            # Debug print to see what's being returned
            tf.print("Mass computed: ", mass_result)
            
            return mass_result
        except Exception as e:
            tf.print("Error in _compute_mass: ", e)
            # Return a safe default value in case of error
            return tf.constant(1.0, dtype=tf.float32)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        scale = tf.math.divide(1.0, self.kernel_scale)
        kernel = tf.math.multiply(scale, self.kernel)
        outputs = tf.nn.conv2d(inputs, kernel,
                              strides=[1, self.stride, self.stride, 1],
                              padding=self.padding)
        outputs = tf.nn.bias_add(outputs, self.bias)
        output_dim = tf.cast(self.output_dim, tf.float32)

        if self.normalization:
            outputs = tf.math.multiply(tf.math.sqrt(2/output_dim), tf.cos(outputs))
        else:
            outputs = tf.cos(outputs)

        if self.mass:
            mass_factor = tf.stop_gradient(self._compute_mass())
            outputs = tf.multiply(mass_factor, outputs)
            
        return outputs
    

def ConvRFF_block(x, deepth, mul_dim=3, block_id='01', trainable_W=True,
                 kernel_size=3, kernel_regularizer=None):

    phi_units = np.round(deepth*mul_dim).astype(np.uint32)
    x = ConvRFF(output_dim=phi_units, kernel_size=kernel_size,
               padding="SAME", trainable_scale=False, 
               trainable_W=trainable_W, name=f'ConvRFF_{block_id}', 
               mass=True,
               kernel_regularizer=kernel_regularizer)(x)
    return x