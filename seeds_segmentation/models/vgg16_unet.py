import tensorflow as tf

def upsample(filters, size, strides=2, padding="same", batchnorm=False, dropout=0):
    """
    Creates an upsampling block with transposed convolution, optional batch normalization,
    optional dropout, and ReLU activation.
    
    Args:
        filters: Number of filters in the transposed convolution
        size: Kernel size for the transposed convolution
        strides: Stride size for the transposed convolution
        padding: Padding type ('same' or 'valid')
        batchnorm: Whether to include batch normalization
        dropout: Dropout rate (0 means no dropout)
        
    Returns:
        A sequential model for upsampling
    """
    layer = tf.keras.Sequential()
    layer.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides, padding, use_bias=not batchnorm))

    if batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())

    layer.add(tf.keras.layers.ReLU())
    
    if dropout > 0:
        layer.add(tf.keras.layers.Dropout(dropout))

    return layer

def get_encoder(input_shape=[None, None, 3], trainable=True, name="encoder"):
    """
    Creates an encoder based on VGG16.
    Extracts features at different resolutions for use in the skip connections.
    
    Args:
        input_shape: Shape of the input image
        trainable: Whether the VGG16 weights should be trainable
        name: Name for the encoder model
        
    Returns:
        A Keras model that outputs feature maps at different resolutions
    """
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.vgg16.VGG16(input_tensor=Input, include_top=False)
    
    # Select layers for skip connections
    layer_names = [
        'block1_conv2',   # 64x64
        'block2_conv2',   # 32x32
        'block3_conv3',   # 16x16
        'block4_conv3',   # 8x8
        'block5_conv3',   # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder = tf.keras.Model(inputs=Input, outputs=layers, name=name)
    encoder.trainable = trainable

    return encoder

def get_decoder(skips, dropout=0):
    """
    Creates a decoder that gradually upsamples the feature maps and combines them
    with corresponding skip connections from the encoder.
    
    Args:
        skips: List of feature maps from the encoder at different resolutions
        dropout: Dropout rate for regularization
        
    Returns:
        Tensor representing the decoded feature map
    """
    # Define upsampling blocks with decreasing filter sizes
    up_stack = [
        upsample(512, 3, dropout=dropout),  # 4x4 -> 8x8
        upsample(256, 3, dropout=dropout),  # 8x8 -> 16x16
        upsample(128, 3, dropout=dropout),  # 16x16 -> 32x32
        upsample(64, 3, dropout=dropout),   # 32x32 -> 64x64
    ]
    
    # Start with the bottleneck features
    x = skips[-1]
    
    # Reverse the skip connections (excluding the bottleneck)
    # to process from bottom to top of the U-Net
    skips = reversed(skips[:-1])

    # Apply each upsampling block and concatenate with the corresponding skip connection
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()([x, skip])
        
        # Add extra convolution after concatenation to better process combined features
        x = tf.keras.layers.Conv2D(
            filters=concat.shape[-1] // 2,  # Reduce number of channels
            kernel_size=3,
            padding='same',
            activation='relu'
        )(concat)
        
        # Add batch normalization for stability
        x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    
    return x

def unet_vgg16(input_shape=(128, 128, 3), out_channels=1, out_ActFunction='sigmoid', 
               trainable=False, dropout=0, name="unetVgg16"):
    """
    Creates a U-Net model with VGG16 as the encoder for semantic segmentation.
    
    Args:
        input_shape: Input image shape (should be divisible by 32 for optimal performance)
        out_channels: Number of output channels (1 for binary segmentation)
        out_ActFunction: Activation function for the output layer
        trainable: Whether to fine-tune the VGG16 backbone
        dropout: Dropout rate for regularization
        name: Model name
        
    Returns:
        A Keras Model for semantic segmentation
    """
    # Verify input dimensions are compatible
    if input_shape[0] % 32 != 0 or input_shape[1] % 32 != 0:
        print(f"Warning: Input dimensions {input_shape[:2]} are not divisible by 32. "
              f"This may cause issues with feature map alignment.")
    
    # Create input layer
    input = tf.keras.layers.Input(shape=input_shape)
    
    # Get encoder features
    skips = get_encoder(input_shape=list(input.shape[1:]), trainable=trainable)(input)
    
    # Get decoder features
    x = get_decoder(skips, dropout=dropout)
    
    # Final upsampling to original image size (64x64 -> 128x128)
    x = tf.keras.layers.Conv2DTranspose(
        out_channels, 
        kernel_size=3,  # Changed from 1x1 to 3x3 for better feature extraction
        strides=2,
        padding='same',
        activation=out_ActFunction
    )(x)
    
    # Create and return the model
    model = tf.keras.Model(inputs=input, outputs=x, name=name)
    return model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    model = unet_vgg16()
    model.summary()