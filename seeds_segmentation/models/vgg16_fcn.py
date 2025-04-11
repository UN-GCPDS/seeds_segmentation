import tensorflow as tf 

def get_encoder(input_shape=[None, None, 3], trainable=True, name="encoder"):
    """
    Creates an encoder based on VGG16.
    Extracts features at different resolutions for use in the skip connections.
    """
    # Ensure input dimensions are compatible with VGG16 (multiple of 32)
    Input = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.VGG16(input_tensor=Input, include_top=False)
    
    # Select feature maps at different resolutions
    layer_names = [
        'block2_conv2',   # 64x64
        'block3_conv3',   # 32x32
        'block4_conv3',   # 16x16
        'block5_conv3',   # 8x8
        'block5_pool',    # 4x4 (capa de pooling en lugar de conv3)
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    encoder = tf.keras.Model(inputs=Input, outputs=layers, name=name)
    encoder.trainable = trainable

    return encoder 

def get_decoder(skips, num_classes):
    """
    Creates a decoder that gradually upsamples the feature maps and combines them
    with corresponding skip connections from the encoder.
    """
    # Start with the bottleneck features (4x4)
    x = skips[-1]
    
    # Define filter sizes for each upsampling stage (gradually decreasing)
    filter_sizes = [256, 128, 64, 32]
    
    # Process each skip connection, from deepest to shallowest
    for i, skip in enumerate(reversed(skips[:-1])):
        filters = filter_sizes[i] if i < len(filter_sizes) else 32
        
        # Upsample current features
        x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Process skip connection with 1x1 convolution to match channels if needed
        skip_processed = tf.keras.layers.Conv2D(filters, 1, padding="same")(skip)
        skip_processed = tf.keras.layers.BatchNormalization()(skip_processed)
        
        # Combine upsampled features with skip connection
        x = tf.keras.layers.Concatenate()([x, skip_processed])
        
        # Additional convolution to process the combined features
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    # Final upsampling to original resolution (64x64 -> 128x128)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
     
    return x

def fcn_vgg16(input_shape=(128, 128, 3), out_channels=1, out_ActFunction='sigmoid', trainable = False, name="fcnVGG16"):
    """
    Creates a Fully Convolutional Network based on VGG16 for semantic segmentation.
    
    Args:
        input_shape: Input image shape, should be divisible by 32 for optimal performance
        num_classes: Number of output classes (1 for binary segmentation)
        trainable: Whether to fine-tune the VGG16 backbone
        name: Model name
        
    Returns:
        A Keras Model for semantic segmentation
    """
    # Verify input dimensions are compatible with VGG16
    if input_shape[0] % 32 != 0 or input_shape[1] % 32 != 0:
        print(f"Warning: Input dimensions {input_shape[:2]} are not divisible by 32. "
              f"This may cause issues with feature map alignment.")
    
    # Create the model
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    skips = get_encoder(input_shape=list(input_tensor.shape[1:]), trainable=trainable)(input_tensor)
    x = get_decoder(skips, out_ActFunction)
    x   = tf.keras.layers.Conv2D(out_channels, 1, padding="same", activation=out_ActFunction)(x)
    
    # Create and return the model
    model = tf.keras.Model(inputs=input_tensor, outputs=x, name=name)
    return model
if __name__ == '__main__':
    model = fcn_vgg16()
    model.summary()