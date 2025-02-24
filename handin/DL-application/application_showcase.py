import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from src.config import config
from src.create_dataset import create_dataset
from src.training_callbacks import ringring_callbackplease
from src.tensorboard import TensorboardLauncher

def build_model():
    inputs = tf.keras.layers.Input(shape=(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3))
    x = tf.keras.layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='valid',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001633))(inputs)
    # Layer 0
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 1: Conv
    x = tf.keras.layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='valid',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001633))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2: Conv
    x = tf.keras.layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='valid',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001633))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten before Dense layers
    x = tf.keras.layers.Flatten()(x)

    # Dense Layer 0
    x = tf.keras.layers.Dense(512, activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def create_visualization_model(model):
    """Create a model that returns the outputs of all Conv2D layers."""
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    return tf.keras.Model(inputs=model.input, outputs=layer_outputs)

def feature_map_to_rgb(feature_map):
    """Convert single channel to RGB by repeating the channel"""
    normalized = ((feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255).astype(np.uint8)
    # Stack the same values for R, G, and B channels
    rgb_image = np.stack([normalized] * 3, axis=-1)
    return rgb_image

def save_feature_maps(image, feature_maps):
    """Save and display feature maps in RGB format."""
    vis_dir = os.path.join(config.OUTPUT_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    n_layers = len(feature_maps)
    n_channels = 8  # Show 8 channels per layer
    
    # Create single figure with all visualizations
    plt.figure(figsize=(15, 3 * (n_layers + 1)))
    
    # Original image
    plt.subplot(n_layers + 1, 1, 1)
    plt.imshow(image)
    plt.title('Original Image')
    
    # Feature maps
    for layer_idx, feature_map in enumerate(feature_maps):
        for channel in range(n_channels):
            plt.subplot(n_layers + 1, n_channels, (layer_idx + 1) * n_channels + channel + 1)
            rgb_map = feature_map_to_rgb(feature_map[0, :, :, channel])
            plt.imshow(rgb_map)
            plt.axis('off')
            if channel == 0:
                plt.ylabel(f'Layer {layer_idx + 1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_maps.png'))
    plt.show()

def main():
    # Load data
    train_ds, val_ds = create_dataset()
    tensorboard = TensorboardLauncher(config.LOGS_DIR)
    tensorboard.start_tensorboard()
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.00040288),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # Print model summary
    model.summary()
    model.fit(
        train_ds,
        epochs=1,
        validation_data=val_ds,
        # callbacks=ringring_callbackplease,
    )
    
    # Create visualization model
    vis_model = create_visualization_model(model)
    
    # Get a sample image from validation dataset
    for images, _ in val_ds.take(1):
        sample_image = images[0]  # Take first image from batch
        break
    
    # Get feature maps for the sample image
    feature_maps = vis_model.predict(np.expand_dims(sample_image, axis=0))
    
    # Save and display feature maps
    save_feature_maps(sample_image, feature_maps)
    
if __name__ == "__main__":
    main()
