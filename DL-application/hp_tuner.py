import keras
import keras_tuner
import os
from config import config
from fileHandler import createOutputDir
from dataLoader import create_dataset
from training_callbacks import ringring_callbackplease

def build_model(hp):
    """Build CNN model with tunable hyperparameters"""
    
    # Input layer. Sets the input shape (dimensions) of the input data
    inputs = keras.Input(shape=(None, None, 3))  # Fixed input shape


    """
    IMAGE PREPROCESSING
    https://keras.io/api/layers/preprocessing_layers/
    """
    
    # Apply AutoContrast to the input image
    x = keras.layers.AutoContrast()(inputs)
    
    # Resize the input image to the target size
    x = keras.layers.Resizing(config.TARGET_SIZE[0], config.TARGET_SIZE[1])(x)

    
    """
    IMAGE AUGMENTATION
    https://keras.io/api/layers/preprocessing_layers/image_augmentation/
    """

    # Apply random augmentations
    x = keras.layers.RandAugment(seed=config.RANDOM_STATE)(x)
    
    # Rescale the image, so pixel values are in the range [0, 1]
    x = keras.layers.Rescaling(1.0 / 255)(x)

    """
    DATA NORMALIZATION
    https://keras.io/api/layers/normalization_layers/
    """

    norm_choice = hp.Choice("normalization", ["standard", "layer", "batch", "none"])
    if norm_choice == "standard":
        x = keras.layers.Normalization()(x)
    elif norm_choice == "layer":
        x = keras.layers.LayerNormalization()(x)
    elif norm_choice == "batch":
        x = keras.layers.BatchNormalization()(x)
    else:  # none
        x = x
    
    
    """
    CONVOLUTIONAL LAYERS
    https://keras.io/api/layers/convolution_layers/
    """
    
    # Tune the amount of convolutional layers
    n_conv_layers = hp.Int("Amount of convolutional layers", min_value=2, max_value=4, default=2)
    
    for i in range(n_conv_layers):
        
        # Tune the number of filters in each convolutional layer
        filters = hp.Int(f"Layer {i}: Amount of filters", min_value=32, max_value=256, step=32)
        activation_function = hp.Choice(f"Layer {i}: Activation function", ["relu", "leaky_relu", "selu", "tanh", "gelu"], default="relu")
        dropout_rate = hp.Float(f"Layer {i}: dropout rate", min_value=0.0, max_value=0.5, step=0.1)
        kernel_size = hp.Choice(f"Layer {i}: kernel size", [3, 5, 7])

        # Tune batch normalization
        if hp.Boolean(f"Layer {i}: Use batch normalization"):
            x = keras.layers.BatchNormalization()(x)

        # Convolutional layer
        x = keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = keras.layers.Activation(activation_function)(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    
    # Flatten before dense layers
    x = keras.layers.Flatten()(x)
    
    # Dense layers
    n_dense_layers = hp.Int("Amount of dense layers", min_value=1, max_value=3)
    
    for i in range(n_dense_layers):
        units = hp.Int(f"Dense layer {i}: Units", min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float(f"Dense layer {i}: dropout rate", min_value=0.0, max_value=0.5, step=0.1)
        activation_function = hp.Choice(f"Dense layer {i}: Activation function", ["relu", "selu"], default="relu")
        
        x = keras.layers.Dense(units)(x)
        x = keras.layers.Activation(activation_function)(x)
        
        if hp.Boolean(f"Dense layer {i}: Use batch normalization"):
            x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Dropout(dropout_rate)(x)
    
    outputs = keras.layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer_choice = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    # Load data using create_dataset
    train_ds, val_ds = create_dataset()
    
    # Create tuner
    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=50,
        directory=config.TUNER_PATH,
        project_name='rock_paper_scissors_tuning'
    )
    
    # Start the search
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease()
    )
    
    # Get and print best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    
    # Save best hyperparameters
    with open(config.BEST_HP_PATH, 'w') as f:
        for param, value in best_hp.values.items():
            f.write(f'{param}: {value}\n')
    
    # Build and save best model
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=ringring_callbackplease()
    )
    best_model.save(config.BEST_MODEL_PATH)

if __name__ == "__main__":
    main()
