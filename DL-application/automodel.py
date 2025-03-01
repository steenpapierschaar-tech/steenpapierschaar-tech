import autokeras as ak
import keras
import tensorflow as tf
import os
from src.config import config
from src.create_dataset import apply_augmentation, rescale_dataset
from src.tensorboard import TensorboardLauncher
from src.training_callbacks import ringring_callbackplease

def main():

    # Enable mixed precision training for better performance on compatible GPUs
    # This allows some operations to use float16 instead of float32
    keras.mixed_precision.set_global_policy("mixed_float16")

    # Set global random seed for reproducibility
    # This ensures consistent results across runs
    keras.utils.set_random_seed(config.RANDOM_STATE)

    # Clear any existing Keras session to free up memory
    # This helps prevent memory leaks between training runs
    keras.utils.clear_session(free_memory=True)

    # Initialize and start TensorBoard for visualization
    # This allows monitoring of training metrics in real-time
    tensorboard = TensorboardLauncher()
    tensorboard.start_tensorboard()

    train_data = ak.image_dataset_from_directory(
        directory=config.DATASET_ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        seed=config.RANDOM_STATE,
        shuffle=True,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        image_size=config.TARGET_SIZE,
    )

    validation_data = ak.image_dataset_from_directory(
        directory=config.DATASET_ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        seed=config.RANDOM_STATE,
        shuffle=True,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        image_size=config.TARGET_SIZE,
    )

    # Apply augmentation and rescaling
    train_data = apply_augmentation(train_data)
    validation_data = rescale_dataset(validation_data)

    # Calculate steps before modifying datasets
    train_steps = len(train_data)
    validation_steps = len(validation_data)

    # Autotune the dataset for performance and add repeat
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)
    validation_data = validation_data.prefetch(buffer_size=AUTOTUNE)
    input = ak.ImageInput()
    output = ak.ImageBlock(
        block_type="vanilla",
    )(input)

    classification = ak.ClassificationHead(metrics=["accuracy", "precision", "recall"])(
        output
    )

    # Check if we should load existing model or train new one
    if config.SKIP_TRAINING_AUTO_KERAS:
        print("Loading existing AutoKeras model...")
        clf = keras.models.load_model(config.PATH_AUTO_KERAS_BEST)
        exported_model = clf
    else:
        print("Training new AutoKeras model...")
    # Define the AutoKeras image classifier
    clf = ak.AutoModel(
        inputs=input,
        outputs=classification,
        directory=config.DIR_AUTO_KERAS_MODEL,
        max_trials=config.MAX_TRIALS,
        seed=config.RANDOM_STATE,
        project_name="AutoKeras",
    )

    # Train the classifier
    clf.fit(
        x=train_data,
        validation_data=validation_data,
        callbacks=ringring_callbackplease(
            logs_dir=config.DIR_AUTO_KERAS_LOGS,
            csv_log_path=config.PATH_AUTO_KERAS_LOG,
            use_model_checkpoint=False,
            use_early_stopping=True,
            use_csv_logger=True,
            use_timeout=True,
            use_custom_callback=False,
            use_tensorboard=True
        ),
        epochs=config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
    )

    # Export and save the model
    exported_model = clf.export_model()
    exported_model.save(config.PATH_AUTO_KERAS_BEST)

    print("Model training complete.")

    # Evaluate the classifier
    test_data = ak.image_dataset_from_directory(
        directory=config.DATASET_EXTERNAL_DIR,
        batch_size=config.BATCH_SIZE,
        seed=config.RANDOM_STATE,
        shuffle=True,
        image_size=config.TARGET_SIZE,
    )

    print("Evaluating the model...")
    results = clf.evaluate(test_data)
    print("Model evaluation complete.")
    accuracy, precision, recall = results[:3]  # Unpack the metrics in order
    print("Test metrics:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")

if __name__ == "__main__":
    main()
