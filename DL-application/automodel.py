import numpy as np
from config import config
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.callbacks import EarlyStopping

epochs = config.EPOCHS

# Reduce verbosity of TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
tf.get_logger().setLevel('ERROR')

batch_size = config.BATCH_SIZE
random_seed = config.RANDOM_STATE

dataset_path = pathlib.Path(config.DATASET_ROOT_DIR)
output_path = pathlib.Path(config.OUTPUT_DIRECTORY)

image_res = PIL.Image.open(list(dataset_path.glob("*/*.png"))[0])

ds_train = ak.image_dataset_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    image_size=config.TARGET_SIZE,
    shuffle=True,
    seed=random_seed,
    interpolation="bilinear",
    validation_split=config.VALIDATION_SPLIT,
    subset="training"
)

ds_val = ak.image_dataset_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=config.TARGET_SIZE,
    shuffle=True,
    seed=random_seed,
    interpolation="bilinear",
    validation_split=config.VALIDATION_SPLIT,
    subset="validation"
)

AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    block_type="vanilla",
    augment=True,
    normalize=True
    )(input_node)
output_node = ak.ClassificationHead()(output_node)

model = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    directory=output_path,
    objective="val_loss",
    seed=random_seed,
    overwrite=False,
    max_trials=100
)

# Create early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=2
)

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=epochs,
    callbacks=[early_stopping]
)

try:
    # Export model
    export_path = output_path / "exported_model"
    model.export_model()
    print(f"\nModel successfully exported to: {export_path}")
    
    # Evaluate model
    evaluation = model.evaluate(ds_val, verbose=1)
    print(f"\nTest loss: {evaluation[0]:.4f}")
    print(f"Test accuracy: {evaluation[1]:.4f}")
except Exception as e:
    print(f"\nError during model export or evaluation: {str(e)}")
finally:
    # Clean up TensorFlow session
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
