import numpy as np
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.callbacks import EarlyStopping


# Reduce verbosity of TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
tf.get_logger().setLevel('ERROR')

batch_size = 32
random_seed = 42

dataset_path = pathlib.Path("photoDataset")
output_path = pathlib.Path("output")

image_res = PIL.Image.open(list(dataset_path.glob("*/*.png"))[0])

ds_train = ak.image_dataset_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    image_size=(image_res.height, image_res.width),
    shuffle=True,
    seed=random_seed,
    interpolation="bilinear",
    validation_split=0.2,
    subset="training"
)

ds_val = ak.image_dataset_from_directory(
    directory=dataset_path,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=(image_res.height, image_res.width),
    shuffle=True,
    seed=random_seed,
    interpolation="bilinear",
    validation_split=0.2,
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
    max_trials=500
)

# Create early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=50,
    callbacks=[early_stopping]
)

model.export_model()

# Evaluate with proper dataset formatting
evaluation = model.evaluate(ds_val, verbose=1)
print(f"\nTest loss: {evaluation[0]:.4f}")
print(f"Test accuracy: {evaluation[1]:.4f}")