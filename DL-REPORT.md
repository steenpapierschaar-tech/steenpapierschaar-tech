# 1. Introduction

As part of the Embedded Vision and Machine Learning module, this portfolio focuses on deep learning (DL) applications for object identification and form recognition, which aligns directly with the project work. While there are no prior ML projects in the portfolio, this course serves as an introduction to the tools and methodologies used in the field, such as OpenCV. Machine learning plays a critical role in the studies. The course emphasizes hands-on applications of machine learning for computer vision, which is directly relevant to work with embedded systems. The focus on image processing and object classification aligns well with the project objectives, allowing the development of practical solutions for real-world vision systems.

The project group focuses on machine learning’s role in object detection and form identification. With the increasing demand for intelligent vision systems across industries, ML techniques for object recognition are essential in fields such as automation, robotics, and smart devices. The course’s hands-on approach helps build critical skills that can be applied to embedded systems, where real-time image recognition is often required.

The main project centers around using machine learning to identify objects and forms. By leveraging the computer vision techniques covered in the course, the aim is to develop a system that efficiently detects and classifies various objects. This practical application of ML is key to achieving the project’s goals and deploying these models in real-world environments.

This document is a continuation of a previous machine learning report and specifically focuses on deep learning. The previous machine learning project serves as the foundation for this continuation, where the concepts and techniques applied in the earlier project are further explored and expanded to more advanced deep learning methods. Deep learning, as a subset of machine learning, offers more powerful techniques for complex tasks such as object recognition and image processing. These techniques are applied to improve and refine the performance of our system. The goal is to leverage and deepen the insights from the previous project by using more advanced models that are better suited for real-time applications in embedded vision systems.

Given the nature of part-time studies and the need to balance jobs alongside academic commitments, the primary goal is to complete this project as efficiently as possible. While a foundational understanding of model training and performance analysis is aimed for, the focus is on applying these skills practically within the available time to finish the project.

## 2. Problem Statement

This project aims to create a deep learning model that can recognize hand gestures for Rock, Paper, and Scissors in real-time. The challenge is to make this work well using only a Raspberry Pi 4 and Pi camera, which have limited power. The solution uses deep learning methods to achieve reliable gesture recognition.

The goal is to develop a model that can accurately recognize Rock, Paper and Scissors gestures. This involves training the model using supervised learning and a simple neural network so it can recognize gestures quickly and accurately.

To keep the project focused, a list of requirements has been established. These requirements have been approved by the stakeholders after an alignment session.

### Functional requirements

Functional requirements specify the core functionalities needed for accurate and user-friendly gesture recognition.

| NR   | Priority (MoSCoW) | Description |
| --- | --- | --- |
| FR01 | Must | Can recognize the gestures paper, rock and scissors |
| FR02 | Must | The captured image should have uniform lighting to improve gesture detection accuracy |
| FR03 | Must | The application shows the contour of the hand with the name of the gesture |
| FR04 | Must | The application displays the probability score for each identified gesture, indicating the confidence level of the classification. |
| FR05 | Must | The application recognizes gestures on a white background |
| FR06 | Must | The application detects if there is no gesture |
| FR07 | Should | The application should have an accuracy of minimal 90% |

Tabel Functional requirements

### Non-functional requirements

Non-functional requirements detail the technical specifications and performance criteria for the application’s operation on a Raspberry Pi 4.

| ID   | Priority (MoSCoW) | Description |
| --- | --- | --- |
| TR01 | Must | The application runs on a laptop running Python3 |
| TR02 | Must | The application should accurately recognize gestures when the hand is positioned 15-50 cm away from the camera. |
| TR03 | Must | Software is written in Python 3 |
| TR04 | Must | Image segmentation should be implemented in Python 3 using the OpenCV library to isolate hand gestures effectively. |
| TR05 | Must | Deep learning script is done in Python3 using TensorFlow & Keras |
| TR06 | Should | The framerate of the application will be running on 25 fps or higher. |
| TR07 | Should | Give Results in graphs by using the python3 library Sci-kit |

Tabel Technical requirements

## 3. Data Augmentation and Preprocessing

The data augmentation methods applied to the dataset include:

*   **Spatial Transformations:** Randomly flips images horizontally and/or vertically, rotates images, shears images, translates images, and zooms in/out. These transformations are implemented in the `apply_augmentation` function in `src/augmentation.py`.
*   **Color Adjustments:** Randomly adjusts color properties such as brightness, contrast, saturation, and sharpness. These adjustments are also implemented in the `apply_augmentation` function in `src/augmentation.py`.
*   **Normalization:** Pixel values are normalized to the range \[0, 1] using the `Rescaling` layer.

The `apply_augmentation` function, defined in `src/augmentation.py`, implements the image augmentation pipeline. This pipeline includes:

*   `AutoContrast`: Normalizes the contrast across the image.
*   `RandomFlip`: Randomly flips images horizontally and/or vertically.
*   `RandomRotation`: Randomly rotates images.
*   `RandomShear`: Applies random shearing transformations.
*   `RandomTranslation`: Randomly translates images.
*   `RandomZoom`: Randomly zooms in/out.
*   `RandomColorJitter`: Randomly adjusts color properties.
*   `RandomSharpness`: Randomly adjusts image sharpness.
*   `Resizing`: Resizes images to the target dimensions.
*   `Rescaling`: Normalizes pixel values to the range \[0, 1].

In `hp_tuner.py`, different normalization strategies can be chosen as hyperparameters, including "standard", "layer", "batch", and "none".

## 4. CNN Architecture, Training, and Validation

In this chapter, the design, training approach, and validation procedures for the Convolutional Neural Network (CNN) are presented. Three strategies were explored in this project, each offering a unique way of constructing and refining the CNN: (1) a manually designed architecture, (2) a semi-automatic approach with hyperparameter tuning, and (3) an automatic CNN creation method.

## 4.1 Manual CNN Creation

The first approach was a custom CNN architecture defined in `application.py`. The model consists of three convolutional layers. Each convolutional layer uses:

*   224 filters
*   7x7 kernel sizes
*   Leaky ReLU activation function
*   Batch normalization for stable training
*   MaxPooling2D with a 2x2 pool size

Following the convolutional layers, a Flatten layer is used to convert the 3D feature maps to a 1D feature vector. This is followed by a Dense layer of 96 units with Leaky ReLU activation. A dropout rate of 0.1 is applied before the final output layer to prevent overfitting. The final output layer has three units and softmax activation, which is suitable for a 3-class classification problem.

The model was compiled using the AdamW optimizer with a learning rate of 0.00040288 and categorical crossentropy as the loss function. The model was trained for a set number of epochs, with accuracy monitored as a primary metric.

## 4.2 Semi-Automatic CNN with Hyperparameter Tuning

In the second strategy, defined in `hp_tuner.py`, Keras Tuner's Bayesian Optimization was used to explore a range of architectural and hyperparameter settings. The search space included:

*   Number of convolutional layers (1 or 2)
*   Various activation functions (e.g., ReLU, LeakyReLU, SELU, tanh, GELU)
*   Different optimizers (Adam, RMSprop, SGD, AdamW)
*   Learning rate schedules (constant, step, exponential, polynomial, cosine)
*   Dropout rates (0.0 to 0.5), regularization methods (l1, l2, l1_l2), and pooling strategies (maximum, average)

This approach systematically searches for an optimal configuration. It reduces the need for manual experimentation and speeds up the process of discovering a performant model. Early stopping is used to prevent overfitting by monitoring the validation loss and halting training when it stops improving.

## 4.3 Automatic CNN Creation

The third strategy involves creating a more automated solution. The goal is to reduce manual overhead by programmatically generating architectures. This approach is implemented in `automodel.py`. The `automodel.py` script uses AutoKeras to automatically search for an optimal CNN architecture. It includes:

*   Image augmentation (horizontal and vertical flips)
*   Normalization layer
*   Convolutional block (AutoKeras searches for the optimal CNN architecture)
*   Classification head

The AutoKeras model automatically searches for the best architecture and hyperparameters.

## 4.4 Performance Metrics

The performance of each model is evaluated using the following metrics:

*   **Accuracy:** The percentage of correctly classified samples.
*   **Loss:** The categorical crossentropy loss, which measures the difference between the predicted and actual class distributions.

These metrics are monitored during training using TensorBoard and are used to evaluate the performance of the models on the validation set.

## 4.5 Training and Validation

Regardless of the chosen strategy, the training workflow includes:

*   Splitting the dataset into training and validation subsets using the `create_dataset()` function. The training dataset is created from the 'photoDataset' directory, and the validation dataset is created from the 'photoDataset_external' directory.
*   Applying data augmentation techniques (e.g., horizontal and vertical flips) to improve generalization.
*   Training with callbacks for:
    *   `EarlyStopping`: This callback stops the training process when the validation loss stops improving, preventing overfitting.
    *   `TensorBoard`: This callback visualizes training metrics in real time, allowing for monitoring of the model's performance.
    *   `CSVLogger`: This callback logs training metrics to a CSV file, providing a record of the training process.
    *   `TimeoutCallback`: This callback stops the training process if an epoch exceeds a specified time limit, preventing excessively long training times.
*   Evaluating the trained model on the validation set and comparing metrics like validation accuracy and loss over epochs. The models are evaluated using the `evaluate` function, which returns metrics such as accuracy and loss. The validation accuracy and loss are monitored during training.

In the following sections, training results, confusion matrices, and feature map visualizations will be presented to assess each approach's performance and determine where refinements may be needed. [UNCLEAR: Results, confusion matrices, feature map visualizations]

## 5. Deployment and Testing

### Deployment Setup

The model was initially intended to be deployed on a Raspberry Pi 4 using a Pi camera. However, due to time constraints, it was decided to deploy the model using a Flask web application with SocketIO for real-time communication instead. The application is located in the `model_testing/` directory. The model is loaded from the `model_testing/model/current_model.keras` file.

The preprocessing and prediction pipeline works as follows:

1.  The application receives frames from the webcam via SocketIO.
2.  The application preprocesses the frames by resizing them to the model's input shape.
3.  The preprocessed frames are fed to the model for prediction.
4.  The model outputs a probability score for each gesture (rock, paper, scissors).
5.  The application sends the predicted gesture and its probability score back to the client via SocketIO.

[UNCLEAR: Specific details on how these tests will be conducted and how the results will be measured]
[UNCLEAR: Test results and comparison to targets. Also, any unexpected behaviors or limitations]

## 6. Conclusion

[UNCLEAR: Summarize the main steps and key decisions in your project. Did your results meet your SMART objectives? If not, why? How well does your model generalize to unseen data (training vs. test performance)? What worked well, and what would you improve in future iterations?]

## 7. References

[1] „SMART criteria,” 14 05 2020. [Online]. Available: https://en.wikipedia.org/wiki/SMART\_criteria.

[2] A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, Sebastopol, Canada.: O’Reilly Media, 2019.

## 8. Code Appendices

### Data Preprocessing (from `template.py`)

```python
x = keras.layers.Rescaling(1.0 / 255)(x)
x = keras.layers.Normalization()(x)
```

### Model Architecture (from `application.py`)

```python
model = keras.Sequential()
model.add(layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='same',
                        kernel_regularizer=regularizers.l2(0.0001633),
                        input_shape=(config.IMAGE_ROWS, config.IMAGE_COLS, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
```

### Training Loop (from `application.py`)

```python
model.fit(
    train_ds,
    epochs=config.EPOCHS,
    validation_data=val_ds,
)
```

### Evaluation Methods (from `automodel.py`)

```python
eval = auto_model.evaluate(val_ds)
print(eval)
```

### Data Preprocessing (from `template.py`)

```python
x = keras.layers.Rescaling(1.0 / 255)(x)
x = keras.layers.Normalization()(x)
```

### Model Architecture (from `application.py`)

```python
model = keras.Sequential()
model.add(layers.Conv2D(224, (7, 7), activation='leaky_relu', padding='same',
                        kernel_regularizer=regularizers.l2(0.0001633),
                        input_shape=(config.IMAGE_ROWS, config.IMAGE_COLS, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
```

### Training Loop (from `application.py`)

```python
model.fit(
    train_ds,
    epochs=config.EPOCHS,
    validation_data=val_ds,
)
```

### Evaluation Methods (from `automodel.py`)

```python
eval = auto_model.evaluate(val_ds)
print(eval)
