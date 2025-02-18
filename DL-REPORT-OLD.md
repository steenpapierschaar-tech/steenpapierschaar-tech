

![](images/image_1.png)

<details>
<summary>AI Image Analysis</summary>

[AI-Generated Image Description]

HAN_ UNIVERSITY
OF APPLIED SCIENCES
</details>


*By submitting this portfolio the authors certify that this is their original work, and they have cited all the referenced materials properly.*

Deep Learning report

Koen Derksen, 2106875

JeRoen Wijnands, 1462623

Jaap-Jan Groenendijk, 1548148

Minor: Embedded VIsion and Machine Learning

Group: 10

Date: 30-11-2024

# INSTRUCTIONS FOR PROJECT submission

**When submitting your project results, please follow these steps:**

1. **Rename this file**
   * **Replace {YOUR GROUP\_NUMBER} with your group number**
   * **Replace {YOUR\_NAME} with your name**
   * **Replace {YOUR\_STUDENT\_NUMBER} with your student number**
2. **Delete this instruction page**
3. **Prepare the project results:**
   * **Report as a Word file**
   * **code in a zip file**
   * **deployment video**
4. **Per group, email the project results to the instructor**
5. **Upload your project results to HAND-IN**
   * **Each student must upload INDIVIDUALLY**
   * **Use your renamed file**
   * **PLEASE DO NOT UPLOAD YOUR DATA AND MODEL.**

Contents

[1 Introduction 3](#_Toc88129256)

[2 Problem statement 4](#_Toc88129257)

[3 Data augmentation and preprocessing 5](#_Toc88129258)

[4 CNN architecture and training 6](#_Toc88129259)

[5 Deploy and test 7](#_Toc88129260)

[6 Conclusion 8](#_Toc88129261)

[7 References 9](#_Toc88129262)

[Code appendices 10](#_Toc88129263)

# Introduction

| Assignment | Introduce your DL portfolio  Explain how it fits in the minor program  Describe the importance of DL in your areas of interest  Show how DL relates to your main project in the minor  List your learning objectives |
| --- | --- |
| Acceptance criteria | DL relation to the minor is discussed.  DL portfolio relation to main project in the minor is discussed. |
| Size | Max 1 A4 |

As part of the Embedded Vision and Machine Learning module, our portfolio will focus on deep learning (DL) applications for object identification and form recognition, which aligns directly with our project work. Although we do not have prior ML projects in the portfolio, this course serves as an introduction to the tools and methodologies used in the field, such as OpenCV. While this program is a module rather than a minor for our study group, machine learning plays a critical role in our studies. The course emphasizes hands-on applications of machine learning for computer vision, which is directly relevant to our work with embedded systems. The focus on image processing and object classification aligns well with our project objectives, allowing us to develop practical solutions for real-world vision systems.

Our project group focuses on machine learning’s role in object detection and form identification. With the increasing demand for intelligent vision systems across industries, ML techniques for object recognition are essential in fields such as automation, robotics, and smart devices. The course’s hands-on approach will help us build critical skills that can be applied to embedded systems, where real-time image recognition is often required.

Our main project centers around using machine learning to identify objects and forms. By leveraging the computer vision techniques covered in the course, we aim to develop a system that efficiently detects and classifies various objects. This practical application of ML is key to achieving our project’s goals and deploying these models in real-world environments.

This document is a continuation of the previous machine learning report and specifically focuses on deep learning. The previous machine learning project will serve as the foundation for this continuation, where we will further explore and expand the concepts and techniques applied in the earlier project to more advanced deep learning methods. Deep learning, as a subset of machine learning, offers more powerful techniques for complex tasks such as object recognition and image processing. We will apply these techniques to improve and refine the performance of our system. The goal is to leverage and deepen the insights from our previous project by using more advanced models that are better suited for real-time applications in embedded vision systems.

Given the nature of part-time studies and the need to balance jobs alongside academic commitments, our primary goal is to complete this project as efficiently as possible. While we aim to gain a foundational understanding of model training and performance analysis, the focus will be on applying these skills practically within the available time to finish the project.

# Problem statement

| Assignment | 1. Update the objective stated in your Machine Learning (ML) portfolio 2. Also update the requirements table and prioritize these requirements. Think about how you would test or prove whether your final result has met a requirement. |
| --- | --- |
| Acceptance criteria | Problem definition is specific and measurable [1].  Functional and technical requirements are listed and prioritized. |
| Size | Max 1 A4 |

## Problem statement

This project aims to create a deep learning model that can recognize hand gestures for Rock, Paper, and Scissors in real-time. The challenge is to make this work well using only a Raspberry Pi 4 and Pi camera, which have limited power. The solution will use deep learning methods to achieve reliable gesture recognition.

## Goal

The goal is to develop a model that can accurately recognize Rock, Paper and Scissors gestures. This will involve training the model using supervised learning and a simple neural network so it can recognize gestures quickly and accurately.

## Requirements

To keep the project focused, a list of requirements has been established. These requirements have been approved by the stakeholders after an alignment session.

### Functional requirements

Functional requirements specify the core functionalities needed for accurate and user-friendly gesture recognition.

| NR | Priority (MoSCoW) | Description |
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

| ID | Priority (MoSCoW) | Description |
| --- | --- | --- |
| TR01 | Must | The application runs on a laptop running Python3 |
| TR02 | Must | The application should accurately recognize gestures when the hand is positioned 15-50 cm away from the camera. |
| TR03 | Must | Software is written in Python 3 |
| TR04 | Must | Image segmentation should be implemented in Python 3 using the OpenCV library to isolate hand gestures effectively. |
| TR05 | Must | Deep learning script is done in Python3 using TensorFlow & Keras |
| TR06 | Should | The framerate of the application will be running on 25 fps or higher. |
| TR07 | Should | Give Results in graphs by using the python3 library Sci-kit |

Tabel Technical requirements

# Data augmentation and preprocessing

| Assignment | 1. Increase your data setStart with your ML image setChoose data augmentation methodsApply these methods to create new imagesExplain why you chose these methods 2. Prepare images for CNNsChoose preprocessing steps to:- Make patterns clearer for CNNs, or- Reduce CNN complexityConsider these factors:- Image size- Color depth- Image enhancement-Normalization   Build a preprocessing pipeline  Explain each step in your pipeline   1. Test your preprocessingRun some images through your pipelineShow before and after examplesExplain how each step improves the images |
| --- | --- |
| Acceptance criteria | Data augmentation methods are used and explained  Preprocessing pipeline is implemented  Each preprocessing step is explained and justified |
| Size | Max 3 A4 |

## Dataset increasing

Data augmentation is a technique used to improve machine learning models when the training dataset is small. By creating variations of existing images—such as rotating, flipping, or adjusting brightness—it helps the model learn general patterns instead of memorizing specific examples. This reduces *overfitting*, a problem where models perform poorly on new, unseen data

### Key Details About the Dataset

The original dataset contains 255 images (85 each for "rock," "paper," and "scissors") captured with a Raspberry Pi camera under controlled lighting against a white background. Each image is 240×320 pixels in RGB format and stored as a JPEG file. The dataset is split into 80% for training (about 204 images) and 20% for validation.

### Why Data Augmentation is Needed

A small dataset poses challenges. For instance, the model might struggle with images taken from new angles, lighting conditions, or hand positions not present in the training data. Without augmentation, the model could memorize specific details of the limited images instead of learning to recognize broader patterns, leading to poor real-world performance.

### How Data Augmentation Works

**Spatial Transformations** mimic different camera angles or hand positions. This includes rotating images slightly left or right (±10%), flipping them horizontally or vertically to handle left/right gestures or upside-down views, and shifting the hand position within the frame through translation. Zooming (90–110%) simulates varying distances between the hand and camera, while shear adjustments (±10%) alter perspective to replicate tilted views.

**Color and Lighting Adjustments** account for environmental or camera differences. Brightness and contrast are tweaked (±10%) to adapt to dim lighting or glare. Saturation and hue changes reflect variations in color temperature (e.g., warm vs. cool lighting), and sharpness adjustments (±10%) mimic focus differences or motion blur. These modifications ensure the model can handle real-world scenarios with inconsistent lighting or camera settings.

### Implementation Steps

The augmentation process is applied in real-time during training using TensorFlow/Keras preprocessing layers. Instead of generating and storing thousands of altered images, random transformations—such as flips, rotations, or color shifts—are applied each time the model processes a batch of images. This approach ensures the model encounters unique variations across training epochs (10 epochs total, with 32 images per batch), maximizing data diversity without increasing storage needs.

### Benefits of Data Augmentation

By expanding the dataset virtually, augmentation helps the model learn robust features. It reduces overfitting, improves adaptability to new environments (e.g., different hand gestures or lighting), and speeds up training. Resizing images to 96×128 pixels (84% smaller than the original 240×320) further enhances efficiency by reducing memory usage and computation time.

### Image Preprocessing Pipeline

Before training, images undergo several preparation steps. First, they are resized to 96×128 pixels to balance clarity and efficiency. Color information (RGB) is retained to preserve details that grayscale might lose. Next, automated contrast adjustments and random transformations (flips, rotations, zooms) are applied. Brightness, contrast, and saturation are varied by ±20% to simulate real-world conditions, and sharpness is tweaked (±10%) to mimic focus changes. Finally, pixel values are normalized to a 0–1 range, and batch normalization stabilizes training by standardizing inputs across batches.

Function apply\_augmentation can be found in appendice: Image preprocessing pipeline.

## Before/After Examples

Below you can find images showcasing the augmentation process. The augmentation code is written to easily adapt to different datasets, and can be found in file DL-Application/augmentation.py.

![A screenshot of a computer

AI-generated content may be incorrect.](images/image_2.png)

<details>
<summary>AI Image Analysis</summary>

[AI-Generated Image Description]

[INFO] Final training set size: 1500
[INFO] Validation set size: 51
[INFO] Creating CNN model with 3 classes
Model: "functional"
Layer (type)                                      Output Shape             Param #
input_layer (InputLayer)                         (None, None, None, 3)     0
conv2d (Conv2D)                                 (None, None, None, 32)    896
max_pooling2d (MaxPooling2D)                     (None, None, None, 32)    0
conv2d_1 (Conv2D)                                (None, None, None, 64)    18,496
max_pooling2d_1 (MaxPooling2D)                   (None, None, None, 64)    0
conv2d_2 (Conv2D)                                (None, None, None, 64)    36,928
global_average_pooling2d (GlobalAveragePooling2D) (None, 64)                0
dense (Dense)                                    (None, 64)                4,160
dropout (Dropout)                                (None, 64)                0
dense_1 (Dense)                                  (None, 3)                 195
Total params: 60,675 (237.01 KB)
Trainable params: 60,675 (237.01 KB)
Non-trainable params: 0 (0.00 B)
1/30
</details>


![A close-up of a hand

AI-generated content may be incorrect.](images/image_3.png)

<details>
<summary>AI Image Analysis</summary>

[AI-Generated Image Description]

Augmentation Preview - Press → for next, ← for previous, ESC to quit
Original Image 11/32
Augmented Image
(x, y) = (218.1, 123.2)
[164, 93, 91]
</details>


## AutoML

As an experiment, the AutoML solution AutoKeras was chosen to see how an automated CNN setup would perform against our own model. AutoKeras is a framework around Keras, which in itself is a framework around TensorFlow. AutoKeras makes it easy to perform automated hyperparameter tuning and searches for the most optimal neural network architecture. The goal of using an AutoML solution was to easily benchmark our self-created model against an automated solution.

AutoKeras is setup to find the CNN architecture and parameters with the lowest validation loss, while validating against an external dataset. From website Kaggle.com, the dataset called “Rock-Paper-Scissors Images” was used for validation. Scores from validating against an external dataset are better for testing the generalization of the model than using a subset of the self-created dataset.

Using AutoML and an external dataset, it is possible to get an indication of how much augmentation is needed to for a model to perform well (and generalize!) when tested on external data.

After training the AutoML solution on the external dataset for 1 hour on an Apple M1 device, the AutoML solution indicated a small benefit to using 0.1 change for spatial and color transformations. This value was also applied to the custom hyperparameter tuning model that’s made with Keras Tuner.

## Benefits and Outcome

The preprocessing pipeline reduces image size from 230,400 pixels (240×320) to 36,864 pixels (96×128), cutting processing time by 84% . Tests with 99 sample images confirmed the system’s accuracy, and live previews verified that transformations preserve critical details. Smaller image dimensions also reduce computational load—for example, the first layer of the neural network processes 193,000 fewer calculations per image.

**Summary**

Data augmentation addresses the limitations of small datasets by artificially expanding training data through realistic transformations. Spatial adjustments (rotation, flips, zoom) and color/lighting variations help models adapt to diverse real-world conditions. Combined with efficient preprocessing—resizing, normalization, and in-memory transformations—this approach enhances model performance while minimizing computational costs. The result is a robust system capable of accurately classifying rock-paper-scissors gestures even in unpredictable environments.

## Dataset increasing

To get more data for the algorithm to learn we have used data augmentation. In the repo from Jeroen Veen there is a test “Augmentor.py” which we have used to expand our dataset from 255 picture to 5000 pictures. The following adaptions are made:

* Rotation 🡪 to make the algorithm less sensitive for the exact corner of the gesture
* Flipping horizontal 🡪 to create the possibility to use left and right hand gestures
* Shearing 🡪 to make the algorithm less sensitive for the exact distance to the camera for all parts of the hand
* Scaling 🡪 to make the algorithm less sensitive for the exact distance to the camera of the gesture
* Contrast and brightness adaptions 🡪 To handle different lighting conditions better.

## Prepare images for CNNs

Resize images to 24\*32px to speed up DL and reduce the number of pixels to process.

# CNN architecture, training and validation

For finding the optimal CNN architecture, 3 different strategies have been explored. A manual CNN setup, an AutoML solution using Auto Keras and a semi manual model from Keras-Tuner. Each strategy has its pros and cons, which will be explained in the following paragraphs.

## Manual CNN architecture

## AutoML solution with AutoKeras

## Keras Tuner

Autokeras voor het vinden van optimale model.

Terminologie uitleggen

Transfer learning uitwerken. 🡪 Zie p.375 in Hands-on machine learning. Hier staat een voorbeeld

[Mahdi\_Hassani83 | rock-paper-scissors-model | Kaggle](https://www.kaggle.com/models/mahdihassani83/rock-paper-scissors-model)

![Afbeelding met tekst, schermopname, software, nummer

Automatisch gegenereerde beschrijving](images/image_4.png)

<details>
<summary>AI Image Analysis</summary>

[AI-Generated Image Description]

Augmentation Preview - Press → for next, ← for previous, ESC to quit
Original Image 6/32
Augmented Image
</details>


* + 1. Decide number of layers

For a rock-paper-scissors recognition task, which is relatively simple compared to other computer vision problems:

* **3 to 6 convolutional layers** is usually sufficient for high accuracy. This is because the features (hand gestures) in this task are not as complex as those in tasks like object detection or natural image classification.
* **Suggested CNN Architecture**

1. **Input Layer**: Input images resized to a fixed size (e.g., 64x64 or 128x128 pixels).
2. **Convolutional Layers**:
   * 2 to 4 convolutional layers with filters of increasing size (e.g., 16, 32, 64).
   * Use a **3x3 kernel size** with **ReLU activation**.
   * Apply **batch normalization** after each convolutional layer to stabilize training.
3. **Pooling Layers**: Add a **max-pooling layer** (e.g., 2x2) after every 1 or 2 convolutional layers to reduce dimensionality and computational load.
4. **Dropout Layers**: Use dropout (e.g., 0.25 to 0.5) to prevent overfitting, especially if the dataset is small.
5. **Fully Connected Layers**:
   * One or two dense layers with units (e.g., 128, 64) and ReLU activation.
   * Use a **softmax layer** at the end to output probabilities for each class (rock, paper, scissors).
6. **Output Layer**: 3 neurons (one for each class).

| Assignment | 1. Design CNN architecture and/or use transfer learningDesign a custom CNN: - Decide number of layers - Choose number of neurons per layer - Select pooling methods - Pick activation functionsUse transfer learning: - Select a pretrained model- Decide which layers to freeze - Design new top layers for your specific taskExplain each choice in your design 2. Train and optimize CNNChoose appropriate performance measuresConsider accuracy, precision, recall, F1-score and explainTrain your CNN using the training setUse cross-validation to check performanceOptimize hyperparameters, e.g.: try different learning rates, adjust batch sizes, experiment with optimizer types   Prevent overfitting (explain which method you used and why)   1. Test your modelUse the test set to check model performanceCreate confusion matrixCompute performance measuresCheck for overfitting or underfittingDiscuss trade-offs, e.g. precision vs. recall or bias vs. variance 2. Visualize CNN learningShow how an input image transforms through your networkVisualize at least 2 different layersExplain what features each layer detects |
| --- | --- |
| Acceptance criteria | Architecture is designed and argued.  Data is split into stratified subsets and checked.  CNN is trained, cross-validated, and fine-tuned.  Performance is evaluated using appropriate methods.  Visualization of network's internal representations is provided |
| Size | Max 5 A4 |

# Deploy and test

| Assignment | 1. Deploy your CNN model Set up preprocessing and prediction pipelineChoose where to run your model:- On your target machine (e.g., Raspberry Pi)- Or on your training machine 2. Make a test planReview your SMART problem definitionList requirements to measure, such as:- Model performance (e.g., accuracy, precision)- Inference speed (e.g., frame rate)- Technical factors (e.g., camera angles, distances, lighting conditions)Set target levels for each measureExplain how you will test each measure 3. Conduct testsRun tests based on your planRecord all test resultsCompare results to your targetsNote any unexpected behaviors or limitations 4. Document your workWrite down your test planRecord all test resultsExplain any differences between results and targets |
| --- | --- |
| Acceptance criteria | Preprocessing and prediction pipeline deployed.  Test plan present.  Documentation of test results. |
| Size | Max 5 A4 |

# Conclusion

| Assignment | 1. Summarize your projectList the main steps you tookExplain your key decisions 2. Evaluate your resultsCompare your results to your initial goalsDiscuss if you met your SMART objectivesExplain any differences between goals and results 3. Reflect on generalization performanceDiscuss how well your model works on new, unseen dataCompare performance on training, validation, and test setsExplain any differences in performance across these sets 4. Analyze your approachIdentify what worked wellPoint out areas for improvementSuggest changes for future projects |
| --- | --- |
| Acceptance criteria | Results are compared to initial goals and SMART objectives  Generalization performance is analyzed |
| Size | Max 1 A4 |

# References

| Assignment | Give references to the sources that you have used. |
| --- | --- |

| [1] | „SMART criteria,” 14 05 2020. [Online]. Available: https://en.wikipedia.org/wiki/SMART\_criteria. |
| --- | --- |
| [2] | A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, Sebastopol, Canada.: O’Reilly Media, 2019. |

# Code appendices

## Image preprocessing pipeline

| def apply\_augmentation(inputs):  x = inputs    # 1. Initial preprocessing  # AutoContrast normalizes the contrast across the image  # value\_range=(0, 255) specifies the input image range  x = keras.layers.AutoContrast(value\_range=(0, 255))(x)    # Randomly flip image horizontally and/or vertically  x = keras.layers.RandomFlip(mode="horizontal\_and\_vertical")(x)    # Randomly rotate image by up to ±RANDOM\_ROTATION \* 360 degrees  x = keras.layers.RandomRotation(factor=config.RANDOM\_ROTATION)(x)    # Apply random shearing transformation  # Shear factor of 0.2 means up to 20% shear in each direction  x = keras.layers.RandomShear(  x\_factor=config.RANDOM\_SHEAR\_X,  y\_factor=config.RANDOM\_SHEAR\_Y  )(x)    # Randomly translate image in height and width  # Factor of 0.2 means up to 20% translation in any direction  x = keras.layers.RandomTranslation(  height\_factor=config.RANDOM\_TRANSLATION,  width\_factor=config.RANDOM\_TRANSLATION  )(x)    # Randomly zoom in/out  # Factor of 0.2 means zoom range of 80% to 120%  x = keras.layers.RandomZoom(  height\_factor=config.RANDOM\_ZOOM,  width\_factor=config.RANDOM\_ZOOM  )(x)    # 3. Color augmentations  # Randomly adjust color properties while maintaining natural look  x = keras.layers.RandomColorJitter(  value\_range=(0, 255),  brightness\_factor=config.RANDOM\_BRIGHTNESS,  contrast\_factor=config.RANDOM\_CONTRAST,  saturation\_factor=config.RANDOM\_SATURATION,  hue\_factor=config.RANDOM\_HUE,  )(x)    # Randomly adjust image sharpness  # Factor of 0.2 means sharpness range of 80% to 120%  x = keras.layers.RandomSharpness(  factor=config.RANDOM\_SHARPNESS  )(x)    # 4. Final preprocessing  # Resize to target dimensions for model input  x = keras.layers.Resizing(config.TARGET\_SIZE[0], config.TARGET\_SIZE[1])(x)  # Normalize pixel values to 0-1 range  x = keras.layers.Rescaling(1.0 / 255)(x)    return x |
| --- |

| Assignment | 1. Select key code snippetsChoose important parts of your codeInclude snippets for:- Data preprocessing- Model creation- Training process- Evaluation methods 2. Explain your codeAdd comments to each snippetExplain what each part doesDescribe why you made specific coding choices 3. Show coding best practices Use clear variable namesStructure your code logicallyFollow Python style guidelines (PEP 8) |
| --- | --- |
| Acceptance criteria | Code snippets are provided for key parts of the project  Code quality is sufficient |

