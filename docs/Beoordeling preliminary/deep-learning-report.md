

![](images/image_1.png)

OCR Results:
HAN_ UNIVERSITY
OF APPLIED SCIENCES


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

Tabel 1 Functional requirements

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

Tabel 2 Technical requirements

# Data augmentation and preprocessing

Augmentatie: Keras augmentatie layers

Image preparation: Image normalisatie

Visualatie van preprocessing steps

| Assignment | 1. Increase your data setStart with your ML image setChoose data augmentation methodsApply these methods to create new imagesExplain why you chose these methods 2. Prepare images for CNNsChoose preprocessing steps to:- Make patterns clearer for CNNs, or- Reduce CNN complexityConsider these factors:- Image size- Color depth- Image enhancement-Normalization   Build a preprocessing pipeline  Explain each step in your pipeline   1. Test your preprocessingRun some images through your pipelineShow before and after examplesExplain how each step improves the images |
| --- | --- |
| Acceptance criteria | Data augmentation methods are used and explained  Preprocessing pipeline is implemented  Each preprocessing step is explained and justified |
| Size | Max 3 A4 |

## Dataset increasing

To get more data for the algorithm to learn we have used data augmentation. In the repo from Jeroen Veen there is a test “Augmentor.py” which we have used to expand our dataset from 255 picture to 5000 pictures. The following adaptions are made:

* Rotation 🡪 to make the algorithm less sensitive for the exact corner of the gesture
* Flipping horizontal 🡪 to create the possibility to use left and right hand gestures
* Shearing 🡪 to make the algorithm less sensitive for the exact distance to the camera for all parts of the hand
* Scaling 🡪 to make the algorithm less sensitive for the exact distance to the camera of the gesture
* Contrast and brightness adaptions 🡪 To handle different lighting conditions better.

## Prepare images for CNNs

Resize images to 24\*32px to speed up DL and reduce the number of pixels to process.

// Keras

# CNN architecture, training and validation

Autokeras voor het vinden van optimale model.

Terminologie uitleggen

Transfer learning uitwerken. 🡪 Zie p.375 in Hands-on machine learning. Hier staat een voorbeeld

[Mahdi\_Hassani83 | rock-paper-scissors-model | Kaggle](https://www.kaggle.com/models/mahdihassani83/rock-paper-scissors-model)

![Afbeelding met tekst, schermopname, software, nummer

Automatisch gegenereerde beschrijving](images/image_2.png)

OCR Results:
[INFO] Final training set size: 1500
[INFO] Validation set size: 51
[INFO] Creating CNN model with 3 classes
Model: "functional"
Layer (type)                                       Output Shape             Param #
input_layer (InputLayer)                           (None, None, None, 3)     0
conv2d (Conv2D)                                   (None, None, None, 32)   896
max_pooling2d (MaxPooling2D)                       (None, None, None, 32)   0
conv2d_1 (Conv2D)                                 (None, None, None, 64)   18,496
max_pooling2d_1 (MaxPooling2D)                     (None, None, None, 64)   0
conv2d_2 (Conv2D)                                 (None, None, None, 64)   36,928
global_average_pooling2d (GlobalAveragePooling2D) (None, 64)                0
dense (Dense)                                      (None, 64)                4,160
dropout (Dropout)                                  (None, 64)                0
dense_1 (Dense)                                    (None, 3)                 195
Total params: 60,675 (237.01 KB)
Trainable params: 60,675 (237.01 KB)
Non-trainable params: 0 (0.00 B)
1/30


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

| Assignment | 1. Select key code snippetsChoose important parts of your codeInclude snippets for:- Data preprocessing- Model creation- Training process- Evaluation methods 2. Explain your codeAdd comments to each snippetExplain what each part doesDescribe why you made specific coding choices 3. Show coding best practices Use clear variable namesStructure your code logicallyFollow Python style guidelines (PEP 8) |
| --- | --- |
| Acceptance criteria | Code snippets are provided for key parts of the project  Code quality is sufficient |

