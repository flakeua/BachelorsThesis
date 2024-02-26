# Eye-Gaze Estimation with Artificial Neural Networks

This repository contains the code and documentation for a bachelor's thesis on the topic of eye-gaze estimation using artificial neural networks (ANNs). The thesis explores the usage of ANNs to build a reliable eye-gaze estimation system for human-robot interaction (HRI) applications.

## Purpose

The purpose of this project is to investigate the feasibility of using ANNs for estimating eye gaze without the need for special hardware or infrared (IR) filters. By leveraging standard RGB cameras, the goal is to develop a system that can accurately predict where a person is looking, enabling more natural and intuitive human-robot interactions.

## Technologies Used

- **Convolutional Neural Networks (CNNs)**: CNNs are employed as the core technology for eye-gaze estimation, since they are one of the most successful neural network architectures when it comes to working with image data. Various architectures and configurations are explored to improve the accuracy and robustness of the models.

- **Unreal Engine and Metahuman Technology**: Unreal Engine and Metahuman technology are utilized for generating a diverse dataset for training the eye-gaze estimation models. These technologies enable access to high-resolution human models and dynamic lighting conditions, facilitating the creation of realistic and diverse training data.

- **RetinaFace**: RetinaFace is used for face detection, enabling the localization of faces in images or video frames. This is a crucial preprocessing step for extracting eye regions for gaze estimation.

- **SixDRepNet**: SixDRepNet is employed for head pose estimation, providing additional input data to improve the performance of the eye-gaze estimation models.

## Dataset

A diverse dataset for eye-gaze estimation is generated using Unreal Engine and Metahuman technology. This dataset includes variations in parameters such as age, race, eye color, and face shape, as well as different lighting conditions to enhance the generalization capabilities of the models. The dataset is publicly available at [link_to_dataset](https://cogsci.fmph.uniba.sk/metahuman/). It is compiled of over 57,000 images with 15 characters in different eye and head positions. The file names are built like this: "{character name}\_{id}\_OP\_{vertical eye gaze angle}\_{horizontal eye gaze angle}.png".

## Model Architecture

The neural network architectures used for eye-gaze estimation include various configurations of convolutional neural networks (CNNs) and fully connected layers. These architectures are designed to take input images of the eye regions along with head pose information and output the predicted gaze coordinates. Also, after various experiments and optimizations it was found, that giving head pose estimation from the SixDRepNet improves the eye gaze estimation results.

## Model Evaluation

The performance of the eye-gaze estimation models is evaluated on multiple datasets, including the Metahuman dataset, [Columbia Gaze dataset](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) and a combined dataset. Cross-validation across those datasets demonstrates the accuracy and robustness of the trained models across different scenarios and lighting conditions.

### Cross-Validation Results
![crossval](https://github.com/flakeua/BachelorsThesis/assets/26747964/5a8e4f51-df51-43c2-82da-fd7a27d23c6d)


## Conclusion

The thesis demonstrates the effectiveness of using ANNs for eye-gaze estimation in HRI applications. By leveraging advanced technologies such as Unreal Engine, Metahuman, and state-of-the-art neural network architectures, accurate and robust remote eye-gaze estimation systems can be developed without the need for specialized hardware. Further research could focus on improving the models' performance under challenging conditions such as poor lighting or occlusions.
