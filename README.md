**Rice Leaf Disease Detection using Deep Learning**

**Overview**

This project focuses on automated detection and classification of rice leaf diseases using deep learning. A convolutional neural network was trained on labeled rice leaf images to classify multiple bacterial and fungal diseases, along with healthy leaves . The goal is to assist in early disease identification using image-based analysis.
The implementation is done using TensorFlow/Keras , and the model supports both CPU and CUDA (GPU) execution.

<u>**Objective**</u>

This repository contains phase 1 part of our project . A group of six members worked in this project including me .In this phase ,all six of us just performed a comparative evaluation of six different CNN architectures where my role was to automatically identify and classify rice leaf diseases from leaf images using a customized AlexNet-based convolutional neural network.

Dataset:-

Dataset Name: Rice Leaf disease Dataset Original 

Platform: Google drive 

Dataset URL:- https://drive.google.com/drive/folders/1757tAm3XomvaDW4nX6l07kAp91jCYzzX
    
 The project uses a labeled rice leaf image dataset containing 8 classes of healthy and diseased rice leaves

Classes:-

- Bacterial Leaf Blight
- Brown Spot
- Healthy Rice Leaf
- Leaf Blast
- Leaf scald
- Narrow Brown Leaf Spot
- Rice Hispa
- Sheath Blight

- Dataset Distribution

| Dataset | Images | Classes |
|--------|-------:|--------:|
| Training | 1,701 |   8   |
| Validation | 186 |   8   |

The images are organized into class-specific folders so that TensorFlow can automatically assign class labels during dataset loading.

Dataset Structure

Rice leaf disease Dataset/

    ├── Training Data/
    │   ├── Bacterial Leaf Blight/
    |   ├── Brown Spot/
    │   ├── Healthy Rice Leaf/
    │   ├── Leaf Blast/
    │   ├── Leaf scald/
    │   ├── Narrow Brown Leaf Spot/
    │   ├── Rice Hispa/
    │   └── Sheath Blight/
    │
    └── Validation data/
       ├── Bacterial Leaf Blight/
       ├── Brown Spot/
       ├── Healthy Rice Leaf/
       ├── Leaf Blast/
       ├── Leaf scald/
       ├── Narrow Brown Leaf Spot/
       ├── Rice Hispa/
       └── Sheath Blight/

   
Model Architecture

The first phase uses a customized AlexNet-style convolutional neural network implemented with TensorFlow/Keras.

Architecture Components

Input image size: 227 × 227 × 3

Data augmentation

Pixel normalization using rescaling

5 convolutional layers

Batch Normalization

Max Pooling layers

Flatten layer

Two fully connected layers with 4096 neurons each

Dropout regularization with rate 0.5

Final Softmax classification layer

Output classes: 8

Convolutional Configuration

Input: 227 × 227 × 3


Data Augmentation
        ↓
Rescaling (1/255)
        ↓
Conv2D: 96 filters, 11×11 kernel, stride 4
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D: 256 filters, 5×5 kernel
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D: 384 filters, 3×3 kernel
        ↓
Conv2D: 384 filters, 3×3 kernel
        ↓
Conv2D: 256 filters, 3×3 kernel
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense: 4096
        ↓
Dropout: 0.5
        ↓
Dense: 4096
        ↓
Dropout: 0.5
        ↓
Softmax: 8 classes

Data Preprocessing and Augmentation

To expose the network to variations in leaf orientation and appearance, the training pipeline applies the following augmentation operations:

Random horizontal and vertical flipping
Random rotation
Random zoom
Pixel value rescaling to the range used by the network

The images are loaded directly from their class folders using TensorFlow's image_dataset_from_directory utility.

The dataset pipeline also uses prefetching with tf.data.AUTOTUNE to improve data-loading efficiency.

Training Configuration
Parameter	  Configuration
Framework	   TensorFlow / Keras
Input Size	    227 × 227
Batch Size	     32
Epochs	         30
Optimizer	     Adam
Learning Rate	 0.0001
Loss Function	 Sparse Categorical Crossentropy
Evaluation	      Validation Accuracy

Training was performed using a CUDA-enabled NVIDIA Tesla T4 GPU environment.

Performance

The model was trained for 30 epochs.



Final Epoch Results
Metric	               Result
Training Accuracy	    79.72%
Validation Accuracy	     69.35%

The validation set contains 186 images across the eight rice leaf classes.

Validation Classification Report

                        precision    recall  f1-score   support

    Bacterial Leaf Blight    0.79      0.75      0.77        20
            Brown Spot       0.63      0.70      0.67        27
     Healthy Rice Leaf       0.50      0.95      0.65        19
            Leaf Blast       0.78      0.90      0.84        31
            Leaf scald       0.86      0.52      0.65        23
    Narrow Brown Leaf Spot   1.00      0.31      0.48        16
            Rice Hispa       0.61      0.64      0.62        22
         Sheath Blight       0.78      0.64      0.71        28

              accuracy                           0.69       186
             macro avg       0.74      0.68      0.67       186
          weighted avg       0.74      0.69      0.69       186

Model Evaluation

The trained AlexNet model was evaluated on the validation dataset using:

1)Accuracy

2)Precision

3)Recall

4)F1-score

5)Confusion Matrix

The repository also includes visualization of training and validation accuracy and loss across the training epochs.

Saved Model

After training, the trained model was saved in Keras format:

           rice_leaf_alexnet_model.keras

This saved model is subsequently used in Phase 2 for probability extraction and ensemble prediction.

Requirements

Install the required Python packages before running the notebook:

     pip install tensorflow numpy matplotlib seaborn scikit-learn
GPU execution requires a compatible CUDA-enabled TensorFlow environment.

How to Run

1)Download or prepare the rice leaf dataset.

2)Arrange the images according to the documented directory structure.

3)Open the Phase 1 notebook.

4)Update the dataset paths according to your local environment.

5)Run the notebook cells sequentially.

6)Train the AlexNet model.

7)Evaluate the model using the validation dataset.

8)The trained model will be saved as:
            rice_leaf_alexnet_model.keras


Author

Subhajit Sarkar

Final Note

This repository represents the AlexNet phase of the rice leaf disease detection project.The trained model and its probability outputs are later utilized in the second phase, where AlexNet is combined with ResNet-18 through a weighted ensemble approach.
