# Gesture-Recognition
In this project, we are going to build a 3D Conv models and Conv2D+LSTM/GRU models that will be able to predict the 5 gestures correctly.

## Table of Contents
* [General Information](#general-information)
* [Understanding the Dataset](#understanding-the-dataset)
* [Architecture](#architecture)
* [Methods Used](#methods-used)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Model Statistics](#model-statistics)
* [Acknowledgements](#acknowledgements)

### General Information

A home electronics company which manufactures state of the art smart televisions want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote. 

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

 - Thumbs up:  Increase the volume
 - Thumbs down: Decrease the volume
 - Left swipe: 'Jump' backwards 10 seconds
 - Right swipe: 'Jump' forward 10 seconds  
 - Stop: Pause the movie

### Understanding the Dataset

The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

The data is in a folder which is a zip file (https://drive.google.com/drive/folders/18q3ImiHjhjzk2UN1PzkW3q1wSbnIaNIH?usp=sharing). The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders where each subfolder represents a video of a particular gesture. Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos).


### Architecture
Two types of architectures are suggested for analyzing videos using deep learning: 

1. CNN + RNN architecture - The conv2D network will extract a feature vector for each image, and a sequence of these feature vectors is then fed to an RNN-based network. The output of the RNN is a regular softmax (CLASSIFICATION ALWAYS USES SOFTMAX).

2. CON3D(3D CONVOLUTIONAL NETWORKS) -3D convolutions are a natural extension to the 2D convolutions you are already familiar with.Just like in 2D conv, you move the filter in two directions (x and y), in 3D conv, you move the filter in three directions (x, y and z). In this case, the input to a 3D conv is a video (which is a sequence of 30 RGB images). If we assume that the shape of each image is 100x100x3, for example, the video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels.
  
### Methods Used

We used Keras API with TensorFlow backend.

- loading required libraries.
- connecting the data to our notebook.
- Reading the Images folder
- Preprocessing the Images - Crop, Resize, Standardize.
- performing data augumentation if needed.
- Choosing Appropriate Batch size, Number of epochs,Image size, Number of required frames in a video.
- Creating Generator Function.
- Applying diffirent models to check which gives the best training and validation accuracy.

# Model Statistics
- Model 1 :
  number of epochs=50
  Training Accuracy=0.99
  Validation Accuracy=0.51
  The model is clearly overfitting as the training and validation accuracy have a huge difference ie…overfitting, re-introducing Dropout layers.


- Model 2 :
  number of epochs=50
  Training Accuracy=0.97
  Validation Accuracy=0.56
  After using the dropouts of 0.2 and 0.5 after each layer we can still see that the data is overfitting.so now we try to implement a new architecture. And the accuracy is reduced from 99% to 97%.

- Model 3 :
  number of epochs=50
  Training Accuracy=0.99
  Validation Accuracy=0.59
  Using the time distributed and dense layer architecture the trainable and non-trainable parameters have decreased but it had a good training accuracy and not validation so model overfits we will use diff architecture.


- Model 4 :
  number of epochs=50
  Training Accuracy=0.98
  Validation Accuracy=0.56
  So when compared to above architecture we can observe that the total no.of parameters have been reduced though the accuracy is slightly different and which is over fitting too so lets try using LSTM architecture


- Model 5 :
  number of epochs=50
  batch_size=8
  Training Accuracy=0.92
  Validation Accuracy=0.93
  Choosing the time dense+LSTM is giving us a good accuracy of about 92% of train and 93% on validation hence its not over fitting and we can consider it as our final model.

### Conclusion

timedistributed+LSTM2D  model gave Training accuracy of 0.92 and Validation Accuracy of 0.93 which is close and indicates a good trained model. Since training accuracy and validation have almost the same accuracy therefore the model doesnt overfit.Hence, we are choosing timedistributed+LSTM2D to be the final model.

## Technologies Used

- matplotlib - version 3.5.1
- pandas - version 1.4.2
- numpy - version 1.21.5
- plotly - version 5.6.0
- tensorflow - version 2.16.1

## Acknowledgements
Greatful to upgrad for giving a chance to work on this project.

## Contact
Created by [@tejaswini1968] - feel free to contact me!















