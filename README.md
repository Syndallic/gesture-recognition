# Hand Gesture Recognition Using Background Elimination and a CNN

## Description

This is a fairly minor modification of a project by [Sparsha Saha](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network), who deserves full credit for the core of this project (and much of this README).
The project uses a CNN with a background elimination pre-processing technique to detect any static hand gesture the model is trained on. 
The background elimination algorithm very effectively extracts foreground objects from the webcam images and results in much better thresholding and segmentation than other methods like explicit lower/upper colour thresholding. More information about the algorithm can be found at the bottom of this README.

The main modifications so far are:
* Making the program run on my machine and webcam
* Hardcoding of classes has been reduced to make using custom datasets easier
* Reduced the amount of code in ModelTrainer.ipynb
* Custom dataset creation has been largely automated in the ContinuousGesturePredictor.py code (reducing the amount of duplicate code in the old [PalmReader.py](https://github.com/SparshaSaha/Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network/blob/master/PalmReader.py) module)
* Added my own dataset and trained the model to recognise 6 commands corresponding to the number of fingers held up on one right hand
* Added recalibration keybinding for the background elimination in case the background or lighting levels change

## Requirements

* Python 3
* Tensorflow
* TfLearn
* OpenCV (opencv-python)
* Numpy
* Pillow (PIL)
* Imutils

## File Description

[ContinuousGesturePredictor.py](https://github.com/Syndallic/gesture-recognition/blob/master/ContinuousGesturePredictor.py) : Running this file opens up your webcam and takes continuous frames of your hand and then predicts the class of your hand gesture in realtime. It can also be used to create a custom dataset automatically. 

[ModelTrainer.ipynb](https://github.com/Syndallic/gesture-recognition/blob/master/ModelTrainer.ipynb) : This is the model trainer file. Run this file if you want to retrain the model using your custom dataset.

## How to use

* Aim your webcam at a relatively blank background for best results. 

* Run the [ContinuousGesturePredictor.py](https://github.com/Syndallic/gesture-recognition/blob/master/ContinuousGesturePredictor.py) file and you will see a window named **Video Feed** appear on screen. 

* Wave your hand into the green box representing the ROI (Region-of-Interest) and a window named **Thresholded** will appear, showing what the program can see.

* Choose between predicting or recording. 

* At any time, press **"c"** to re-calibrate the background elimination if the background lighting changes

##### Prediction
* To begin real-time prediction, press **"s"**. 

##### Recording
* If you want to create a custom dataset instead, press **"r"**.

* Enter the name of the new class as prompted by the console. 

* Press **"r"** again when ready to begin recording new images. 

## Some key architectural insights into the project

### Background Elimination Algorithm

OpenCV is used to take a running average of the background for 60 frames. This average is then used to subtract from subsequent frames to effectively isolate new foreground elements. 

The background elimination code is largely based on [Gogul09's](https://github.com/Gogul09) implementation.

He has written an awesome article on the problem and you can read it up [here](https://gogul09.github.io/software/hand-gesture-recognition-p1).

### The Deep Convolution Neural Network

The network contains **7** hidden convolution layers with **ReLU** as the activation function and **1** fully connected layers.

The network is trained across **50** iterations with a batch size of **64**.

The model achieved an accuracy of **96.6%** on Sparsha's validation dataset.

The ratio of training set to validation set is **1000 : 100**.
