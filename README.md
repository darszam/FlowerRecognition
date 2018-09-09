# Flower Recognition

## About
Creating and using Convolutional Neural Network to recognize flower by its picture. Using tensorflow, keras and python.

## Table of Contents
- [Flower Recognition](#flower-recognition)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Project scripts](#project-scripts)
    + [Creating CNN](#creating-cnn)
    + [Learning CNN](#learning-cnn)
    + [Predicting] (#predicting)

## Project Scripts
I divided project into 3 parts: First creating cnn, defining its layers, importing packages etc. Second in which cnn is being taught to flower dataset. And third part predicting where I use picture from the internet and asking cnn to classify it to the right class.

### Creating CNN
This script contains many most of the imports of the project. I'm defining then sequence of layers in the flower recognizer model. General structure of my CNN is: few layers of Conv2D, 1 Flatten layer, 1 Dense layer, and 1 output layer which is another Dense layer. Two of conv2D layers have strides to shorten the time of learning cnn. I was thinking about adding dropout layer but I think it would decrease overall accuracy of the model.

### Learning CNN
In this script I'm managing whereabouts of the flower images dataset and using it to teach cnn model. Teaching process takes ~1 hour so I save model to the .h5 file

### Predicting
Last script contains loading weights of the saved model in case we don't want to wait and teaching&using new model. Next step is getting picture of flower we want to classify. After scaling operations we can run predict function of our model. Prediction result is array in order - daisy, dandelion, rose, sunflower, tulip. 

This is the result of predicting purple rose image.
<img src="result.png" alt="Classifying flower"/>


