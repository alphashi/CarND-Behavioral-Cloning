**Behavioral Cloning**
======================

This is my solution for the project 3 Behavioral Cloning of Self-Driving Car Nanodegree. The objective in this
project was to build a neural network model able to steer the car autonomously in the simulator. Udacity provided with
the training data - series of frames (views from left,right,center cameras) with corresponding steering angles collected
by completing a few laps in the simulator in train mode (using PS3 controller).

I tried experiments with quite complex models proposed by Nvidia and comma.ai.
It turned out that very simple convolutional neural network with only 63 parameters performed
very well despite of only 5 layers. The initial idea based on: <https://github.com/xslittlegrass/CarND-Behavioral-Cloning>
The model performed well because the simulator environment is very basic comparing to
the real world. There is not much varying in shadowed areas, surface of the road, light reflections etc.

<div align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=fYcwJP4lei4
" target="_blank"><img src="http://img.youtube.com/vi/fYcwJP4lei4/0.jpg"
alt="youtube link" width="480" height="360" border="10" /></a>
<br>
<br>
The model also works on the test track.
<br>
<br>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=xivMFiIsGMg
" target="_blank"><img src="http://img.youtube.com/vi/xivMFiIsGMg/0.jpg"
alt="youtube link" width="480" height="360" border="10" /></a>
</div>



### Model

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 16, 32, 1)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 14, 30, 2)     20          lambda_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 3, 7, 2)       0           convolution2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 3, 7, 2)       0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 42)            0           dropout_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             43          flatten_1[0][0]
====================================================================================================
Total params: 63
Trainable params: 63
Non-trainable params: 0
____________________________________________________________________________________________________

```
Layers:

+ **Lambda** - normalization layer.
+ **Convolution2D** - convolution with 3x3 kernel, 1x1 stride, padding valid and activation ELU.
+ **MaxPooling2D** - reducing dimensions.
+ **Dropout** - prevents overfiting.
+ **Flatten** - converting output of convolutional part of the CNN into a 1D feature vector.
+ **Dense** - regression output - the steering angle.

Small insight into the model:

![](https://github.com/michalfaber/CarND-Behavioral-Cloning/raw/master/model_visualization.gif)

Top row represents original image.
Bottom row contains 3 images, respectively original image, preprocessed image, weights at flatten layer

### Dataset

Images from center/left/right cameras have been used for training. The main purpose of the images form left/right
cameras is to learn model how to recover when the vehicle drives off the center the lane. Small value 0.3
has been added to the steering angle for left camera and subtracted form the steering angle for right camera.
This is quite inaccurate approximation because we don't know parameters of the vehicle (wheelbase, distance between cameras, etc.)
For the real word applications applying Ackerman geometry to calculate proper recover steering angles would give better results.

Also each image and corresponding steering angle has been flipped and added to the training set. This helps to balance data
because original training data contains more turns to the left.

Images have been converted to HSV color space and only the S channel is used

Images have been resized to this shape : (16, 32, 1)

All training set is stored in the memory so there is no need to use keras generator. I noticed that after
such significant size reduction all data fits nicely in memory, training is faster, code is cleaner.

### Training

Adam optimizer has been used with default configuration.

Model checkpoint and early stopping (20 - 25 epochs)

15% of data is used as a validation set.

Training on the GTX 1070, Athlon II X4, 8 GB RAM


