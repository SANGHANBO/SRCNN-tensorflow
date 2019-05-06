SRCNN-tensorflow
================
This is a tensorflow implementation of image super-resolution using deep convolutional network.
To refer to the original website, please click [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

Requirement
-----------
* tensorflow
* h5py
* matlab

Experiment
----------
There are two input sets to train the model. One is 91 images in 'Train' directly read and processed under Python, and the other is pre-processed input image 
imported from MATLAB, which is genereted by generete_train.m and generete_test.m. You can choose which input data to use by following the annotate in SRCNN.py to 
modify the code. Make sure that you change the checkpoint in checkpoint/my_srcnn, because they are adapt to different model. <br>
I give two input sets because I find that the bicubic results under the MATLAB and Python structure are quite diffrent, with Python gets around 3dB lower than 
MATLAB, so the SRCNN results are also diffrent. However, in spite of lower PSNR with Python, the recovery effect of the image is quite the same. <br>
If you don't want to train by yourself, you can use the checkpoint to test your image with demo.py. In this way, I recommend you to use mode in Python mode, 
because I haven't written MATLAB code to pre-process input image. 