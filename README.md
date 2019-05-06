SRCNN-tensorflow
================
This is a tensorflow implementation of image super-resolution using deep convolutional network.
To refer to the original website, please click [here.](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

Requirement
-----------
* tensorflow
* h5py
* matlab

Experiment
----------
There are two input sets to train the model. One is 91 images in 'Train' directly read and processed under Python, and the other is pre-processed input image 
imported from MATLAB, which is genereted by generete_train.m and generete_test.m. You can choose which input data to use by following the annotate in SRCNN.py 
to modify the code. Make sure that you change the checkpoint in checkpoint/my_srcnn, because they are adapt to different model. <br>
Two input sets are given because I find that the bicubic results under the MATLAB and Python structure are quite diffrent, with Python gets around 3dB lower 
than MATLAB, so the SRCNN results are also diffrent. However, in spite of lower PSNR with Python, the recovery effect of the image is quite the same. <br>
If you don't want to train by yourself, you can use the checkpoint to test your image with demo.py. In this way, I recommend you to use mode in Python mode, 
because I haven't written MATLAB code to pre-process input image.

Result
------ 
With train_image imported from MATLAB, the average PSNR is 32.34dB, while 32.39 in paper. So my model works not bad. One sample is as follow.

Original woman image: <br>
![image](https//github.com/SANGHANBO/SRCNN-tensorflow/blob/master/sample/orig.png) <br>
Bicubic interpolation: <br>
![image](https//github.com/SANGHANBO/SRCNN-tensorflow/blob/master/sample/cubic.png) <br> 
srcnn super-resolved image: <br>
![image](https//github.com/SANGHANBO/SRCNN-tensorflow/blob/master/sample/srcnn.png) <br>

Additionally, I notice that although the model with train_image generated in Python gains an average PSNR of around 29dB, 3dB lower than the above, but the 
super-resolved effect is quite the same as it. In this case, I regard that the differ of the result is just the cause of different bottom structure in Python 
and MATLAB, and it's not a matter to apply the model in Python, although it gains a lower PSNR.

Learning process
----------------
This is the first time for me to build a deep CNN network with tensorflow, so it is absolutely a challenge for me. Fortunately, I make it and harvest a lot. 
As a milestone for me, I record my learning step here. <br><br>
First, as I'm new to tensorflow, it's a must to learn the structure of it. From my view, it's a rather hard tackle compared to similar ones, such as Keras.
It takes me about two days to follow the instruction in the Tensorflow Community and make quite clear about it. However, the summary and visualization in 
tensorboard are still a puzzle to me, and it's my further work. <br><br>
Then, I begin to do pre-process of train image to make the input, such as interpolation and cropping sub-image from the origin. After that, I build my network. 
I think pre-process of data is of great importance, and building model is relatively smooth. It takes three days altogether. <br><br>
Lastly, I run my model. To speed the training, I use GPU on Google Colaborator. This step consumes quite a lot time, because the result is 3dB lower than the 
paper. After many trials on both Python and MATLAB, I find that the output of interpolation under these two structures are different, and I guess that this is 
just the cause. So I generate training data from MATLAB, store them in h5py file which can be read in Python. The output gets a better PSNR, similar with the 
paper. However, I find that although the values differ with 3dB, they have equal performance on super-resolution.

Weakness and further work
-------------------------
There are many drawbacks with SRCNN:
* Speed of convergence is quite slow, especially in the end of training. It takes two days to train a model on GPU!
* The performance of the model relies heavily on pre-process method, in other word, the way of image blurry. I feed a test image intepolated by a factor of 2 to 
my model trained on scale 3, and its PSNR is rather low. So one pre-trained model can only cope with one type of blurry.

Reference
---------
[Dong C.,Loy C.C.,He K.,Tang X.:Learning a Deep Convolutional Network for Image Super-Resolution.](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) <br>
I refer to its original code and use the code to generate train and test input in MATLAB.