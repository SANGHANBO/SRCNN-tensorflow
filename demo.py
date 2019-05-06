from inputdata import (
get_data,
crop,
preprocess
)
import tensorflow as tf
import scipy.misc
import scipy.ndimage
import math
import numpy as np
import glob
import os
import h5py

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer('image_number', 0, "The serial number of the image in test set")       # 希望处理的测试图片编号
FLAGS = flags.FLAGS

def imagecrop(image, padding):
    x, y = image.shape
    image = image[padding:x-padding, padding:y-padding]
    return image

def test_model(image):
    weight = {
        'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
        'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
        'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }
    bias = {
        'b1': tf.Variable(tf.zeros([64]), name='b1'),
        'b2': tf.Variable(tf.zeros([32]), name='b2'),
        'b3': tf.Variable(tf.zeros([1]), name='b3')
    }
    conv1 = tf.nn.relu(tf.nn.conv2d(image, weight['w1'], strides=[1, 1, 1, 1], padding='VALID') + bias['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight['w2'], strides=[1, 1, 1, 1], padding='VALID') + bias['b2'])
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weight['w3'], strides=[1, 1, 1, 1], padding='VALID') + bias['b3'])
    return conv3

def load(sess, saver, path):
    print('Loading...')
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Load Successfully')
        model = ckpt.model_checkpoint_path      # 读取path内的模型
        # model = tf.train.latest_checkpoint(path)    # 读取最后保存的模型
        saver.restore(sess, model)
    else:
        print('Fail to load')

def test(input_image, input_label):
    x, y = input_image.shape
    input_image = input_image.reshape(1, x, y, FLAGS.c_dim)
    x, y = input_label.shape
    input_label = input_label.reshape(1, x, y, FLAGS.c_dim)
    image = tf.placeholder(tf.float32, [1, None, None, FLAGS.c_dim], name='image')
    label = tf.placeholder(tf.float32, [1, None, None, FLAGS.c_dim], name='label')
    conv3 = test_model(image)
    loss = tf.reduce_mean(tf.square(conv3 - label))
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        loadpath = 'checkpoint/my_srcnn'
        load(sess, saver, loadpath)

        print('Testing...')
        test_loss, output = sess.run([loss, conv3], feed_dict={image: input_image, label: input_label})
        output = output.squeeze()
        # output[output < 0] = 0
        # output[output > 1] = 1
        scipy.misc.imsave('sample/srcnn.png', output)
        print('size of output:', output.shape)
        print('max value:', np.max(output))
        print('min value：', np.min(output))
        psnr = 10 * math.log(pow(1, 2)/test_loss, 10)
        print("test_loss: [%.8f], PSNR:[%3.3f]" % (test_loss, psnr))

'''
下面直接读取MATLAB处理得到的测试图片的内插矩阵，只进行'baby'一图的测试
若想读取Python处理所得的内插结果，请注释掉下面的代码，并恢复注释代码
'''
# data = glob.glob(os.path.join('checkpoint/testimage', '*.h5'))
# datapath = data[FLAGS.image_number]
# with h5py.File(datapath, 'r') as hf:
    # im_label = np.array(hf.get('origin'))
    # im_image = np.array(hf.get('bicubic'))

print(FLAGS.image_number)
datapath = 'Test/Set5'
data = glob.glob(os.path.join(datapath, '*.bmp'))
origin_image = scipy.misc.imread(data[FLAGS.image_number], flatten=True, mode='YCbCr')    # 原图
origin_image = np.array(origin_image, dtype=np.float)
# 预处理，获取加噪图像
im_image, im_label = preprocess(FLAGS, data[FLAGS.image_number])

scipy.misc.imsave('sample/orig.png', im_label)
scipy.misc.imsave('sample/cubic.png', im_image)
cubic_loss = np.mean(np.square(im_label - im_image))
cubic_pnsr = 10 * math.log(1/cubic_loss, 10)
print('cubic loss:', cubic_loss)
print('cubic psnr:', cubic_pnsr)

padding = 6
im_label = imagecrop(im_label, padding)
test(im_image, im_label)