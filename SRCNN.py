from inputdata import (
    get_data,
    get_subimage,
    make_h5file,
    read_h5file
)
import tensorflow as tf
import numpy as np
import os
import time
import math

# 输入训练集和验证集
def input_data(config):
    train_data = get_data(datapath='Train')
    valid_data = get_data(datapath='Test')
    # 获取训练、验证集的sub_image和sub_label
    train_image, train_label = get_subimage(config, train_data)
    valid_image, valid_label = get_subimage(config, valid_data)
    # 制作h5py文件储存
    train_path = 'checkpoint/train.h5'
    valid_path = 'checkpoint/valid.h5'
    make_h5file(train_image, train_label, train_path)
    make_h5file(valid_image, valid_label, valid_path)

'''
下面开始SRCNN模型的构建
'''
def build_model(image):
    # 构建权重和偏置，'tf.random_normal'为高斯分布，'stddev'表示方差；均值0，方差1e-3
    weight = {
        'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),     # 改用tf.truncated_normal试试？
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

# 加载模型
def load_model(sess, path):
    print('Loading...')
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Load Successfully')
        model = tf.train.latest_checkpoint(path)
        saver.restore(sess, model)
    else:
        print('Fail to load')

'''
下面定义模型的训练过程
'''
def train(conv3):
    # 直接导入MATLAB中数据，并重组矩阵
    train_dir = 'checkpoint/predata/train.h5'
    valid_dir = 'checkpoint/predata/test.h5'
    train_image, train_label = read_h5file(train_dir)    # N*1*33*33，N*1*21*21
    m, n, p, q = train_image.shape
    train_image = train_image.reshape(m, p, q, n)
    m, n, p, q = train_label.shape
    train_label = train_label.reshape(m, p, q, n)
    valid_image, valid_label = read_h5file(valid_dir)
    m, n, p, q = valid_image.shape
    valid_image = valid_image.reshape(m, p, q, n)
    m, n, p, q = valid_label.shape
    valid_label = valid_label.reshape(m, p, q, n)
    '''
    上面的数据是Train和Test图片经MATLAB处理后导入的
    若希望通过Python读入图片，请注释掉上述代码，并开启input_data
    我的实验表明，MATLAB读取的图片内插后具有更小的损失
    '''
    # input_data(FLAGS)

    init = tf.initialize_all_variables()
    train_loss = tf.reduce_mean(tf.square(conv3 - label))  # 训练损失
    valid_loss = tf.reduce_mean(tf.square(conv3 - label))  # 验证损失
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(train_loss)

    with tf.Session() as sess:
        sess.run(init)
        # 建立日志
        # merged = tf.summary.merge_all()     # 将图形、训练过程等数据合并
        # writer = tf.summary.FileWriter('logs', sess.graph)
        startTime = time.time()
        loadpath = 'checkpoint/my_srcnn'
        savepath = 'checkpoint/my_srcnn/SRCNNmodel.ckpt'
        load_model(sess, loadpath)
        print('training...')
        train_batch_number = len(train_image) // FLAGS.batch_size  # // 表示整除

        max_psnr = 0
        for ep in range(FLAGS.epoch):
            for n in range(train_batch_number):
                train_batch_image = train_image[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size]    # batch_size*33*33*1
                train_batch_label = train_label[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size]    # batch_size*21*21*1
                # 运行图，计算损失
                _, train_loss_value = sess.run([train_op, train_loss],
                                               feed_dict={image: train_batch_image, label: train_batch_label})

            if ep % 10 == 0:
                valid_loss_value = sess.run(valid_loss, feed_dict={image: valid_image, label: valid_label})
                psnr = 10 * math.log(pow(1, 2)/valid_loss_value, 10)
                print("Epoch: [%2d], time: [%4.4f], train_loss: [%.8f], valid_loss: [%.8f], PSNR:[%3.3f]"
                      % ((ep+1), time.time()-startTime, train_loss_value, valid_loss_value, psnr))
            # writer.add_summary(tf.summary.scalar('train_loss', train_loss_value), ep)
            # writer.add_summary(tf.summary.scalar('PSNR', psnr), ep)

            # 保存验证集上psnr最大的模型
            if ep % 500 == 0:
                if psnr > max_psnr:
                    max_psnr = psnr
                    saver.save(sess, savepath, global_step=ep + 1)

'''
下面进行参数的输入和模型训练
'''
# flags添加了命令行的可选参数，若无输入则定义值；FLAGS是全局变量
flags = tf.flags
flags.DEFINE_integer('epoch', 2020, 'number of epoch [2020]')
flags.DEFINE_integer('batch_size', 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")          # 卷积核移动的步长
FLAGS = flags.FLAGS
# 创建目录
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
if not os.path.exists('sample'):
    os.makedirs('sample')
if not os.path.exists('checkpoint/my_srcnn'):
    os.makedirs('checkpoint/my_srcnn')

# 输入与占位符
image = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim], name='image')
label = tf.placeholder(tf.float32, [None, FLAGS.label_size, FLAGS.label_size, FLAGS.c_dim], name='label')
conv3 = build_model(image)
saver = tf.train.Saver()
train(conv3)