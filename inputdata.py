import numpy as np
import os
import glob
import h5py
import scipy.misc
import scipy.ndimage

'''
下面进行数据的输入和预处理
'''
# 读取路径中所有bmp图片
# 入口参数datapath：'Train'或'Test'，训练/测试集所在目录；出口参数：目录中所有bmp图片构成的列表
def get_data(datapath):
    if datapath == 'Train':
        data_dir = os.path.join(os.getcwd(), datapath)
        data = glob.glob(os.path.join(data_dir, '*.bmp'))
    elif datapath == 'Test':
        data_dir = os.path.join(os.path.join(os.getcwd(), datapath), 'Set5')        # 将Set5作为验证集
        data = glob.glob(os.path.join(data_dir, '*.bmp'))
    return data

# 将图片裁剪成scale的整数倍
# 入口参数：图片数组，灰度图维数为2，彩色为3；出口参数：裁剪后的图片数组
def crop(config, imagearray):
    dim = imagearray.shape  # 图片的维度
    if len(dim) == 3:
        x = dim[0] - np.mod(dim[0], config.scale)       # mod为取余数
        y = dim[1] - np.mod(dim[1], config.scale)
        imagearray = imagearray[0:x, 0:y, :]
    else:
        x = dim[0] - np.mod(dim[0], config.scale)
        y = dim[1] - np.mod(dim[1], config.scale)
        imagearray = imagearray[0:x, 0:y]
    return imagearray

# 归一化；插值
# 入口参数：YCbCr格式图片；出口参数：interpolate_image加噪声后图像(大小与原图相同），pre_image原图像
def preprocess(config, data):
    # 将亮度单通道（灰度）图片读取为数组；若需读取彩色图片，c_dim!=1
    if config.c_dim == 1:
        pre_image = np.array(scipy.misc.imread(data, flatten=True, mode='YCbCr'), dtype=np.float)
    else:
        pre_image = np.array(scipy.misc.imread(data, flatten=False, mode='YCbCr'), dtype=np.float)
    pre_image = crop(config, pre_image)         # 将pre_image裁剪为scale的整数倍
    pre_image = pre_image / 255         # 归一化
    interpolate_image = scipy.ndimage.interpolation.zoom(pre_image, (1./config.scale), prefilter=False)  # 下采样，缩小1/3
    interpolate_image = scipy.ndimage.interpolation.zoom(interpolate_image, config.scale/1., prefilter=False)    # 三次插值，放大3倍
    return interpolate_image, pre_image

# 将准备好的sub_image和sub_label以h5py形式存储
# 入口参数：两个数组，保存路径
def make_h5file(arr1, arr2, path):
    savepath = os.path.join(os.getcwd(), path)
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=arr1)
        hf.create_dataset('label', data=arr2)

# 读取checkpoint路径下的h5py文件，返回存储的所有sub_image和sub_label
# 入口参数：路径；出口参数：两个数组
def read_h5file(path):
    savepath = os.path.join(os.getcwd(), path)
    with h5py.File(savepath, 'r') as hf:
        arr1 = np.array(hf.get('data'))
        arr2 = np.array(hf.get('label'))
    return arr1, arr2

# 获取sub_image和sub_label
# 入口参数：bmp图像文件列表；出口参数：sub_image和sub_label构成的数组
def get_subimage(config, data):
    sub_image = []
    sub_label = []
    padding = int((config.image_size - config.label_size) / 2)
    for i in range(len(data)):
        input_image, input_label = preprocess(config, data[i])  # 获取加噪前后的图像（整体）
        dim = input_image.shape  # len(dim) == 2（灰度图） 或 3
        # 以步长stride==14获取子图
        # 由于layer不作padding，sub_image通过网络后的label只取image中间的21格，因此sub_label只需取中心的21*21
        for m in range(0, dim[0] - config.image_size + 1, config.stride):
            for n in range(0, dim[1] - config.image_size + 1, config.stride):
                sub_image_ = input_image[m:m + config.image_size, n:n + config.image_size]  # 33*33的image
                sub_label_ = input_label[m + padding:m + padding + config.label_size,
                             n + padding:n + padding + config.label_size]     # 21*21的label
                # train 时只用灰度图（亮度单通道）
                sub_image_ = sub_image_.reshape(config.image_size, config.image_size, 1)
                sub_label_ = sub_label_.reshape(config.label_size, config.label_size, 1)
                sub_image.append(sub_image_)
                sub_label.append(sub_label_)
    # 将列表转换成数组
    arrimage = np.asarray(sub_image)
    arrlabel = np.asarray(sub_label)
    return arrimage, arrlabel