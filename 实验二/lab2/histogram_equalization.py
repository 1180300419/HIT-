import numpy as np
import cv2
import matplotlib.pyplot as plt


'''
对pic表示的图片进行直方图均衡化，返回进行直方图均衡化之后的图像
'''
def histogram_equalization(pic):
    pic_row, pic_col = pic.shape ##获得图像的大小
    pic_ones = np.ones(pic_row * pic_col) ##获得一个和图像大小一样的1数组
    # 首先统计nj
    N = np.zeros(256)
    C = np.zeros(256) ##保存前面的像素的和
    pic_flatten = pic.flatten() ##将图像展开成一个向量

    for i in range(256):
        N[i] = np.sum(pic_ones[pic_flatten == i]) ##统计每个灰度值的像素点的个数

    ##计算每个像素值得点所占的概率
    P = N / (pic_row * pic_col)

    ##计算前面的累计概率
    C[0] = P[0]
    for i in range(1, 256):
        C[i] = C[i - 1] + P[i]

    for i in range(pic_row * pic_col):
        pic_flatten[i] = np.floor(255 * C[pic_flatten[i]] + 0.5)

    ans = np.uint8(pic_flatten.reshape(pic_row, pic_col))

    return ans



