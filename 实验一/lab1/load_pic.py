from PIL import Image
import numpy as np
import cv2
from add_noise import *

def loadImage(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    '''
    flags：读入图片的标志
    cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
    cv2.IMREAD_GRAYSCALE：读入灰度图片
    cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
    '''
    return img

