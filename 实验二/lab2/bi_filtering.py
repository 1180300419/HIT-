import numpy as np
import math
'''
双边滤波算法
参数是要处理的图片
待处理区域的半径
sigmas是像素权重的标准差
sigmar是位置权重的标准差
'''
'''
将一个直径划分成半径
'''

def bilateral_filter(pic, dim, sigmas, sigmar):
    pic_row, pic_col = pic.shape  # 获得图片的行和列
    tmp = pic.copy()

    add_row = dim
    add_col = dim
    ans = np.zeros((pic_row, pic_col))  # 用来存放最终的答案
    # 先对边界做处理
    for i in range(add_row):
        tmp = np.vstack((tmp, tmp[-1]))
        tmp = np.vstack((tmp[0, :], tmp))
    for i in range(add_col):
        tmp = np.c_[tmp, tmp[:, -1]]
        tmp = np.c_[tmp[:, 0], tmp]
    tmp = tmp.astype(np.int16)
    # 遍历图像中的每一个位置
    for i in range(pic_row):
        for j in range(pic_col):
            weight = 0
            sum = 0
            for x in range(i, i + 2 * dim + 1):
                for y in range(j, j + 2 * dim + 1):
                    Gs = np.exp(-((i + dim - x) ** 2 + (j + dim - y) **2) / (2 * sigmas ** 2))
                    Gr = np.exp(-((tmp[i,j] - tmp[x, y]) ** 2) / (2 * (sigmar ** 2)))
                    sum += Gr * Gs * tmp[x, y]
                    weight += Gr * Gs
            ans[i, j] = sum / weight
    return np.uint8(ans)

