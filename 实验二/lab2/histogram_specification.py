''''''
'''
将pic1进行直方图规定化，使其直方图和pic2的直方图近似
'''
import numpy as np
from histogram_equalization import histogram_equalization


def histogram_specification(pic1, pic2):
    ans1 = histogram_equalization(pic1)
    ans2 = histogram_equalization(pic2)
    # 分别对pic1和pic2进行直方图均衡化
    pic_row, pic_col = pic2.shape  # 获得图片的行和列
    list2 = np.ones(256) * 256  # 将像素初值化为一个不可能的值，也就是最大值
    for i in range(pic_row):
        for j in range(pic_col):
            if pic2[i, j] < list2[ans2[i, j]]:
                list2[ans2[i, j]] = pic2[i, j]
    pic_row, pic_col = ans1.shape
    for i in range(pic_row):
        for j in range(pic_col):
            ans1[i, j] = list2[ans1[i, j]] if list2[ans1[i, j]] != 256 else ans1[i, j]

    return np.uint8(ans1.reshape(pic_row, pic_col))


##要寻找反映射






