import matplotlib.pyplot as plt
import numpy as np
'''
直方图均衡化函数
绘制直方图函数
'''

'''
pic：要绘制直方图的图像
filename：保存相应的直方图时，使用的路径和文件名
'''
def plot_hist(pic, filename):
    ##绘制直方图函数
    x = np.arange(0, 256)
    plt.hist(pic.reshape(1, -1)[0], x, histtype='bar', color='black')

    plt.xlabel('gray value')
    plt.ylabel('number')
    plt.title(filename)

    plt.savefig(filename)  # 保存图片
    plt.show()  # 展示图片