''''''
import numpy as np
'''
傅里叶变换
'''
def FFT(pic):
    pic_row, pic_col = pic.shape  # 获得图像的行和列，也就是获得了取样的个数
    ans = np.zeros(pic_row, pic_col)
    for u in range(pic_row):
        for v in range(pic_col):
            for x in range(pic_row):
                for y in range(pic_col):
                    ans[u, v] += pic[x, y] * np.exp(-1.j * 2 * np.pi * (u * x / pic_row + v * y / pic_col))
    return ans
'''
傅里叶逆变换
'''
def IFFT(pic):
    pic_row, pic_col = pic.shape  # 获得图像的行和列，也就是获得了取样的个数
    ans = np.zeros(pic_row, pic_col)
    for x in range(pic_row):
        for y in range(pic_col):
            for u in range(pic_row):
                for v in range(pic_col):
                    ans[x, y] += pic[u, v] * np.exp(1.j * 2 * np.pi * (u * x / pic_row + v * y / pic_col))
            ans[x, y] /= (pic_row * pic_col)
    return ans
'''
同态滤波器
输入是需要进行滤波的灰度图像
同态滤波器

输出经过同态滤波之后的图像
'''

def homomorphic_filtering(pic, rh, rl, c, D0):
    pic_row, pic_col = pic.shape  # 获得图像的行和列
    P = 2 * pic_row
    Q = 2 * pic_col  # 得到填充参数，避免混淆误差
    pic_p = np.zeros((P, Q))  # 填充后的图像
    # 1、取对数
    process_pic = np.log(pic + 1)  # 加一可以防止产生ln0的情况

    # 对图像填充0，并且移动到变换中心
    for i in range(pic_row):
        for j in range(pic_col):
            pic_p[i, j] = process_pic[i, j] * (-1) ** (i + j)

    # 2、进行傅里叶变换
    pic_f = np.fft.fft2(pic_p)

    # 3、使用合适的滤波器对图像进行滤波，希望可以压缩灰度动态范围，增强对比度

    # 首先生成滤波模板
    Homo = np.zeros((P, Q))
    const_D02 = D0 ** 2
    dr = rh - rl
    for u in range(P):
        for v in range(Q):
            dis = (u - pic_row) ** 2 + (v - pic_col) ** 2
            Homo[u, v] = dr * (1 - np.exp(-c * dis / const_D02)) + rl

    # 将模板与进行傅里叶变换之后的矩阵相乘
    G = pic_f * Homo

    # 4、取逆变换，将频域图像转换到空域中
    pic_log = np.fft.ifft2(G)

    # 5、取指数运算，得到同态滤波的系统输出
    ans = np.zeros((pic_row, pic_col))
    for x in range(pic_row):
        for y in range(pic_col):
            ans[x, y] = np.real(pic_log[x, y]) * (-1) ** (x + y)
    ans = np.exp(ans) - 1
    # 5、进行归一化处理，将范围变到 0-255
    Max = np.max(ans)
    Min = np.min(ans)
    tmp = Max - Min
    # ans = np.uint8(255 * (ans - Min) / tmp)
    for i in range(pic_row):
        for j in range(pic_col):
            ans[i, j] = np.uint8(255 * ((ans[i, j] - Min) / tmp))
    return ans