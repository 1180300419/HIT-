##
import numpy as np

##这个有点慢，能不能加快一些速度呢？
##使用向量化处理
def average_filtering(pic, template):
    pic_row, pic_col = pic.shape ##获得图片的行和列
    tmp = pic.copy()
    row, col = template.shape##获得模板的大小
    ans = np.zeros((pic_row, pic_col)) ##存放最终答案的矩阵
    add_row = (row - 1) // 2 ##表示要添加的行的数量
    add_col = (col - 1) // 2 ##表示要添加的列的数量
    ##先对边界做处理
    for i in range(add_row):
        tmp = np.vstack((tmp, tmp[-1]))
        tmp = np.vstack((tmp[0, :], tmp))
    for i in range(add_col):
        tmp = np.c_[tmp, tmp[:, -1]]
        tmp = np.c_[tmp[:, 0], tmp]
    ##遍历图像中的每一个位置
    for i in range(pic_row):
        for j in range(pic_col):
            ans[i][j] = np.round(np.sum(tmp[i:i+row, j:j+col] * template))

    return np.uint8(ans)

##分别表示要进行滤波处理的图片
##和滤波处理的核半径
def median_filter(pic, radius):
    pic_row, pic_col = pic.shape  ##获得图片的行和列
    tmp = pic.copy()
    ans = np.zeros((pic_row, pic_col))  ##存放最终答案的矩阵
    ##先对边界做处理
    for i in range(radius):
        tmp = np.vstack((tmp, tmp[-1]))
        tmp = np.vstack((tmp[0, :], tmp))
    for i in range(radius):
        tmp = np.c_[tmp, tmp[:, -1]]
        tmp = np.c_[tmp[:, 0], tmp]
    for i in range(pic_row):
        for j in range(pic_col):
            list = []
            for x in range(i, i + 2 * radius + 1):
                for y in range(j, j + 2 * radius + 1):
                    list.append(tmp[x][y])
            list.sort()
            ans[i][j] = list[len(list) // 2]
    return np.uint8(ans)

##生成高斯滤波模板
def generated_gassian_filter_template(dim=5, sigma=1.4):
    center = (dim - 1) // 2 ##得到中心
    ans = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            ans[i][j] = 1/(2 * np.pi * sigma**2) * np.exp(((i - center)**2 + (j - center)**2) / (-2 * sigma**2))
    ans = ans / np.sum(ans)
    return ans

##模板是归一化的，所以卷积之后不会溢出
##将pic和模板进行卷积操作
def juanji(pic, template):
    pic_row, pic_col = pic.shape  ##获得图片的行和列
    radius = (template.shape[0] - 1) // 2  ##获得模板的半径
    ans = np.zeros((pic_row - 5, pic_col - 5))  ##存放最终答案的矩阵
    ##遍历图像中的每一个位置
    for i in range(pic_row - 2 * radius - 1):
        for j in range(pic_col - 2 * radius - 1):
            ans[i][j] = np.sum(pic[i:i + 2 * radius + 1, j:j + 2 * radius + 1] * template)
    return ans

##罗伯特交叉梯度算子
def roberts(pic):
    pic_row, pic_col = pic.shape
    ans = np.zeros((pic_row - 1, pic_col - 1))
    for i in range(pic_row - 1):
        for j in range(pic_col - 1):
            tmp = np.sqrt((int(pic[i, j]) - int(pic[i + 1, j + 1]))**2 + (int(pic[i + 1, j]) - int(pic[i, j + 1]))**2)
            if tmp <= 255:
                ans[i, j] = tmp
            else:
                ans[i, j] = 255
    return np.uint8(ans)





