import numpy as np
import cv2

def sobel(pic, ths):
    pic_row, pic_col = pic.shape  ##获得图片的行和列
    tmp = pic.copy()
    ans = np.zeros((pic_row, pic_col))  ##存放最终答案的矩阵
    add_row = 1
    add_col = 1
    ##先对边界做处理
    for i in range(add_row):
        tmp = np.vstack((tmp, tmp[-1]))
        tmp = np.vstack((tmp[0, :], tmp))
    for i in range(add_col):
        tmp = np.c_[tmp, tmp[:, -1]]
        tmp = np.c_[tmp[:, 0], tmp]
    ##遍历图像，提取图像的边缘
    Gx = np.zeros((pic_row, pic_col))
    Gy = np.zeros((pic_row, pic_col))
    for i in range(pic_row):
        for j in range(pic_col):
            Gx[i, j] = tmp[i][j] + 2 * tmp[i][j + 1] + tmp[i][j + 2]\
                         -tmp[i + 2][j] - 2 * tmp[i + 2][j + 1] - tmp[i + 2][j + 2]
            Gy[i, j] = tmp[i][j] + 2 * tmp[i +1][j] + tmp[i + 2][j]\
                   -tmp[i][j + 2] - 2 * tmp[i + 1][j + 2] - tmp[i + 2][j + 2]
            des = np.abs(Gx[i, j]) + np.abs(Gy[i, j])
            if des > ths:
                ans[i][j] = des
            else:
                Gx[i, j] = 0
                Gy[i, j] = 0
    return np.uint8(ans), Gx, Gy

def sharpen_pic(pic, sharpen_pic, pro):
    pic_row, pic_col = pic.shape ##获得图像的航和列
    ans = (pic + pro * sharpen_pic) / (1 + pro) #存储最终的答案
    ans = ans.flatten()
    ans[ans > 255] = 255
    ans = ans.reshape((pic_row, pic_col))
    return np.uint8(ans)

from smooth_pic import *

from load_pic import *

def canny(pic):
    gaussian = generated_gassian_filter_template(5, 1)
    new_gray = juanji(pic, gaussian)
    pic_row, pic_col = new_gray.shape
    Gx = np.zeros([pic_row - 1, pic_col - 1])
    Gy = np.zeros([pic_row - 1, pic_col - 1])
    d = np.zeros([pic_row - 1, pic_col - 1])
    for i in range(1, pic_row - 1):
        for j in range(1, pic_col - 1):
            Gx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            Gy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(Gx[i, j]**2+Gy[i, j]**2)
    pic_row, pic_col = Gx.shape
    ans = np.zeros([pic_row, pic_col])
    for i in range(1, pic_row - 1):
        for j in range(1, pic_col - 1):
            if Gx[i, j] == 0 and Gy[i, j] == 0:
                ans[i, j] = 0
            else:
                if np.abs(Gy[i, j]) > np.abs(Gx[i, j]):
                    weight = np.abs(Gx[i, j]) / np.abs(Gy[i, j])
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if Gx[i, j] * Gy[i, j] > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]
                # 如果X方向幅度值较大
                else:
                    weight = np.abs(Gy[i, j]) / np.abs(Gx[i, j])
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if Gx[i, j] * Gy[i, j] > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if d[i, j] >= gradTemp1 and d[i, j] >= gradTemp2:
                    ans[i, j] = d[i, j]
                else:
                    ans[i, j] = 0
    W3, H3 = ans.shape
    DT = np.zeros([W3, H3])
    TL = 0.2 * np.max(ans)
    TH = 0.3 * np.max(ans)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if (ans[i, j] < TL):
                DT[i, j] = 0
            elif (ans[i, j] > TH):
                DT[i, j] = 1
            elif ((ans[i - 1, j - 1:j + 1] < TH).any() or (ans[i + 1, j - 1:j + 1]).any()
                  or (ans[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    cv2.imshow('canny', DT)
    cv2.waitKey()

def canny_official(image):
    blurred = cv2.GaussianBlur(image, (3,3), 0) ##生成3*3的高斯核
    # 求X方向上的梯度
    grad_x = cv2.Sobel(blurred, cv2.CV_16SC1, 1, 0)
    # 求y方向上的梯度
    grad_y = cv2.Sobel(blurred, cv2.CV_16SC1, 0, 1)
    # 将梯度值转化到8位上来
    x_grad = cv2.convertScaleAbs(grad_x)
    y_grad = cv2.convertScaleAbs(grad_y)
    # 将两个梯度组合起来
    src1 = cv2.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)
    # 组合梯度用canny算法，其中50和100为阈值
    edge = cv2.Canny(src1, 50, 100)
    cv2.imshow("Canny_official_1", edge)
    edge1 = cv2.Canny(grad_x, grad_y, 10, 100)
    cv2.imshow("Canny_official_2", edge1)
    cv2.waitKey()


def Krisch(pic, th):
    pic_row, pic_col = pic.shape
    ans = np.zeros([pic_row - 1, pic_col - 1])
    for i in range(1, pic_row - 1):
        for j in range(1, pic_col - 1):
            A0 = pic[i - 1, j - 1]
            A1 = pic[i - 1, j]
            A2 = pic[i - 1, j + 1]
            A3 = pic[i, j + 1]
            A4 = pic[i + 1, j + 1]
            A5 = pic[i + 1, j]
            A6 = pic[i + 1, j - 1]
            A7 = pic[i, j - 1]
            tmp_list = []
            tmp1 = 5 * (A0 + A1 + A2) - 3 * (A3 + A4 + A5 + A6 + A7) * 1.0
            tmp_list.append(tmp1)
            tmp2 = 5 * (A1 + A2 + A3) - 3 * (A4 + A5 + A6 + A7 + A0)
            tmp_list.append(tmp2)
            tmp3 = 5 * (A2 + A3 + A4) - 3 * (A5 + A6 + A7 + A0 + A1)
            tmp_list.append(tmp3)
            tmp4 = 5 * (A3 + A4 + A5) - 3 * (A6 + A7 + A0 + A1 + A2)
            tmp_list.append(tmp4)
            tmp5 = 5 * (A4 + A5 + A6) - 3 * (A7 + A0 + A1 + A2 + A3)
            tmp_list.append(tmp5)
            tmp6 = 5 * (A5 + A6 + A7) - 3 * (A0 + A1 + A2 + A3 + A4)
            tmp_list.append(tmp6)
            tmp7 = 5 * (A6 + A7 + A0) - 3 * (A1 + A2 + A3 + A4 + A5)
            tmp_list.append(tmp7)
            tmp8 = 5 * (A7 + A0 + A1) - 3 * (A2 + A3 + A4 + A5 + A6)
            tmp_list.append(tmp8)
            tmp = np.max(tmp_list)
            if tmp > th:
                ans[i, j] = tmp
            if tmp > 255:
                ans[i, j] = 255
    return np.uint8(ans)

