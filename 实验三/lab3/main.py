'''
首先经过预处理
车牌定位
利用矩形将车牌圈起来

'''
import cv2
import numpy as np
'''
预处理函数
转换成灰度图
适当的改善图像质量
去噪声
进行边缘检测
改善图像质量
'''
def preprocessing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色图转换成灰度图
    histogram_equalization_img = cv2.equalizeHist(gray_img)  # 改善图像
    GaussianBlur_img = cv2.GaussianBlur(histogram_equalization_img, (3, 3), 0,0)  # 取出噪声
    # canny边缘检测,得到的边界比较细，效果反而不好
    Sobel_img = cv2.Sobel(GaussianBlur_img, -1, 1, 0, ksize=3)
    #canny_img = cv2.Canny(GaussianBlur_img, GaussianBlur_img.shape[0], GaussianBlur_img.shape[1])
    ret, binary_img = cv2.threshold(Sobel_img, (np.max(Sobel_img) + np.min(Sobel_img)) / 2, 255, cv2.THRESH_BINARY)

    # 闭合运算和开启运算结合，可以连接断开的部分，并且将一些不是块状的或者是比较小的部分去除
    # 进行闭运算
    kernel = np.ones((5, 19), np.uint8)
    closingimg = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # 进行开运算
    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)

    # 部分图像得到的轮廓边缘不整齐，因此在此进行开启运算
    kernel = np.ones((11, 5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)

    return openingimg

'''
定位车牌号
'''
def locate_license(img, afterimg):
    # 利用该函数来查找物体的轮廓
    # 该参数接收的参数是二值图
    # 第一个参数是寻找轮廓的图像
    # 第二个参数表示轮廓的检索模式：只检测外轮廓
    # 第三个参数表示只保留参数的四个点的坐标
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 测试边框识别结果

    #将上面找到的轮库变形成规矩的矩形
    block = []
    for c in contours:
        x = []
        y = []
        for p in c:
            y.append(p[0][0])
            x.append(p[0][1])
        r = [min(y), min(x), max(y), max(x)]  # 表示一个矩形
        a = (r[2] - r[0]) * (r[3] - r[1])  # 面积
        s = (r[2] - r[0]) * (r[3] - r[1])  # 长度比
        block.append([r, a, s])
    ## 只要面积最大的3个
    block = sorted(block, key=lambda b: b[1])[-3:]
    # 使用颜色识别找出车牌所在的区域
    maxweight, maxindex = 0, -1
    for i in range(len(block)):
        # b是从原图中取出来的一块
        b = afterimg[block[i][0][1]:block[i][0][3], block[i][0][0]:block[i][0][2]]
        # RGB 转 HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 蓝色车牌的范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩膜，限定lower和upper以消除不满足要求的区域
        mask = cv2.inRange(hsv, lower, upper)
        mean = cv2.mean(mask)
        if mean[0] > maxweight:
            maxweight = mean[0]
            maxindex = i
    return block[maxindex][0]

if __name__ == '__main__':
    for i in range(1, 4):
        img = cv2.imread('./pic/'+str(i) + '.jpg', cv2.IMREAD_COLOR)
        # 预处理图像
        openingimg = preprocessing(img)
        # 定位车牌的位置
        rect = locate_license(openingimg, img)
        # 将车牌所在的区域，使用矩形框起来
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 100, 200), 2)
        cv2.imshow('The license plate image', img)
        cv2.waitKey()