import cv2
from plot_hist import *
from histogram_equalization import *
from histogram_specification import *

pic = cv2.imread('./pic/1.jpg', cv2.IMREAD_GRAYSCALE)  # 从文件中读取图片，按照灰度图形式读取

plot_hist(pic, './pic/histogram_original.png')  # 绘制原图像的直方图

histogram_equalization_pic = histogram_equalization(pic)  # 进行直方图均衡化处理，得到直方图均衡化处理之后的图像
cv2.imwrite('./pic/pic_histogram_equalization.png', histogram_equalization_pic)

plot_hist(histogram_equalization_pic, './pic/histogram_equalization.png')  # 绘制经过直方图均衡化处理之后的图片的直方图

# 展示原图片和进行直方图处理之后的图片，展示直方图处理的效果
pics = np.hstack([pic, histogram_equalization_pic])
cv2.imshow('original pic vs. histogram equalization pic', pics)
cv2.waitKey()

# 进行直方图规定化
pic_Lenna = cv2.imread('./pic/Lenna.png', cv2.IMREAD_GRAYSCALE)  # 读取Lenna图像
plot_hist(pic_Lenna, './pic/histogram_Lenna.png')
pic_specification = histogram_specification(pic, pic_Lenna)
cv2.imwrite('./pic/pic_histogram_specification.png', pic_specification)  # 保存直方图规定化之后的图像
plot_hist(pic_specification, './pic/histogram_specification.png')

# 展示原图片和进行直方图规定化之后的图片，展示直方图规定化处理的效果
pics = np.hstack([pic, pic_specification])
cv2.imshow('original pic vs. histogram specification pic', pics)
cv2.waitKey()

# 下面检测同态滤波函数
from homomorphic_filtering import homomorphic_filtering

pic_homo = homomorphic_filtering(pic, 2, 0.5, 0.3, 85)

cv2.imshow('homo', pic_homo)
cv2.imwrite('./pic/pic_homo.png', pic_homo)
cv2.waitKey()

# 下面检测双边滤波算法
# 像素标准差是30，位置标准差是10
# 滤波模板大小可以取30,30
from bi_filtering import *

pic = cv2.imread('./pic/shaungbian.jpg', cv2.IMREAD_GRAYSCALE)
after_bi = bilateral_filter(pic, 15, 30, 30)
pics = np.hstack([pic, after_bi])
cv2.imshow('original pic vs. histogram specification pic', pics)
cv2.imwrite('./pic/after_shuangbianlvbo.png', after_bi)
cv2.waitKey()

