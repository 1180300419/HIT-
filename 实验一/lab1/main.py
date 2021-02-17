''''''
'''
1、首先加载图片
2、为图像添加高斯噪声和椒盐噪声并将进行图像展示和保存
3、对添加噪声之后的图片分别进行中值滤波和均值滤波处理，对处理后的图片进行展示并进行保存
4、使用sobel算子对添加了高斯噪声并且进行均值滤波的图片进行锐化处理，展示锐化的图像，并保存图片
5、使用candy算子对添加了椒盐噪声，并且进行了均值滤波处理的图片进行锐化处理，展示锐化的图片，并保存图片
'''


import cv2
from load_pic import *
from smooth_pic import *

img = loadImage('./pic/Lenna.png')

img_salt_noise = add_salt_noise(img, 0.95)
cv2.imwrite('./pic/salt_Lenna.png', img_salt_noise)
##使用该函数将矩阵写成png文件
cv2.imshow('salt noise, prop=0.95', img_salt_noise)
##展示图像，第一个参数是窗口名字，第二个是要展示的矩阵
cv2.waitKey()
##等待图片关闭
img_gauss_noise = add_gauss_noise(img, 0, 400)
cv2.imshow('gauss noise', img_gauss_noise)
cv2.imwrite('./pic/gauss_noise_Lenna.png', img_gauss_noise)
##展示图像，第一个参数是窗口名字，第二个是要展示的矩阵
cv2.waitKey()

##对椒盐噪声的图片进行两次均值滤波处理
template = np.ones((5, 5)) * 1 / 25
template[2, 2] = 0
salt_pic_average_filter = average_filtering(img_salt_noise, template)

##对添加椒盐噪声的图片进行中值滤波处理
salt_pic_median_filter = median_filter(img_salt_noise, radius=1)
##使用cv2在一个窗口中展示多个图片
imgs = np.hstack([salt_pic_average_filter, salt_pic_median_filter])
cv2.imshow("salt noise: average filter vs. median filter", imgs)
cv2.waitKey()

##对添加了高斯噪声的图片进行均值滤波处理
gassian_pic_average_filter = average_filtering(img_gauss_noise, template)
gassian_pic_average_filter = average_filtering(gassian_pic_average_filter, template)
gassian_pic_median_filter = median_filter(img_gauss_noise, radius=3)
imgs = np.hstack([gassian_pic_average_filter, gassian_pic_median_filter])
cv2.imshow("gassian noise: average filter vs. median filter", imgs)
cv2.waitKey()

##将上面滤波得到的处理后的图像保存
cv2.imwrite('./pic/salt_pic_average_filter.png', salt_pic_average_filter)
cv2.imwrite('./pic/salt_pic_median_filter.png', salt_pic_median_filter)

cv2.imwrite('./pic/gassian_pic_average_filter.png', gassian_pic_average_filter)
cv2.imwrite('./pic/gassian_pic_median_filter.png', gassian_pic_median_filter)


from sharpen_pic import *
##下面进行滤波处理
border_pic, Gx, Gy = sobel(gassian_pic_median_filter, 60)
cv2.imshow('sobel filter', border_pic)
cv2.imwrite('./pic/sobel_filter.png', border_pic)
cv2.waitKey()

sharpen_pic = sharpen_pic(salt_pic_median_filter, border_pic, 0.2)
imgs = np.hstack([salt_pic_median_filter, sharpen_pic])
cv2.imwrite('./pic/sobel_sharpen.png', sharpen_pic)
cv2.imshow('salt_pic_median_filter pic vs. sharpen_pic', imgs)
cv2.waitKey()

ans = canny(img)
cv2.imshow('canny', ans)
cv2.waitKey()

from sharpen_pic import *
canny_official(img)
cv2.waitKey()

##学习新算法

##图像锐化算法
roberts_pic = roberts(img)
cv2.imshow('roberts', roberts_pic)
cv2.imwrite('./pic/roberts.png', roberts_pic)
cv2.waitKey()

##Kirsch算法

krischs_pic = Krisch(img, 130)
cv2.imshow('krischs', krischs_pic)
cv2.imwrite('./pic/krischs.png', krischs_pic)
cv2.waitKey()



