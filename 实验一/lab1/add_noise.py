##表示待处理的图像矩阵和信噪比
##信噪比是1表示没有噪声，是0表示全是噪声
import numpy as np
import random
def add_salt_noise(img, prop):
    np.random.seed(0)
    ans = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_digit = np.random.rand() ##生辰一个0-1区间内的随机数
            if random_digit > prop:
                random_digit_salt = np.random.rand()
                if random_digit_salt > 0.5:
                    ans[i][j] = 255
                elif random_digit_salt < 0.5:
                    ans[i][j] = 0
    return ans

def add_gauss_noise(img, mu, sigma):
    gauss_noise = np.random.normal(mu, sigma ** 0.5, img.shape)
    ans = img + gauss_noise
    ans = np.uint8(ans)
    ans[ans > 255] = 255
    ans[ans < 0] = 0
    return ans