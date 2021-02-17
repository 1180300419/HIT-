import wave
import numpy as np
import struct

class DPCM_4_Q:
    ''''''
    '''
    该函数读取filename表示的wave文件，并且返回所有采样点组成的数组
    '''
    def read_data(self, filename):
        f = wave.open(filename, "rb")
        params = f.getparams()  # 获得文件的格式信息
        nchannels, sampwidth, framerate, nframes = params[:4]  # 依次获得各参数
        # 输出一些辅助参数
        print('该文件的声道数是: ', nchannels)
        print('该文件中样本点占用的字节数是: ', sampwidth)
        print('该文件的采样频率是: ', framerate)
        print('该文件的采样个数是: ', nframes)
        # 获得采样点的数据
        str_data = f.readframes(nframes)
        f.close()
        # 将波形数据转换成数组
        # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
        wave_data = np.frombuffer(str_data, dtype='i2')
        return wave_data
    '''
    wave_data:表示要进行编码的波形数据
    bits_digit:表示压缩编码之后，用多少比特
    a:量化参数取值
    该函数范围编码后的数据和dn
    '''
    def encoder(self, wave_data, bits_digit, a):
        ans_data = []  # 用来存储编码后的数据
        dnn = []  # 辅助我们进行a的选取
        flag = False  # 用来判断是不是第一个待编码数据
        pre_data = 0  # 译码器需要用到的数据
        for data in wave_data:
            if not flag:
                ans_data.append(data)
                pre_data = data
                dnn.append(data)
                flag = True
            else:
                dn = data - pre_data
                dnn.append(dn)
                cn = self.quantitative(dn, bits_digit, a)
                pre_data = pre_data + cn * a - a // 2  # 更新该值
                ans_data.append(cn)  # 存储的是量化之后的数据
        return ans_data, dnn
    '''
    将encoded_data表示的数据写入filename表示的文件中
    '''
    def write_encoded_data(self, filename, encoded_data):
        f = open(filename, 'wb')
        f.write(struct.pack('h', encoded_data[0]))  # 转换为字节流
        for i in range(1, len(encoded_data) // 2 + 1):
            f.write(struct.pack('B', ((encoded_data[2 * i - 1] + 8) << 4) | (encoded_data[2 * i] + 8)))
        f.close()
    '''
    从filename表示的文件中读取编码之后的数据，返回差分数组
    '''
    def read_encoded_data(self, filename):
        ans_data = []
        with open(filename, 'rb') as f:
            ans, = struct.unpack('h', f.read(2))
            ans_data.append(ans)
            while True:
                tmp_data = f.read(1)
                if not tmp_data: break
                ans, = struct.unpack('B', tmp_data)
                ans_data.append((ans // 16) - 8)
                ans_data.append((ans % 16) - 8)
        return ans_data
    '''
    量化函数
    i：表示要量化的数字
    bits_digit:表示量化之后的位数
    a:表示量化因子
    '''
    def quantitative(self, i, bits_digit, a):
        high_level = np.power(2, bits_digit - 1) - 1
        low_level = -high_level - 1
        if i >= high_level * a:
            return high_level
        elif i <= low_level * a:
            return low_level
        floor = np.int(np.floor(np.abs(i) / a))
        ceil = np.int(np.ceil(np.abs(i) / a))
        if i >= 0: return ceil
        else: return -floor
    '''
    encoded_data：表示待解码数据组成的数组,
    返回解码之后的数据
    '''
    def decoder(self, bits_digit, encoded_data, a):
        length = len(encoded_data)
        ans_data = np.zeros(length)
        pre_data = 0
        for i in range(length):
            ans_data[i] = pre_data + encoded_data[i] * a - a // 2
            pre_data = ans_data[i]
        ans_data = ans_data.astype(np.int16)
        return ans_data
    '''
    将解码后的数据decoded_data写进filename表示的文件中
    '''
    def write_decoded_data(self, filename, decoded_data):
        f = open(filename, 'wb')
        for i in range(len(decoded_data)):
            f.write(decoded_data[i])
        f.close()

my_dpcm = DPCM_4_Q()

# 编码步骤
a = 370
wave_data = my_dpcm.read_data('../data/1.wav')
encoded_data, dnn = my_dpcm.encoder(wave_data, 4, a)
import matplotlib.pyplot as plt
ans = []
# 绘制dnn的密度分布函数，如果不限制其范围是[-2000,2000]则绘制出的图像比较分散，不易观察
# 2000是通过不断尝试选取的一个比较合适的范围
for i in range(len(dnn)):
    if np.abs(dnn[i]) <= 2000:
        ans.append(dnn[i])
plt.hist(ans, bins=50, color='steelblue')
plt.show()
my_dpcm.write_encoded_data('../results/1_4bit_q.dpc', encoded_data)
# 解码步骤
encoded_data = my_dpcm.read_encoded_data('../results/1_4bit_q.dpc')
decoded_data = my_dpcm.decoder(4, encoded_data, a)
my_dpcm.write_decoded_data('../results/1_4bit_q.pcm', decoded_data)
from SNR import *
snr = SNR(wave_data, decoded_data)
print('4比特量化因子法得到的信噪比是: ', snr, '分贝')