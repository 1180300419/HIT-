import wave
import numpy as np
import struct

class DPCM_8_DQ:
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
    '''
    def encoder(self, wave_data, bits_digit):
        ans_data = []
        flag = False
        # 用来判断是不是第一个待编码数据
        pre_data = 0
        for data in wave_data:
            if not flag:
                ans_data.append(data)
                pre_data = data
                flag = True
            else:
                dn = data - pre_data
                cn = self.direct_quantitative(dn, bits_digit)
                pre_data = pre_data + cn  # 更新该值
                ans_data.append(cn)  # 存储的是量化之后的数据
        return ans_data
    '''
    将encoded_data表示的数据写入filename表示的文件中
    '''
    def write_encoded_data(self, filename, encoded_data):
        f = open(filename, 'wb')
        f.write(struct.pack('h', encoded_data[0]))  # 转换为字节流
        for i in range(1, len(encoded_data)):
            f.write(struct.pack('B', encoded_data[i]+ 128))
        f.close()
    '''
    从filename表示的文件中读取编码之后的数据，并按照bits_digit表示的位数对其解码
    返回解码之后的差分数组
    '''
    def read_encoded_data(self, filename, bits_digit):
        ans_data = []
        with open(filename, 'rb') as f:
            ans, = struct.unpack('h', f.read(2))
            ans_data.append(ans)
            while True:
                tmp_data = f.read(1)
                if not tmp_data: break
                ans, = struct.unpack('B', tmp_data)
                ans_data.append(ans - 128)
        return ans_data
    '''
    直接量化函数
    i：表示要量化的数字
    bits_digit:表示量化之后的位数
    '''
    def direct_quantitative(self, i, bits_digit):
        high_level = np.power(2, bits_digit - 1) - 1
        low_level = -high_level - 1
        if i >= high_level:
            return high_level
        elif i <= low_level:
            return low_level
        return i
    '''
    encoded_data：表示待解码数据组成的数组,
    返回解码之后的数据
    '''
    def decoder(self, encoded_data):
        length = len(encoded_data)
        ans_data = np.zeros(length)
        pre_data = 0
        for i in range(length):
            ans_data[i] = pre_data + encoded_data[i]
            pre_data = ans_data[i]
        ans_data = ans_data.astype(np.int16)
        return ans_data

    def write_decoded_data(self, filename, decoded_data):
        f = open(filename, 'wb')
        for i in range(len(decoded_data)):
            f.write(decoded_data[i])
        f.close()
my_dpcm = DPCM_8_DQ()

# 编码步骤
wave_data = my_dpcm.read_data('../data/1.wav')
encoded_data = my_dpcm.encoder(wave_data, 8)
my_dpcm.write_encoded_data('../results/1_8bit_dq.dpc', encoded_data)
# 解码步骤
encoded_data = my_dpcm.read_encoded_data('../results/1_8bit_dq.dpc', 8)
decoded_data = my_dpcm.decoder(encoded_data)
my_dpcm.write_decoded_data('../results/1_8bit_dq.pcm', decoded_data)
from SNR import *
snr = SNR(wave_data, decoded_data)
print('8比特直接量化得到的信噪比是: ', snr, '分贝')

