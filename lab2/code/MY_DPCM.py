import wave
import numpy as np
import struct

class MY_DPCM:
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
                cn = self.quantitative(dn, bits_digit)
                cnn = self.inverse_quantitative(cn)
                #print(dn, cn, cnn)
                pre_data = pre_data + cnn  # 更新该值
                ans_data.append(cn)  # 存储的是量化之后的数据
        return ans_data
    '''
    将encoded_data表示的数据写入filename表示的文件中
    '''
    def write_encoded_data(self, filename, encoded_data):
        f = open(filename, 'wb')
        f.write(struct.pack('h', encoded_data[0]))  # 转换为字节流
        for i in range(1, len(encoded_data)):
            f.write(struct.pack('b', encoded_data[i]))
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
                ans, = struct.unpack('b', tmp_data)
                ans_data.append(ans)
        return ans_data
    '''
    直接量化函数
    i：表示要量化的数字
    bits_digit:表示量化之后的位数
    '''
    def quantitative(self, i, bits_digit):
        # 判断待量化的数值是正数还是负数
        if i < 0: neg = True
        else: neg = False
        # 判断量化的时候是否进行精确表示
        if -32 <= i <= 31: precise = True
        else: precise = False

        if precise == True:
            # 对正数进行精确存储
            if not neg: return i
            else: return np.abs(i) | 64
            # 对负数进行精确存储
        elif np.abs(i) <= 3969:
            # 对正数进行近似存储
            if not neg: return (np.int8(np.round(np.sqrt(i))) | -128)
            # 对负数进行近似存储
            else: return (np.int8(np.round(np.sqrt(np.abs(i)))) | -64)
        elif i > 3969: return -65
        else: return -1

    def inverse_quantitative(self, encoded_data):
        # 判断是否进行精确表示
        if encoded_data >> 7 != 0: precise = False
        else: precise = True
        neg = encoded_data & 64
        encoded_data = encoded_data & 63
        if not precise:
            if not neg: return encoded_data * encoded_data
            else: return -encoded_data * encoded_data
        else:
            if not neg: return encoded_data
            else: return -encoded_data

    '''
    encoded_data：表示待解码数据组成的数组,
    返回解码之后的数据
    '''
    def decoder(self, encoded_data):
        length = len(encoded_data)
        ans_data = np.zeros(length)
        pre_data = encoded_data[0]
        ans_data[0] = encoded_data[0]
        for i in range(1, length):
            cnn = self.inverse_quantitative(encoded_data[i])
            ans_data[i] = pre_data + cnn
            pre_data = ans_data[i]

        ans_data = ans_data.astype(np.int16)
        return ans_data

    def write_decoded_data(self, filename, decoded_data):
        f = open(filename, 'wb')
        for i in range(len(decoded_data)):
            f.write(decoded_data[i])
        f.close()

my_dpcm = MY_DPCM()

# 编码步骤
wave_data = my_dpcm.read_data('../data/1.wav')

encoded_data = my_dpcm.encoder(wave_data, 8)
my_dpcm.write_encoded_data('../results/1_8bit_mq.dpc', encoded_data)
# 解码步骤
encoded_data = my_dpcm.read_encoded_data('../results/1_8bit_mq.dpc', 8)

decoded_data = my_dpcm.decoder(encoded_data)
my_dpcm.write_decoded_data('../results/1_8bit_mq.pcm', decoded_data)

from SNR import *
snr = SNR(wave_data, decoded_data)
print('8比特量化因子法得到的信噪比是: ', snr, '分贝')