import wave
import numpy as np

# 计算传递进来的参数data的全部能量
def compute_powers(data):
    sum = 0
    for i in range(len(data)):
        sum += int(data[i]) * int(data[i])
    return sum

# 计算传递进来的参数data的过零率
def compute_zeros_rate(data):
    length = len(data)  # 获取字符串的长度
    signs = np.sign(data)  # 获取data的符号
    diff_value = np.abs(signs[1:] - signs[:length - 1])
    return np.sum(diff_value) / (2 * length)

# 传递进的参数是采样数据，每一帧的能量数据 和 每一帧的过零率数据
def double_threshold_method(data, powers, zeros, frame_len):

    MH = np.average(powers) / 4  # 能量的高阈值
    ML = (np.average(powers[:4])+ MH) / 5  # 能量的低阈值
    ZS = np.average(zeros[:4]) * 3  # 过零率的阈值
    # 下面利用上面这三个、两类阈值 确定哪些是语音部分，哪些是噪声部分
    nframes = len(data) // frame_len + 1  # 因为要为每一帧划定标签，因此计算帧数为下面的处理做准备

    # 下面两个列表用来存储浊音部分和清音部分
    voiced_sound = np.zeros(nframes)
    unvoiced_sound = np.zeros(nframes)

    # 首先利用能量的高阈值确定浊音部分，同时将voiced_sound和unvoiced_sound相应的帧数标为1
    for i in range(nframes):
        if powers[i] > MH:
            voiced_sound[i] = 1
            unvoiced_sound[i] = 1
    # 利用能量的低阈值继续延伸语音段
    for i in range(len(voiced_sound)):
        # 因为最后一帧不是完整的，因此要判断是不是最后一帧
        if voiced_sound[i] == 1 and i > 0 and voiced_sound[i - 1] == 0:
            for j in range(i, 0, -1):
                if powers[j] > ML:
                    unvoiced_sound[j] = 1
                else:  # 一旦出现一个低于低阈值的段，则停止标记
                    break
        if voiced_sound[i] == 1 and i + 1 < nframes and voiced_sound[i + 1] == 0:
            for j in range(i + 1, nframes):
                if powers[j] > ML:
                    unvoiced_sound[j] = 1
                else:
                    break
    # 利用过零率值判断清音部分
    for i in range(len(unvoiced_sound)):
        # 因为最后一帧可能不是完整的，因此需要进行判断是不是最后一帧
        if unvoiced_sound[i] == 1 and i > 0 and unvoiced_sound[i - 1] == 0:
            for j in range(i - 1, 0, -1):
                if zeros[j] > ZS:
                    unvoiced_sound[j] = 1
                else:
                    break
        if unvoiced_sound[i] == 1 and i + 1 < nframes and unvoiced_sound[i + 1] == 0:
            for j in range(i, nframes):
                if zeros[j] > ZS:
                    unvoiced_sound[j] = 1
                else:
                    break
    return unvoiced_sound

for file_n in range(1, 11):
    # 打开wav文件，打开方式是读取二进制文件
    f = wave.open(r"./data/" + str(file_n) + ".wav", "rb")
    params = f.getparams()  # 获得文件的格式信息
    nchannels, sampwidth, framerate, nframes = params[:4]  # 依次获得各参数
    # 输出一些辅助参数
    print('该文件的声道数是: ', nchannels)
    print('该文件中样本点占用的字节数是: ', sampwidth)
    print('该文件的采样频率是: ', framerate)
    # 获得采样点的数据
    str_data  = f.readframes(nframes)
    f.close()
    #将波形数据转换成数组
    #需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_data = np.frombuffer(str_data,dtype = 'i2')
    frame_len = 256  # 设置帧长是256个采样点
    nframes = nframes // frame_len + 1  # 计算帧数
    # 用来存储各帧的能量值和过零率值
    powers = []
    zeros = []
    # 打开相应的文件，用来存储每帧的能量和过零率
    pow_file = open('./powers/'+ str(file_n) + '_en.txt', mode='w')
    zeros_file = open('./zeros/' + str(file_n) + '_zero.txt', mode='w')
    # 获取每一帧的数据，利用上面两个函数计算相应的值
    for i in range(nframes):
        if (i + 1) * frame_len < len(wave_data):
            frame_data = wave_data[i * frame_len : (i + 1) * frame_len]
        else:
            frame_data = wave_data[i * frame_len:]
        powers.append(compute_powers(frame_data))
        zeros.append(compute_zeros_rate(frame_data))
        pow_file.write(str(powers[i]) + '\n')
        zeros_file.write(str(zeros[i]) + '\n')
    pow_file.close()
    zeros_file.close()

    data_new = double_threshold_method(wave_data, powers, zeros, frame_len = 256)

    file_pcm = open('./pcm/' + str(file_n) + '.pcm', 'wb')
    for i in range(len(data_new)):
        if data_new[i] == 1:
            if (i + 1) * frame_len < len(wave_data):
                file_pcm.write(wave_data[i * frame_len:(i + 1) * frame_len])
            else:
                file_pcm.write(wave_data[i * frame_len:])
    file_pcm.close()