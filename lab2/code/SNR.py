import numpy as np

def SNR(input, output):
    input = np.int64(input)
    input_power = np.sum(input * input)
    noise = np.int64(input - output)
    noise_power = np.sum(noise * noise)
    return 10 * np.log10(input_power / noise_power)

