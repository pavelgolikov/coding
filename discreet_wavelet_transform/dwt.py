import numpy as np
import pywt

def is_power_of_two(number : int) -> bool:
    while number != 1:
        if number % 2:
            return False
        number /= 2
    return True

def scaled_haar_wavelet(j, approx):
    """
    Return list representing the mother wavelet stretched by a factor of j.
    """
    if not approx:
        wave = []
        for l in range(1, (2 ** j) + 1):
            arg = (l - 1) / (2 ** j)
            value = 1 if arg < 1/2 and arg >= 0 else -1 if arg >= 1/2 and arg < 1 else 0
            wave.append((2 ** ((-1 * j) / 2 )) * value)
        return wave
    else:
        wave = []
        for l in range(1, (2 ** j) + 1):
            arg = (l - 1) / (2 ** j)
            value = 1 if arg < 1 and arg >= 0 else 0
            wave.append((2 ** ((-1 * j) / 2 )) * value)
        return wave

def compute_dwt(data, level):
    # create the scaled haar wavelets
    wave = []
    # create scaled haar wavelets for filtering
    for i in range(1, level + 1):
        wave.append(scaled_haar_wavelet(i, False))
        
    # scaling function for the last filter
    wave.append(scaled_haar_wavelet(3, True))
    # print(wave)
    
    # now we convolve the haar wavelets across the data
    res = []
    for i in range(0, level + 1):
        # step size is equal to wavelet size
        wavelet_size = len(wave[i])
        # wavelet size and size of data will determine the step and how many times we convolve
        num_conv = len(data) // wavelet_size
        conv_res = []
        for k in range(num_conv):
            conv_start = k * wavelet_size
            conv_sum = 0
            for j in range(wavelet_size):
                # print(j, conv_start + j)
                conv_sum += wave[i][j] * data[conv_start + j]
            conv_res.append(conv_sum)
        res.append(np.array(conv_res))
    res.reverse()
    return res

if __name__ == "__main__":
    # data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 0.0]
    data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 0.0, 8.0, 1.0, 0.0, 3.0, 4.0, 5.0, 2.0, 0.0]
    assert is_power_of_two(len(data)) == True, "length of data array should be a power of 2"
    print(compute_dwt(data, 3))
    coeffs = pywt.wavedec(data, 'haar', level=3)
    print(coeffs)
