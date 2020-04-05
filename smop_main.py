from constants import *
from modulating_functions import *
from demodulating_functions import *
from general_functions import *


if __name__=="__main__":
    bits = np.array([1,0,1,0,1,0,1,1,0,0,0,1]*5)
    to_send = multiply(bits, 3)
    final_sample_list = modulation(to_send, mode="FM")
    frame_rate, data = wavfile.read("new_FM_test.wav")
    data = data[:,1]
    # data = data[40000:]
    n_frames = len(final_sample_list)
    eps = find_thresh(data, n_frames, len(to_send))
    print(eps)
    dat = cut_irellevant(data, thresh=eps, draw=False)
    print(len(final_sample_list))
    print(len(dat))
    dat = DecypherFreqShift(dat)
    print(dat - bits)
    # print(bits)




