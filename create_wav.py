from constants import *
from modulating_functions import *
from general_functions import *


if __name__ == "__main__":
    bits = np.array([1,0,1,0,1,0,1,1,0,0,0,1]*5)
    to_send = multiply(bits, 3)
    final_sample_list = modulation(to_send, mode="FM")
    save_wave(SAMPLE_RATE, final_sample_list, 'samples/FM_BT_500.wav')