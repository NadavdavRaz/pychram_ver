from constants import *
from modulating_functions import *
from general_functions import *


if __name__ == "__main__":
    bits = BITS
    final_sample_list = modulation(bits, mode="FM-nbit")
    path = "samples/FM__BT{}_{}freqs_FS{}.wav".format(SAMPLES_PER_SYMBOL, NUM_OF_FREQS, FREQ_SHIFT)
    save_wave(SAMPLE_RATE, final_sample_list, path)