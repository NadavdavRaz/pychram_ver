from general_functions import *

if __name__ == "__main__":
    path = "FM__BT{}_{}freqs_FS{}.wav".format(SAMPLES_PER_SYMBOL, NUM_OF_FREQS, FREQ_SHIFT)
    record(path, RECORD_TIME)
