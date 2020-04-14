from constants import *
from modulating_functions import *
from demodulating_functions import *
from general_functions import *


def main(bits):
    final_sample_list = modulation(bits, mode="FM-nbit")
    path = "FM__BT{}_{}freqs_FS{}.wav".format(SAMPLES_PER_SYMBOL, NUM_OF_FREQS, FREQ_SHIFT)
    frame_rate, data = wavfile.read(path)
    data = data[:, 1]
    # data = data[:-30000]
    n_frames = len(final_sample_list)
    eps = find_thresh(data, n_frames, len(bits) / NUM_OF_FREQS)
    dat = cut_irellevant(data, thresh=eps, draw=True)

    thresh = syncThresh(dat, BITS)
    dat = dechyperMultyFreq(dat, draw=False, thresh=thresh)
    return dat


if __name__=="__main__":
    bits = BITS
    print(main(bits) - bits)



