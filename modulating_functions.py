from constants import *
from demodulating_functions import *


###------------------###
###  create gausian  ###
###------------------###

a = 1
b1 = 0
b2 = 1
c = 10 / SAMPLE_RATE


###------------------###

def get_gausian(basic_time=BASIC_TIME):
    samples_per_bit = SAMPLES_PER_SYMBOL

    arr1 = np.linspace(0, 0.5, samples_per_bit // 2)
    arr2 = np.linspace(0.5, 1, samples_per_bit // 2)

    gaus1 = lambda x: a * np.exp(-((x - b1) ** 2) / c)
    gaus2 = lambda x: a * np.exp(-((x - b2) ** 2) / c)

    res_arr1 = 1 - gaus1(arr1)
    res_arr2 = 1 - gaus2(arr2)
    arr = np.concatenate((arr1, arr2))
    res = np.concatenate((res_arr1, res_arr2))
    return res


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def createBitArray(bits, n):
    l = len(bits)
    bit_array = []
    res = get_gausian()
    bit_len = SAMPLES_PER_SYMBOL
    for k in range(l):
        bit_array.append(np.ones(bit_len) * bits[k] * res)

    final_bit_array = bit_array[0]
    for arr in range(1, len(bit_array)):
        final_bit_array = np.append(final_bit_array, bit_array[arr])
    return final_bit_array


def write_raw_samples(samples, len_bits, delta, theta):
    samples = SAMPLE_RATE / int(FREQ + delta)
    num_of_samples = int(SAMPLES_PER_SYMBOL * len_bits)
    samples_list = np.arange(num_of_samples)

    sample_list = MAX_VOLUME * np.cos(2 * PI * samples_list / samples + theta)

    return sample_list, num_of_samples


def write_notes(bits, theta=0, delta=0):
    len_bits = len(bits)
    sample_list, num_of_samples = write_raw_samples(samples, len_bits, delta, theta)

    bit_array = createBitArray(bits, num_of_samples)
    sample_list = sample_list * bit_array

    return sample_list.astype(int)


def write_sample_list(bits, d=0, t=0, n=1):
    final_sample_list = []

    ### create the one fitted ###
    one_array = write_notes(bits, delta=d * FREQ_SHIFT)

    ### create the zero fitted ###
    zeros_array = write_notes(1 - bits, theta=t * PI)
    final_array = one_array + zeros_array * n
    for samp in final_array:
        final_sample_list.append([samp, samp])

    return final_sample_list


def check_hilbert(m_t, x, t):
    ###
    plt.figure()
    plt.plot(t, m_t)  # plot modulating signal
    plt.title('Modulating signal')
    plt.xlabel('t')
    plt.ylabel('m(t)')
    # #Add AWGN noise to the transmitted signal
    nMean = 0  # noise mean
    nSigma = 0.1  # noise sigma
    n = np.random.normal(nMean, nSigma, len(t))
    r = x + n + np.sin(2 * PI * 400 * t)  # noisy received signal
    #     r = x

    r = butter_highpass_filter(r, FREQ, SAMPLE_RATE, order=8)

    # Demodulation of the noisy Phase Modulated signal
    z = hilbert(r)  # form the analytical signal from the received vector
    inst_phase = np.unwrap(np.angle(z))  # instaneous phase
    coef_ = np.polyfit(t, inst_phase, 1)
    offsetTerm = coef_[0] * t + coef_[1]
    demodulated = inst_phase - offsetTerm

    ###
    plt.figure()
    plt.plot(t, demodulated)  # demodulated signal
    plt.title('Demodulated signal')
    plt.xlabel('n')
    plt.ylabel('\hat{m(t)}')


### PM bit array ###
def createBitArrayPM(bits):
    """
    creates the array of bits to multiply the sin wave
    """
    l = len(bits)
    bit_array = []
    res = get_gausian(DURATION_PM)
    bit_len = int(DURATION_PM * SAMPLE_RATE)
    for k in range(l):
        bit_array.append(np.ones(bit_len) * bits[k])
    final_bit_array = bit_array[0]
    for arr in range(1, len(bit_array)):
        final_bit_array = np.append(final_bit_array, bit_array[arr])
    return final_bit_array


def PM_modulation(bits, check=False):
    t = np.arange(int(SAMPLE_RATE * DURATION_PM) * len(bits)) / SAMPLE_RATE  # time base

    info = (np.pi / 2) * createBitArrayPM(bits)
    print('info = ', len(info))
    print('t = ', len(t))

    # Phase Modulation
    m_t = info * np.cos(2 * PI * FM_PM * t + info)
    x = np.cos(2 * PI * FREQ * t + BETA + m_t)  # modulated signal
    if check:
        check_hilbert(m_t, x, t)
    final_sample_list = []

    x = x * MAX_VOLUME
    x = x.astype(int)
    for samp in x:
        final_sample_list.append([samp, samp])

    dat = redecyherPM(DecypherPM(x, thresh=1))

    err = np.array(dat) - bits
    n_of_err = np.count_nonzero(err == 1)

    print("number of errors: " + str(n_of_err))

    return final_sample_list

### ----------------- ###
### symbol modulation ###
### ----------------- ###


def multiSymbolModulaion(bits, freq_num = 2):

    array_list = []
    for i in range(freq_num):
        arr = bits[i: len(bits): freq_num]
        array_list.append(arr)
    arr_zeros = np.logical_not(np.logical_or.reduce((array_list)))
    arr_zeros = np.array([1 if i else 0 for i in arr_zeros])

    final_array = write_notes(arr_zeros, delta = freq_num * FREQ_SHIFT)
    for i in range(len(array_list)):
        final_array += write_notes(array_list[i], delta=i * FREQ_SHIFT)

    final_array = (final_array / freq_num).astype("int64")

    final_sample_list = []
    for samp in final_array:
        final_sample_list.append([samp, samp])

    return final_sample_list



def multiply(data, number=3):
    returned = []
    for i in data:
        for _ in range(number):
            returned.append(i)
    return np.asarray(returned)


###------------###
### MODULATION ###
###------------###

def modulation(to_transmit, mode=None):
    if mode == "FM":
        ### simple FREQUENCY_SHIFT modulation
        final_sample_list = write_sample_list(to_transmit, d=1)

    elif mode == "binary":
        ### simple BINARY_SHIFT modulation
        final_sample_list = write_sample_list(to_transmit, n=0)

    elif mode == "phase":
        ### simple PHASE_SHIFT modulation
        final_sample_list = write_sample_list(to_transmit, t=1)

    elif mode == "PM":
        ### PM with carrier wave
        final_sample_list = PM_modulation(to_transmit)

    elif mode == "FM-nbit":
        final_sample_list = multiSymbolModulaion(to_transmit, NUM_OF_FREQS)

    else:
        print("give me something")
        return None

    return final_sample_list


def fmSymbolModulation(bits):
    arr_10 = bits[0: len(bits): 2]
    arr_01 = bits[1: len(bits): 2]
    arr_00 = np.logical_not(np.logical_or(arr_01, arr_10))
    arr_00 = np.asarray([1 if i else 0 for i in arr_00])

    m_arr_10 = write_notes(arr_10 , delta=symbol_dict["10"])
    m_arr_01 = write_notes(arr_01 , delta=symbol_dict["01"])
    m_arr_00 = write_notes(arr_00, delta=symbol_dict["00"])

    final_array = (m_arr_00 + m_arr_01 + m_arr_10)/2

    # freqs = np.fft.fftfreq(final_array.shape[0], d=1 / SAMPLE_RATE)
    # plt.plot(freqs, np.abs(np.fft.fft(final_array)))
    # plt.show()

    final_sample_list = []
    for samp in final_array:
        final_sample_list.append([samp, samp])

    return final_sample_list


