from constants import *


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


### decypher PM signals ###


def DecypherPM(r, thresh):
    chunk_size = int((DURATION_PM * SAMPLE_RATE) / 4)
    t = np.arange(len(r))
    r = butter_highpass_filter(r, FREQ, SAMPLE_RATE, order=8)
    z = hilbert(r)  # form the analytical signal from the received vector
    inst_phase = np.unwrap(np.angle(z))  # instaneous phase
    chunk_inst_phase = [inst_phase[i * chunk_size: (i + 1) * chunk_size] for i in range(len(inst_phase) // chunk_size)]
    t_chunk = [t[i * chunk_size:(i + 1) * chunk_size] for i in range(len(t) // chunk_size)]

    for i in range(len(chunk_inst_phase)):
        coef_ = np.polyfit(t_chunk[i], chunk_inst_phase[i], 1)
        offset = coef_[0] * t_chunk[i] + coef_[1]
        dem = np.abs(chunk_inst_phase[i] - offset)

        if i == 0:
            dem_chunk = dem
        else:
            dem_chunk = np.append(dem_chunk, [dem])

    dem_chunk[dem_chunk > thresh] = thresh
    print(np.max(dem_chunk))

    t = np.arange(len(dem_chunk))
    plt.figure()
    plt.plot(t, np.abs(dem_chunk))

    plt.xlim(3400, 3800)
    #     plt.ylim(5, 20)
    return dem_chunk


def deal_queue(item, max_size, queue):
    queue.insert(0, item)
    if len(queue) > max_size:
        queue.pop()
    return queue


def redecyherPM(data, ChunkSize=(int(DURATION_PM * SAMPLE_RATE)), min_thresh=0.01, queue_max_size=10, sup=1):
    data = abs(data)
    thresh = min_thresh
    queue = []
    NumOfChunks = int(len(data) / ChunkSize)
    ChunkList = np.array_split(data, NumOfChunks)
    data_num_of_times = []
    for chunk in ChunkList:
        if np.sum(chunk) / ChunkSize > thresh:
            data_num_of_times.append(1)
            if np.sum(chunk) / ChunkSize < sup:
                queue = deal_queue(np.sum(chunk) / ChunkSize, queue_max_size, queue)
        else:
            data_num_of_times.append(0)
        if len(queue) > 5:
            thresh = 6 / 10 * np.sum(queue) / len(queue)
    #     plt.figure()
    #     plt.plot(data_num_of_times)
    return np.asarray(data_num_of_times)


###---------------------###
### cut irelevant parts ###
###---------------------###

def cut_irellevant(data, chunk_size=50, thresh=0.2, draw=True):
    l = len(data) // chunk_size
    data = butter_highpass_filter(data, FREQ, SAMPLE_RATE, order=8)

    ### Do FFT by chunks
    chunk_array = [np.array(data[chunk_size * i:(i + 1) * chunk_size]) for i in range(l)]
    freq = np.fft.fftfreq(chunk_size, d=1 / SAMPLE_RATE)
    fft_array = [np.abs(np.fft.fft(arr)) for arr in chunk_array]

    ### Integral over relevant freqs
    cut_index = int(3 * chunk_size / 8)
    integral_array = [np.trapz(arr[cut_index:chunk_size // 2], freq[cut_index:chunk_size // 2]) for arr in fft_array]
    integral_array = np.asarray(integral_array) / (-freq[chunk_size // 2] + freq[cut_index])

    if draw:
        plt.figure()
        plt.plot(integral_array)
        plt.show()
    ### Filter by thresh
    good_chunks = [i for i in range(len(integral_array)) if integral_array[i] > thresh]

    if good_chunks:
        begin = good_chunks[0] * chunk_size
        end = (good_chunks[-1] + 1) * chunk_size
    else:
        return None
    ### Cut and return data
    return data[begin:end]


def find_thresh(data, n_frames, n_bits):
    eps = 1
    i = 0

    while True:
        i += 1
        cur_n = cut_irellevant(data, thresh=eps, draw=False)
        if cur_n is None:
            cur_n = []

        cur_n = len(cur_n)
        dif = np.abs(cur_n - n_frames)
        if (dif < CHUNK_SIZE/2 or i > 30) and cur_n != 0:
            return eps

        if cur_n < n_frames:
            eps = eps - eps / 2
        else:
            eps = eps + eps / 2


def decide(data, number=3):
    final = []
    data = np.array_split(data, len(data) // number)
    for dat in data:
        if np.sum(dat) >= 2:
            final.append(1)
        else:
            final.append(0)
    return np.array(final)


###-----------###
### DECODE FM ###
###-----------###

def DecypherFreqShift(data, ChunkSize=CHUNK_SIZE, draw = False):
    data = butter_highpass_filter(data, FREQ, SAMPLE_RATE, order=8)
    missing = len(data)%CHUNK_SIZE
    missing = CHUNK_SIZE - missing
    data = np.concatenate((data, np.zeros(missing)))

    NumOfChunks = len(data) / ChunkSize
    ChunkList = np.array_split(data, NumOfChunks)
    ForTrans = [np.abs(np.fft.fft(dat)) for dat in ChunkList]

    # ForTrans = [[0] * (ChunkSize // 2 - 1) +
    #             list(ForTrans[index][ChunkSize // 2:3 * ChunkSize // 4]) +
    #             [0] * (ChunkSize // 4 + 1) for index in indexes]

    freqs = np.fft.fftfreq(ChunkSize, d=1 / SAMPLE_RATE)
    freqs_found = np.array([freqs[np.argmax(ForTrans[index])] for index in range(len(ForTrans))])

    found = [1 if np.abs(FREQ + FREQ_SHIFT - np.abs(frequ))
                  < np.abs(FREQ - np.abs(frequ)) else 0 for frequ in freqs_found]
    dat = decide(found, number=1 * RATIO)
    if draw:
        # plt.scatter(found[0])
        plt.plot(freqs, ForTrans[0])
        plt.show()
    return dat


def RestoreChunks(DecyphData):
    Ratio = RATIO
    ChunkData = []
    Chunk = 0
    first = DecyphData[0]
    for i in range(len(DecyphData)):
        if DecyphData[i] == first:
            Chunk += 1
        else:
            first = DecyphData[i]
            ChunkData.append(Chunk)
            Chunk = 1
    ChunkData.append(Chunk)
    Sizes = [chunk // Ratio if chunk % Ratio < Ratio / 2
             else chunk // Ratio + 1 for chunk in ChunkData]
    data = []
    state = DecyphData[0]
    for i in Sizes:
        data = data + [state] * i
        state = 1 - state
    return data

### demodulate multy freq ###


def majorty_decide(data_num_of_times):
    num_of_freqs = NUM_OF_FREQS
    ratio = RATIO
    final_data = []
    for pack_number in range(len(data_num_of_times)//ratio):
        for bit_in_pack in range(num_of_freqs):
            curr = 0
            for pack_checked in range(pack_number*ratio, (pack_number+1)*ratio):
                curr += data_num_of_times[pack_checked][bit_in_pack]
            curr = curr/ratio
            if curr > 1/2:
                curr = 1
            else:
                curr = 0
            final_data.append(curr)
    return final_data


def decide_multiple(lst_of_found, thresh = FREQ_SHIFT/2):
    num_of_freqs = NUM_OF_FREQS
    found_in_binary = []
    for found in lst_of_found:
        was_found = []
        for i in range(num_of_freqs + 1):
            flag = False
            for freq in found:
                if np.abs(freq - (FREQ + i*FREQ_SHIFT)) < thresh:
                    was_found = was_found + [1]
                    flag = True
            if not flag:
                was_found += [0]
        found_in_binary.append(was_found)
    return [found[:-1] for found in found_in_binary]


def dechyperMultyFreq(data, draw = False, thresh = PEAK_THRESH, to_thresh=False):
    missing = len(data) % CHUNK_SIZE
    missing = CHUNK_SIZE - missing
    ### check for defult size
    data = np.concatenate((data, np.zeros(missing)))

    # # final_sample_list = fmSymbolModulation(bits)
    # data = np.array(final_sample_list)[:, 1]
    NumOfChunks = len(data) / CHUNK_SIZE
    ChunkList = np.array_split(data, NumOfChunks)
    final_ChunkList = []
    for i in range(len(ChunkList)):
        # if i % RATIO == 0 or i % RATIO == RATIO -1:
        #     continue
        final_ChunkList.append(ChunkList[i])

    ForTrans = [np.abs(np.fft.fft(dat)) for dat in final_ChunkList]
    freqs = np.fft.fftfreq(CHUNK_SIZE, d=1 / SAMPLE_RATE)

    ### make function that finds thresh
    peaks = [signal.find_peaks(fort, distance=4, prominence=thresh) for fort in ForTrans]
    peaks = [peak[0] for peak in peaks]
    found = [np.array([freqs[peak] for peak in duo]) for duo in peaks]
    found = [foun[foun > FREQ - 2 * FREQ_SHIFT] for foun in found]
    if draw:
        plt.plot(freqs, ForTrans[1])
        plt.xlim(17000, 21000)
        plt.show()

    return majorty_decide(decide_multiple(found))


def syncThresh(data, bits):
    dat = dechyperMultyFreq(data, draw=False)
    eps = PEAK_THRESH
    lst = dat - bits
    n_err = [np.count_nonzero(lst == 1), np.count_nonzero(lst == -1)]
    total_err = n_err[0] + n_err[1]
    while total_err != 0 and (n_err[0] == 0 or n_err[1] == 0):
        if n_err[0] != 0:
            eps = eps + eps / 2
        else:
            eps = eps - eps / 2
        dat = dechyperMultyFreq(data, draw=False, thresh=eps)
        lst = dat-bits
        n_err = [np.count_nonzero(lst == 1), np.count_nonzero(lst == -1)]
        total_err = n_err[0] + n_err[1]
    print("total errors: {}".format(total_err))
    return eps
