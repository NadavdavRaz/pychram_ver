from constants import *
from modulating_functions import *
from demodulating_functions import *
from general_functions import *

def check_if_rellevant(que_in,que_out, chunk_size=50, thresh=0.02):
    chunk_data = []
    while True:
        chunk_data = que_in.get() # chunk_data + [que_in.get() for _ in range(chunk_size - len(chunk_data))]
        chunk_data = butter_highpass_filter(chunk_data, FREQ, SAMPLE_RATE, order=8)

        ### Do FFT by chunks
        freq = np.fft.fftfreq(chunk_size, d=1 / SAMPLE_RATE)
        fft = np.abs(np.fft.fft(chunk_data))
        plt.plot(freq, fft)
        plt.show()

        ### Integral over relevant freqs
        cut_index = 20
        integral = np.trapz(fft[cut_index:chunk_size // 2], freq[cut_index:chunk_size // 2])
        integral = integral / (-freq[chunk_size // 2] + freq[cut_index])
        if que_in.qsize() < chunk_size:
            return
        print(integral)
        # print(print(len(chunk_data)))
        ### Filter by thresh
        if integral > thresh:
            while len(chunk_data) <= SAMPLES_PER_SYMBOL*30:
                if que_in.qsize() < chunk_size:
                    return
                if not que_in.empty():
                    chunk_data = np.concatenate((chunk_data, que_in.get()))
            que_out.put(chunk_data)
            # chunk_data = chunk_data[CHUNK_SIZE:]

que_in = queue.Queue()
que_out = queue.Queue()
alist = np.array([])

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    a = indata[:, 1]
    que_in.put(a)


stream = sd.InputStream(channels=2,blocksize=50,
    samplerate=SAMPLE_RATE, callback=audio_callback)

with stream:
    print("start: ")
    sd.sleep(int(5 * 1000))
print("finish")

# check_if_rellevant(que_in, que_out)
# print(que_out.qsize())
result = []
[result.extend(el) for el in list(que_in.queue)]
final_sample_list = np.asarray([[res, res] for res in result])

# print(final_sample_list[:10])
# print(save_wave(SAMPLE_RATE, final_sample_list, "testest.wav"))
wavfile.write("testim.wav", SAMPLE_RATE, final_sample_list)
lala = np.asarray(result)
# cut_irellevant(lala, draw=True)