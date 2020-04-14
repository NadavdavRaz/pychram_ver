from constants import *


###--------------------###
### play from raw data ###
###--------------------###

def play_data(final_sample_list):
    samples = np.asarray(final_sample_list) / MAX_VOLUME
    default_speaker = sc.default_speaker()
    default_speaker.play(samples, samplerate=SAMPLE_RATE)


###--------###
### RECORD ###
###--------###
def record(file_name, record_time=RECORD_TIME):
    fs = SAMPLE_RATE  # this is the frequency sampling; also: 4999, 64000
    seconds = record_time  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Starting: Speak now!")
    sd.wait()  # Wait until recording is finished
    print("finished")
    write(file_name, fs, myrecording)  # Save as WAV file
    # os.startfile("hello.wav")

###--------------------###
### save/load wav file ###
###--------------------###


def save_wave(frame_rate, audio_data, wave_filename):
    try:
        data = np.asarray(audio_data)
        mask = np.mod(data, 1)
        if sum(mask == 0)[0] < data.shape[0] or sum(mask == 0)[1] < data.shape[0]:
            raise Exception('Invalid audio data')
        wavfile.write(wave_filename, frame_rate, data.astype(np.int16))
        return 0
    except KeyboardInterrupt:
        raise
    except:
        return -1


def load_wave(wave_filename):
    try:
        frame_rate, data = wavfile.read(wave_filename)
        if data.dtype == np.uint8:
            data = (data.astype(np.int16) - 128) * 256
        elif data.dtype != np.int16:
            raise Exception('Unhandeled sample width')
        if len(data.shape) == 1:
            data = np.repeat(data, 2)
            data = data.reshape((int(len(data) / 2), 2))
        elif len(data.shape) == 2 and data.shape[1] > 2:
            data = data[:, 0:2]
        data_list = data.tolist()
        return frame_rate, data_list
    except KeyboardInterrupt:
        raise
    except:
        return -1

