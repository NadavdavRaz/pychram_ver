import numpy as np
from scipy.io import wavfile
import soundfile as sf
import soundcard as sc
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mp
import os
import wave
import sys
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import hilbert
import simpleaudio as sa


global FREQ
global FM_PM
global DURATION_PM
global SAMPLE_RATE

MAX_VOLUME = 32767
PI = np.pi
SAMPLE_RATE = 44000  # 1/Sec
BASIC_TIME = 5000 / SAMPLE_RATE  # ~0.1133 Sec
FREQ = 18500  # Hz
FREQ_SHIFT = 1000
samples = np.array(0.).astype(np.float32)
CHUNK_SIZE = 50
RECORD_TIME = 4
CHUNK_SIZE_FM = 250


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
