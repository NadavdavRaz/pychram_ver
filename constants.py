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
import queue


MAX_VOLUME = 32767
PI = np.pi
SAMPLE_RATE = 44000  # 1/Sec
SAMPLES_PER_SYMBOL = 10000
BASIC_TIME = SAMPLES_PER_SYMBOL / SAMPLE_RATE  # ~0.1133 Sec
FREQ = 18500  # Hz
FREQ_SHIFT = 200
samples = np.array(0.).astype(np.float32)
CHUNK_SIZE = 2000
RECORD_TIME = 7
CHUNK_SIZE_FM = 250
RATIO = int(SAMPLES_PER_SYMBOL / CHUNK_SIZE)
symbol_dict = {"01": FREQ_SHIFT, "10": -FREQ_SHIFT, "00": 5*FREQ_SHIFT}
NUM_OF_FREQS = 5
PEAK_THRESH = 100
BITS = np.array([0,0,1,0, 1, 1, 1, 0, 0,1, 1,1,1,1,1,1,0,0,0,0] * 6)


### PM constants ###
FM_PM = 400  # frequency of modulating signal
DURATION_PM = 2 / FM_PM  # 0.01#duration of the signal
ALPHA = PI / 2  # 0.3 #amplitude of modulating signal
THETA = 0  # phase offset of modulating signal
BETA = 0 # constant carrier phase offset