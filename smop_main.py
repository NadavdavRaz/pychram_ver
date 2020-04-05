import modulating_functions as mf
import demodulating_functions as df
import general_funcrions as gf
import numpy as np
from inspect import getmembers, isfunction

global FREQ
global FM_PM
global DURATION_PM
global SAMPLE_RATE


SAMPLE_RATE = 44000  # 1/Sec
BASIC_TIME = 5000 / SAMPLE_RATE  # ~0.1133 Sec
FREQ = 18500  # Hz
FREQ_SHIFT = 1000

def initilize_global_vars():
    mf.SAMPLE_RATE = SAMPLE_RATE
    mf.FREQ = FREQ
    df.SAMPLE_RATE = SAMPLE_RATE
    df.FREQ = FREQ
    gf.SAMPLE_RATE = SAMPLE_RATE


modulating_f = [o[0] for o in getmembers(mf) if isfunction(o[1])]
demodulating_f = [o[0] for o in getmembers(df) if isfunction(o[1])]
general_f = [o[0] for o in getmembers(gf) if isfunction(o[1])]

if __name__=="__main__":
    initilize_global_vars()
    bits = np.array([1,0,1,0,1,0,1,1,0,0,0,1])
    to_send = mf.multiply(bits, 3)
    final_sample_list = mf.modulation(to_send, mode="FM")
    gf.save_wave(SAMPLE_RATE, final_sample_list, 'test_FM_NEW.wav')

