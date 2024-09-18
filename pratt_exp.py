import time
import itertools
from numpy.random import default_rng
import pandas
import slab
import random
import freefield
import os
import sys
import pickle
from utils import shuffle_pairs

# ========== DIRS AND PARAMS
DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'
samplerate = 44828
### ========== ABSOLUTE MEASURES

# =========== SET-UP
directions = [21, 22, 23, 24, 25]
[speaker1] = freefield.pick_speakers(directions[0])
[speaker2] = freefield.pick_speakers(directions[1])
[speaker3] = freefield.pick_speakers(directions[2])
[speaker4] = freefield.pick_speakers(directions[3])
[speaker5] = freefield.pick_speakers(directions[4])

proc_list = [['RX81', 'RX8', f'{DIR}/rcx/piano.rcx'],
                ['RX82', 'RX8', f'{DIR}/rcx/piano.rcx'],
                 ['RP2', 'RP2', f'{DIR}/rcx/button.rcx']]

freefield.initialize('dome', device=proc_list)
freefield.set_logger('warning')


# ========== STIMULI
with open(f'{STIM_DIR}/freqs.pickle', 'rb') as fp:
    range_freq = pickle.load(fp)

pairs = [(sound, speaker) for sound in range_freq for speaker in directions for _ in range(3)]
stims = shuffle_pairs(pairs)

def run_abs(subject, range_freq):
    # TO DO RESULTS DF

    for stim in stims:
        frequency = stim[0]
        duration = 1
        direction = stim[1]
        [curr_speaker] = freefield.pick_speakers(direction)

        freefield.write('f0', frequency, ['RX81', 'RX82'])
        freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])
        freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
        [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
        freefield.write('chan', 99, other_proc)
        time.sleep(8)
        freefield.play()

        # TO DO - add head position log
        # light shows up on the middle speaker
        freefield.write(tag='bitmask', value=1, processors='RX81')
        # laser pointing at the loudspeaker which produces the sound

        #button press to log the response
        response = freefield.read('response', 'RP2', 0)

        time.sleep(1)





# ========== INDUCED CONTEXT

