import pandas
import os
from utils import separate_melodies, notetofreq, shuffle_pairs
import copy
import pickle
import numpy as np

# ======= DIRS AND PARAMS

DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'

files = [f for f in os.listdir(STIM_DIR) if '.mid' in f]


# ======== CREATE PAIRS
min_tone = 56
max_tone = 107

n = 4  # Number of equally spaced tones you want

tones = np.linspace(min_tone, max_tone, n).round().astype(int)
print(tones)

tones_low = [tone - 1 for tone in tones]
tones_high = [tone + 1 for tone in tones]

all_tones = []
all_tones.extend(tones_low)
all_tones.extend(tones)
all_tones.extend(tones_high)

directions = [21, 22, 23, 24, 25]
directions = [21, 22, 23, 24, 25]


pairs = [(sound, speaker) for sound in all_tones for speaker in directions for _ in range(3)]

with open(f'{STIM_DIR}/tones_sequence.pickle', 'wb') as fp:
    pickle.dump(pairs, fp)


# SHUFFLED SEQUENCE

conditions = ['viola_complex', 'viola', 'flute']
shuffled_pairs = {}
for condition in conditions:
    shuffled_pairs[condition] = shuffle_pairs(pairs)

with open(f'{STIM_DIR}/shuffled_pairs.pickle', 'wb') as fp:
    pickle.dump(shuffled_pairs, fp)

# ======= IMPORT FILES

stims_all = pandas.read_csv(f'{STIM_DIR}/stims_all.csv', sep=",")

separate_stims = separate_melodies(stims_all)

# ====== SAVE SEPARATE MELODIES

for i in range(len(separate_stims)):
    separate_stims[i]['interval'] = ''
    separate_stims[i] = separate_stims[i].reset_index(drop=True)
    for ind in separate_stims[i].index:

        if ind == 0:
            separate_stims[i]['interval'][ind] = 0
        elif separate_stims[i]['freq'][ind] > separate_stims[i]['freq'][ind - 1]:
            separate_stims[i]['interval'][ind] = separate_stims[i]['freq'][ind] / separate_stims[i]['freq'][ind - 1]
        else:
            separate_stims[i]['interval'][ind] = - (separate_stims[i]['freq'][ind - 1] / separate_stims[i]['freq'][ind])
    last_int = round(float(separate_stims[i]['interval'][-1:]), 2)
    separate_stims[i].to_csv(f'{STIM_DIR}/stim_c_{i+1}_{last_int}.csv', index=False)


# ====== TRANSPOSE MELODIES
stims_trans = separate_stims.copy()

for i in range(len(stims_trans)):
    stims_trans[i]['key'] = 'c1'


trans_dict = {'g1' : 7, 'c1' : 12}
for key, value in trans_dict.items():
    print(key, value)
    separate_stims_trans = copy.deepcopy(separate_stims)
    for i in range(len(separate_stims_trans)):
        separate_stims_trans[i]['key'] = key
        for ind in separate_stims_trans[i].index:
            separate_stims_trans[i]['note'][ind] = separate_stims_trans[i]['note'][ind] + value
            separate_stims_trans[i]['freq'][ind] = notetofreq(separate_stims_trans[i]['note'][ind])
            last_int = round(float(separate_stims_trans[i]['interval'][-1:]), 2)
            separate_stims_trans[i].to_csv(f'{STIM_DIR}/stim_{key}_{i + 1}_{last_int}.csv', index=False)
        stims_trans.append(separate_stims_trans[i])





# ======= SAVE DIFFERENT FREQUENCIES
notes = []
for i in range(len(stims_trans)):
    list_notes = stims_trans[i]['note'].to_list()
    first = list_notes[0]
    second_last = list_notes[-2]
    last = list_notes[-1]
    notes.append(first)
    notes.append(second_last)
    notes.append(last)

unique = list(set(notes))
directions = [21, 22, 23, 24, 25]
pairs = [(sound, speaker) for sound in unique for speaker in directions]

stims = shuffle_pairs(pairs)

with open(f'{STIM_DIR}/tones_sequence.pickle', 'wb') as fp:
    pickle.dump(pairs, fp)


# ====== FILE NAMES FOR PIANO AND VIOLIN

viola_tones = [f for f in os.listdir(f'{STIM_DIR}/tones/viola') if '.wav' in f]
directions = [21, 22, 23, 24, 25]
pairs = [(sound, speaker) for sound in piano_tones for speaker in directions]

# ====== COMPILING CHORDS

import slab

STIM_DIR = f'{os.getcwd()}/stimuli/chords'

midi_notes = [55,  56,  57,  72,  73,  74,  89,  90,  91, 106, 107, 108]
duration = 1.0

for midi in midi_notes:
    frequency_1 = notetofreq(midi)
    frequency_3 = notetofreq(midi+4)
    frequency_5 = notetofreq(midi+7)
    sound_1 = slab.Sound.tone(frequency=frequency_1, duration=duration)
    sound_1 = sound_1.ramp(duration=0.01)
    sound_3 = slab.Sound.tone(frequency=frequency_3, duration=duration)
    sound_3 = sound_3.ramp(duration=0.01)
    sound_5 = slab.Sound.tone(frequency=frequency_5, duration=duration)
    sound_5 = sound_5.ramp(duration=0.01)

    combined_sound = sound_1 + sound_3 + sound_5
    combined_sound = combined_sound.ramp(duration=0.05)

    combined_sound.write(f'{STIM_DIR}/harmonic/chord_{midi}.wav')



