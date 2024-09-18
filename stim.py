import pandas
import os
import json
from utils import separate_melodies, notetofreq
import copy
import pickle

# ======= DIRS AND PARAMS

DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'

files = [f for f in os.listdir(STIM_DIR) if '.mid' in f]

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
    stims_trans[i]['key'] = 'c'


trans_dict = {'a' : -3, 'e' : 4, 'g' : 7, 'h' : 11}
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

range_freq = [notetofreq(f) for f in unique]

with open(f'{STIM_DIR}/freqs.pickle', 'wb') as fp:
    pickle.dump(range_freq, fp)
