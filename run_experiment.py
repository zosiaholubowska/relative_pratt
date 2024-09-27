### LOAD LIBRARIES
import time
import pandas
import slab
import random
import freefield
import os
import pickle
from utils import shuffle_pairs
from datetime import datetime
from pratt_exp import load_parameters, load_processors, load_stimuli_absolute, run_abs, load_stimuli_relative, run_rel

### INSERT PARTICIPANT'S NUMBER HERE
subject = 'sub00'

### LOAD PARAMETERS
DIR, STIM_DIR, samplerate, table = load_parameters(subject)

### LOAD PROCESSORS
proc_list, directions = load_processors(DIR)

### LOAD STIMULI - ABSOLUTE MEASURES
step, tone, stims = load_stimuli_absolute(STIM_DIR, subject)

### RUN *EXPERIMENT - PART 1* - absolute measures
run_abs(subject, stims, proc_list, table, step)

### LOAD STIMULI - RELATIVE MEASURES

step, stims, tone = load_stimuli_relative(STIM_DIR, directions)

### RUN *EXPERIMENT - PART 2* - relative measures

run_rel(subject, stims, STIM_DIR, samplerate, proc_list, step, table)
