from pratt_exp import load_parameters, load_processors, load_tones, run_pratt

### INSERT PARTICIPANT'S NUMBER HERE
subject = 'sub00'

### LOAD PARAMETERS
DIR, STIM_DIR, samplerate, table = load_parameters(subject)

### LOAD PROCESSORS
proc_list, directions = load_processors(DIR)

### LOAD STIMULI - ABSOLUTE MEASURES
step, pairs, conditions = load_tones(STIM_DIR)

### RUN *EXPERIMENT 1* - absolute measures
for condition in conditions:
    run_pratt(subject, pairs, proc_list, table, step, condition, STIM_DIR)

    print(f'END OF THE BLOCK - {condition}')
    print('Its time for a break')
    inp = input('Do you want to continue? (y/n)')

    if inp == 'n':
        break

