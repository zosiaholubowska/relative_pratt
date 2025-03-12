from pratt_exp import load_parameters, load_processors, load_tones, run_pratt
from utils import create_dataframe,plot_results_single_participant

### INSERT PARTICIPANT'S NUMBER HERE
subject = 'sub26'

### LOAD PARAMETERS
DIR, STIM_DIR, RESULTS_DIR, PLOT_DIR, samplerate, table = load_parameters(subject)

### LOAD STIMULI - ABSOLUTE MEASURES
step, shuffled_pairs, conditions, viola_harmonic = load_tones(STIM_DIR)

### LOAD PROCESSORS
proc_list, directions = load_processors(DIR)


### RUN *EXPERIMENT 1* - absolute measures

for cond_index, condition in enumerate(conditions):
    run_pratt(subject, shuffled_pairs, proc_list, table, step, condition, STIM_DIR, cond_index, viola_harmonic)

    print(f'END OF THE BLOCK - {condition}')
    print('Its time for a break')
    inp = input('Do you want to continue? [y/n]')

    if inp == 'n':
        break


### PLOT THE RESULTS
elevation_mapping = {21 : 25.0,
                     22 : 12.5,
                     23 : 0.0,
                     24 : -12.5,
                     25 : -25.0}
create_dataframe(RESULTS_DIR, elevation_mapping)
plot_results_single_participant(subject, RESULTS_DIR, PLOT_DIR)

