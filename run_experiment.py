from pratt_exp import load_parameters, load_processors, load_tones, run_pratt
#from analysis import create_dataframe, plot_boxplot, plot_slope

### INSERT PARTICIPANT'S NUMBER HERE
subject = 'sub06-pilot'

### LOAD PARAMETERS
DIR, STIM_DIR, RESULTS_DIR, samplerate, table = load_parameters(subject)

### LOAD STIMULI - ABSOLUTE MEASURES
step, shuffled_pairs, conditions = load_tones(STIM_DIR)

### LOAD PROCESSORS
proc_list, directions = load_processors(DIR)

### RUN *EXPERIMENT 1* - absolute measures

for cond_index, condition in enumerate(conditions):
    run_pratt(subject, shuffled_pairs, proc_list, table, step, condition, STIM_DIR, cond_index)

    print(f'END OF THE BLOCK - {condition}')
    print('Its time for a break')
    inp = input('Do you want to continue? (y/n)')

    if inp == 'n':
        break


### PLOT THE RESULTS
# create_dataframe(RESULTS_DIR)
# plot_slope(subject)
# plot_boxplot(subject)

