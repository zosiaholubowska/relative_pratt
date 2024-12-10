from pratt_exp import load_parameters, load_processors, load_tones, run_pratt
from utils import create_dataframe,plot_results_single_participant

### INSERT PARTICIPANT'S NUMBER HERE
subject = 'sub03'

### LOAD PARAMETERS
DIR, STIM_DIR, RESULTS_DIR, PLOT_DIR, samplerate, table = load_parameters(subject)

### LOAD STIMULI - ABSOLUTE MEASURES
step, shuffled_pairs, conditions = load_tones(STIM_DIR)

### LOAD PROCESSORS
proc_list, directions = load_processors(DIR)

### RUN *EXPERIMENT 1* - absolute measures




### PLOT THE RESULTS
create_dataframe(RESULTS_DIR, elevation_mapping)
plot_results_single_participant(subject, RESULTS_DIR, PLOT_DIR)

