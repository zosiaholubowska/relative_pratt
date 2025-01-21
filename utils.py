import os
from mido import MidiFile
import pandas
import random
import slab
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

# ====== DIRS AND PARAMS

DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'
PLOT_DIR = f'{DIR}/plots'

files = [f for f in os.listdir(STIM_DIR) if '.mid' in f]

elevation_mapping = {21 : 25.0,
                     22 : 12.5,
                     23 : 0.0,
                     24 : -12.5,
                     25 : -25.0}

# ====== FUNCTIONS

def notetofreq(note):
    a = 440
    return (a / 32) * (2 ** ((note - 9) / 12))

def get_midi(file):
    mid = MidiFile(f'{STIM_DIR}/{file}')
    midi_dict_list = []
    for track in mid.tracks:
        for event in track:
            event_dict = event.dict()
            event_dict['type'] = event.type
            midi_dict_list.append(event_dict)

        # Create a DataFrame from the list of dictionaries
    midi_df = pandas.DataFrame(midi_dict_list)
    midi_df = midi_df[midi_df['type'] == 'note_on']

    columns_to_keep = ['type', 'time', 'note', 'velocity']  # Add other column names you want to keep
    midi_df = midi_df[columns_to_keep]

    midi_df['time'] /= 480
    midi_df['time'] = midi_df['time'].cumsum()

    midi_df['offset'] = 0

    midi_df['offset'] = midi_df['time'].shift(-1)
    midi_df.loc[midi_df['velocity'] == 0, 'offset'] = midi_df['time']

    midi_df = midi_df.loc[midi_df['velocity'] != 0]

    midi_df['duration'] = midi_df['offset'] - midi_df['time']
    midi_df.drop(columns=['type'], inplace=True)
    midi_df['freq'] = midi_df['note'].apply(notetofreq)
    midi_df.rename(columns={'time': 'onset_sec'}, inplace=True)
    fl = file[:-4]
    midi_df.to_csv(f'{STIM_DIR}/{fl}_all.csv', index=False)

def separate_melodies(df, interval=8):
    # List to store each smaller dataframe
    split_dfs = []

    # Get the maximum onset time to know the number of splits
    max_time = df['onset_sec'].max()

    # Loop through in steps of the interval
    for start in range(0, int(max_time) + 1, interval):
        # Filter the dataframe to only include rows where onset_sec is within the current interval
        subset = df[(df['onset_sec'] >= start) & (df['onset_sec'] < start + interval)]
        split_dfs.append(subset)

    return split_dfs

def shuffle_pairs(pairs):
    temp_pairs = pairs[:]
    random.shuffle(temp_pairs)
    valid_order = False

    while not valid_order:
        valid_order = True
        for i in range(1, len(temp_pairs)):
            if temp_pairs[i][0] == temp_pairs[i-1][0]:
                random.shuffle(temp_pairs)
                valid_order = False
                break
    return temp_pairs


def create_sound(frequency, midi_note, duration, condition, STIM_DIR):
    if condition == 'pure_tone':
        sound = slab.Sound.tone(frequency=frequency, duration=duration)
        sound = sound.ramp(duration=0.01)
    elif condition == 'irn':
        sound = slab.Sound.irn(frequency=frequency, duration=duration)
        sound = sound.ramp(duration=0.01)
    elif condition == 'complex':
        sound = slab.Sound.harmoniccomplex(f0=frequency, duration=duration, amplitude=[0, -10, -20, -30, -40, -50, -60, -70])
        sound = sound.ramp(duration=0.01)
    elif condition == 'piano':
        sound = slab.Sound(f'{STIM_DIR}/tones/piano/stim_{int(midi_note)}_piano.wav')
        sound = sound.ramp(duration=0.01)
    elif condition == 'viola':
        sound = slab.Sound(f'{STIM_DIR}/tones/viola/stim_{int(midi_note)}_viola.wav')
        sound = sound.ramp(duration=0.05)
    elif condition == 'flute':
        sound = slab.Sound(f'{STIM_DIR}/tones/flute/stim_{int(midi_note)}_flute.wav')
        sound = sound.ramp(duration=0.05)

    return sound

def create_dataframe(RESULTS_DIR, elevation_mapping):
    subjects = [f for f in os.listdir(RESULTS_DIR) if f.startswith('sub')]
    dfs = []
    for subject in subjects:
        dir = f'{RESULTS_DIR}/{subject}'
        files = [f for f in os.listdir(dir) if f.startswith('sub')]
        for file in files:
            temp_data = pandas.read_csv(f'{dir}/{file}', sep=',')
            dfs.append(temp_data)

    data = pandas.concat(dfs)
    data['elevation_ls'] = data['direction'].map(elevation_mapping)
    data['azimuth_ls'] = 0.0
    data['elevation_diff'] = data['elevation'] - data['elevation_ls']
    data = data.apply(lambda x: x.astype(str).str.replace('\t', '') if x.dtype == "object" else x)
    data['midi_note'] = data['midi_note'].astype(int)
    data['midi_bin'] = ''
    data['frequency_bin'] = ''
    data = data.reset_index(drop=True)
    for index, row in data.iterrows():
        if 55 <= row['midi_note'] <= 57:
            data.loc[index, 'midi_bin'] = 56
        elif 72 <= row['midi_note'] <= 74:
            data.loc[index, 'midi_bin'] = 73
        elif 89 <= row['midi_note'] <= 91:
            data.loc[index, 'midi_bin'] = 90
        elif 106 <= row['midi_note'] <= 108:
            data.loc[index, 'midi_bin'] = 107
        data.loc[index, 'frequency_bin'] = round(notetofreq(int(data.loc[index, 'midi_bin'])))
    data['interval'] = data.apply(lambda row:
                                  numpy.nan if row.name == 0
                                  else round(row['frequency_bin'] / data.loc[row.name - 1, 'frequency_bin'], 2),
                                  axis=1)

    data['interval'] = data['interval'].apply(lambda x: -(1 / x) if x < 1 else x)
    data['interval'] = round(data['interval'])

    data.to_csv(f'{RESULTS_DIR}/data.csv', index=False)


def plot_results_single_participant(subject, RESULTS_DIR, PLOT_DIR):
    data = pandas.read_csv(f'{RESULTS_DIR}/data.csv')
    sub_data = data[data['subject'] == subject]

    # Absolute effect
    elevation_frequency = sub_data.groupby(['condition', 'subject', 'frequency_bin']).mean('elevation_diff')
    elevation_frequency = elevation_frequency.reset_index()

    # Specify order for frequency_bin
    freq_order = sorted(elevation_frequency['frequency_bin'].unique())  # Adjust sorting if needed

    g = sns.FacetGrid(elevation_frequency, col="condition", col_wrap=2)
    g.map(sns.boxplot, "frequency_bin", "elevation_diff", color='black', order=freq_order,
          boxprops=dict(facecolor='white', color='black'))
    g.map(sns.swarmplot, "frequency_bin", "elevation_diff", order=freq_order)
    g.add_legend()
    g.savefig(f"{PLOT_DIR}/{subject}_absolute_effect.svg", dpi=300)
    plt.show()

    # Relative effect
    elevation_interval = sub_data.groupby(['condition', 'subject', 'interval']).mean('elevation_diff')
    elevation_interval = elevation_interval.reset_index()

    # Specify order for interval
    interval_order = sorted(elevation_interval['interval'].unique())  # Adjust sorting if needed

    g = sns.FacetGrid(elevation_interval, col="condition", col_wrap=2)
    g.map(sns.boxplot, "interval", "elevation_diff", color='black', order=interval_order)
    g.map(sns.swarmplot, "interval", "elevation_diff", order=interval_order)
    g.add_legend()
    g.savefig(f"{PLOT_DIR}/{subject}_relative_effect.svg", dpi=300)
    plt.show()



