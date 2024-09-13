import os
from mido import MidiFile
import pandas

# ====== DIRS AND PARAMS

DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'

files = [f for f in os.listdir(STIM_DIR) if '.mid' in f]

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
