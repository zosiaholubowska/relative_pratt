import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Parameters
subjects = ['sub01']
conditions = ['pure_tone', 'irn', 'viola', 'flute', 'piano']
idx_range = range(34)  # idx from 0 to 33
directions = [21, 22, 23, 24, 25]

# MIDI to frequency conversion function
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

# Azimuth and elevation generator
def random_azimuth_elevation():
    return np.random.uniform(-3, 3), np.random.uniform(-25, 25)

# Timestamp generator
def generate_timestamps(start_time, num_samples):
    return [start_time + timedelta(seconds=i) for i in range(num_samples)]

# Create data
data = []
start_time = datetime.now()

for subject in subjects:
    for condition in conditions:
        for idx in idx_range:
            midi_note = np.random.randint(55, 88)  # Random MIDI note (60 to 80)
            frequency = midi_to_freq(midi_note)
            direction = np.random.choice(directions)
            azimuth, elevation = random_azimuth_elevation()
            timestamp = start_time + timedelta(seconds=len(data))  # Simulating time per entry
            data.append([timestamp, subject, condition, idx, midi_note, frequency, direction, "NA", azimuth, elevation])

# Convert to DataFrame
columns = ['timestamp', 'subject', 'condition', 'idx', 'midi_note', 'frequency', 'direction', 'interval', 'azimuth', 'elevation']
df = pd.DataFrame(data, columns=columns)

DIR = os.getcwd()
# Save to txt file
df.to_csv(f'{DIR}/Results/sub00/simulated_data.txt', sep='\t', index=False)
