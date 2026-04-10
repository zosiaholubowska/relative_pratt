import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import os

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'

data = pandas.read_csv(f'{RESULTS_DIR}/pratt_0azim_2025-03-28_13-59-47.csv')
data['elev_diff'] = data['pred_elev'] - data['true_elev']

plt.figure(figsize=(15, 7))
sns.pointplot(x='midi_note', y='elev_diff', hue='condition', data=data, dodge=True)
plt.xlabel('MIDI')
plt.ylabel('Elevation Difference')
plt.title('Frequency Distribution by Elevation Difference')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/pratts_effect_dnn.png')

