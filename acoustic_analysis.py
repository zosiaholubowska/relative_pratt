import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from utils import notetofreq, create_dataframe
from sklearn.linear_model import LinearRegression
import slab
from utils import notetofreq, get_acoustic_features
import re

# ====== DIRECTORIES

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TONE_DIR = f'{DIR}/stimuli/tones'

conditions = ['flute', 'viola', 'harmoniccomplex']

# ====== Generate the complex tones

"""
range = numpy.arange(55, 109)

for midi in range:
    freq = notetofreq(midi)
    duration = 1.0
    sound = slab.Sound.harmoniccomplex(f0=freq, duration=duration, amplitude=[0, -10, -20, -30, -40, -50, -60, -70])
    sound.write(filename=f'{TONE_DIR}/harmoniccomplex/stim_{midi}_harmonic.wav')
"""

# ====== Compute spectral features:
# spectral centroid, spectral flatness, spectral roll-off

features = ['centroid', 'flatness', 'rolloff']

acoustic_features_df = get_acoustic_features(conditions, features, TONE_DIR)

# get the midi values from the stimulus name
acoustic_features_df['midi'] = acoustic_features_df['stimulus'].apply(lambda x: int(re.search(r"stim_(\d+)_", x).group(1)))

acoustic_features_df['frequency'] = acoustic_features_df['midi'].apply(notetofreq)

fig, axs = plt.subplots(1, 3)
fig.suptitle('Comparison of Spectral Features', fontsize=18)  # Set font size for the main title
fig.set_figheight(10)
fig.set_figwidth(30)

for idx, feature in enumerate(features):
    data = acoustic_features_df[acoustic_features_df['feature'] == feature]
    sns.stripplot(data=data, x='condition', y='value', hue='frequency', hue_norm=(-100, 10000), ax=axs[idx],
                  palette='Spectral')
    sns.boxplot(data=data, x='condition', y='value', fill=False, color='black', ax=axs[idx])
    axs[idx].set_xlabel('Timbre', fontsize=18)  # Increase font size for x-axis labels
    axs[idx].set_ylabel('Ratio' if feature == "flatness" else "Frequency (Hz)",
                        fontsize=18)  # Set y-axis label and size
    axs[idx].set_title(f'Spectral {feature.capitalize()}', fontsize=18)  # Set title size

    axs[idx].tick_params(axis='both', labelsize=16)  # Set font size for tick labels
    axs[idx].legend(title='Frequency (Hz)', fontsize=14, title_fontsize=16)  # Set font sizes for the legend

plt.show()

fig.savefig(f'{PLOT_DIR}/acoustic_features_stimuli.png', dpi=300)
plt.close()

acoustic_features_df.to_csv(f'{RESULTS_DIR}/acoustic_features.csv')