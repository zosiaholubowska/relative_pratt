import os
import matplotlib.pyplot as plt
import seaborn as sns
from librosa import amplitude_to_db
from utils import notetofreq, get_acoustic_features
from sklearn.linear_model import LinearRegression
import slab
import numpy
import pickle
import re

slab.set_default_samplerate(44828)
# ====== DIRECTORIES

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TONE_DIR = f'{DIR}/stimuli/tones'
STIM_DIR = f'{DIR}/stimuli'

conditions = ['flute', 'viola', 'harmoniccomplex', 'viola_complex']

# ====== Generate the complex tones

"""
range = numpy.arange(55, 109)
# harmonic complex
for midi in range:
    print(midi)
    freq = notetofreq(midi)
    duration = 1.0
    sound = slab.Sound.harmoniccomplex(f0=freq, duration=duration, amplitude=[0, -10, -20, -30, -40, -50, -60, -70])
    sound = sound.ramp(duration=0.01)
    sound.write(filename=f'{TONE_DIR}/harmoniccomplex/stim_{midi}_harmonic.wav')
# viola complex

with open(f'{STIM_DIR}/tones/viola_harmonic.pkl', 'rb') as file:
    viola_harmonic = pickle.load(file)

for midi in range:
    print(midi)
    freq = notetofreq(midi)
    duration = 1.0
    harmonic_complex = slab.Sound.silence(duration=duration)
    freq_peaks = viola_harmonic[midi]['freq_peaks']
    amplitude = viola_harmonic[midi]['amplitude']
    # Loop through frequencies and corresponding amplitude levels
    for freq, amp in zip(freq_peaks, amplitude):
        tone = slab.Sound.tone(frequency=freq, duration=1.0)
        tone.level = 75 + amp  # Set the amplitude level
        harmonic_complex += tone

    sound = harmonic_complex
    sound = sound.ramp(duration=0.05)
    sound.write(filename=f'{TONE_DIR}/viola_complex/stim_{midi}_viola_complex.wav')
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


# ======== Analysis of harmonics
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import find_peaks
import time
import slab
import pickle
import re

slab.set_default_samplerate(44828)

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TONE_DIR = f'{DIR}/stimuli/tones'

viola_tones = [f for f in os.listdir(f'{TONE_DIR}/viola') if 'stim' in f]
flute_tones = [f for f in os.listdir(f'{TONE_DIR}/flute') if 'stim' in f]
complex_tones = [f for f in os.listdir(f'{TONE_DIR}/harmoniccomplex') if 'stim' in f]




def analyze_harmonics(direction, condition, midi_note):
    """
    Analyze harmonics in an audio recording.

    Parameters:
        direction (str): Directory where the audio file is stored.
        condition (str): Subdirectory indicating the condition (e.g., 'viola').
        midi_note (int): MIDI note number corresponding to the sound.
    """
    # Construct the filename
    file_name = f'stim_{midi_note}_{condition}.wav'
    file_path = os.path.join(direction, condition, file_name)

    # Load the audio file
    audio, sr = librosa.load(file_path)

    # Compute the FFT and frequency spectrum
    fft_spectrum = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / sr)
    magnitude = np.abs(fft_spectrum)

    # Find harmonic frequencies
    peaks, _ = find_peaks(magnitude[:len(magnitude) // 2], distance=450, height=0.05 * np.max(magnitude))
    harmonic_frequencies = frequencies[peaks]

    # Plot the frequency spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2])
    plt.scatter(frequencies[peaks], magnitude[peaks], color='red')  # Mark peaks
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Frequency Spectrum: {midi_note}, {condition}')
    plt.show()
    plt.savefig(f'{PLOT_DIR}/stimuli/freq_spectrum_{condition}_{midi_note}.png', dpi=300)

    # Compute and display the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {midi_note}, {condition}')
    plt.show()
    plt.savefig(f'{PLOT_DIR}/stimuli/spectrogram_{condition}_{midi_note}.png', dpi=300)

    normalized_magnitudes = magnitude[peaks] / magnitude[peaks][0]
    amplitudes_db = 20 * np.log10(normalized_magnitudes)

    return frequencies[peaks], amplitudes_db

viola_harmonic = {}

for tone in viola_tones:
    midi_note = int(re.search(r'stim_(\d+)_viola', tone).group(1))
    freq_peaks, amplitude_to_db = analyze_harmonics(TONE_DIR, 'viola', midi_note)
    viola_harmonic[midi_note] = {'freq_peaks': freq_peaks, 'amplitude': amplitude_to_db}

# Save the dictionary as a pickle file
with open(f'{TONE_DIR}/viola_harmonic.pkl', 'wb') as file:
    pickle.dump(viola_harmonic, file)

harmonic_complex = slab.Sound.silence(duration=1.0)

# Loop through frequencies and corresponding amplitude levels
for freq, amp in zip(freq_peaks, amplitude_to_db):
    tone = slab.Sound.tone(frequency=freq, duration=1.0)
    tone.level = 75 + amp  # Set the amplitude level
    harmonic_complex += tone  # Sum the harmonics
    #harmonic_complex.play()
    time.sleep(1)
    #harmonic_complex.waveform()

# Play the harmonic complex tone
harmonic_complex.play()

harmonic_complex.spectrum()



#### Orthogonalise the acoustic features

features = acoustic_features_df['feature'].unique()
conditions = acoustic_features_df['condition'].unique()

result_df = acoustic_features_df.copy()
result_df['value_ortho'] = None

features_to_orthogonalize = ['centroid', 'rolloff']

for condition in conditions:
    for feature in features_to_orthogonalize:

        mask = (result_df['condition'] == condition) & (result_df['feature'] == feature)

        # Skip if no data for this combination
        if not any(mask):
            continue

        # Get the subset of data
        subset = result_df.loc[mask].copy()

        # Prepare the data for regression
        X = subset['frequency'].values.reshape(-1, 1)
        y = subset['value'].values

        # Perform regression and calculate residuals
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        orthogonalized = y - y_pred

        # Store the orthogonalized values
        result_df.loc[mask, 'value_ortho'] = orthogonalized


result_df.to_csv(f'{RESULTS_DIR}/acoustic_features_orthogonalised.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create a figure with subplots
fig, axes = plt.subplots(2, len(conditions), figsize=(5 * len(conditions), 10), sharey='row')

# If there's only one condition, wrap axes in a list for consistent indexing
if len(conditions) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

# Plot for each condition
for i, condition in enumerate(conditions):
    # Get condition data where feature is centroid
    condition_data = result_df[(result_df['condition'] == condition) &
                               (result_df['feature'] == 'centroid')]

    # Original centroid vs frequency
    ax_top = axes[0, i]
    sns.scatterplot(
        x='frequency',
        y='value',
        data=condition_data,
        alpha=0.7,
        color='blue',
        ax=ax_top
    )

    # Add regression line to show relationship
    x = condition_data['frequency'].values
    y = condition_data['value'].values

    # Add regression line only if there are enough data points
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax_top.plot(x, p(x), 'r--', alpha=0.8)

    # Calculate correlation coefficient safely
    try:
        # Filter out NaN values
        valid_data = condition_data.dropna(subset=['frequency', 'value'])
        if len(valid_data) > 1:  # Ensure there are at least 2 points
            corr = np.corrcoef(valid_data['frequency'], valid_data['value'])[0, 1]
            corr_text = f'r = {corr:.2f}'
        else:
            corr_text = 'insufficient data'
    except:
        corr_text = 'correlation error'

    ax_top.set_title(f'{condition}: Original Centroid \n{corr_text}')
    ax_top.set_xlabel('Frequency (Hz)')
    ax_top.set_ylabel('Centroid (Hz)')

    # Orthogonalized centroid vs frequency
    ax_bottom = axes[1, i]
    sns.scatterplot(
        x='frequency',
        y='value_ortho',
        data=condition_data,
        alpha=0.7,
        color='green',
        ax=ax_bottom
    )

    # Add horizontal line at y=0 to show mean of orthogonalized values
    ax_bottom.axhline(y=0, color='r', linestyle='--', alpha=0.8)

    # Calculate correlation coefficient for orthogonalized values safely
    try:
        # Filter out NaN values
        valid_data = condition_data.dropna(subset=['frequency', 'value_ortho'])
        if len(valid_data) > 1:  # Ensure there are at least 2 points
            corr_ortho = valid_data['frequency'].corr(valid_data['value_ortho'])
            corr_text = f'r = {corr_ortho:.2f}'
        else:
            corr_text = 'insufficient data'
    except:
        corr_text = 'correlation error'

    ax_bottom.set_title(f'{condition}: Orthogonalized\n{corr_text}')
    ax_bottom.set_xlabel('Frequency (Hz)')
    ax_bottom.set_ylabel('Orthogonalized Centroid')

plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/stimuli/original_orthogonalised_centroid.png', dpi=300)