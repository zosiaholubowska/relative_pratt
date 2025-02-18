import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from librosa import amplitude_to_db

from simulated_data import midi_note
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