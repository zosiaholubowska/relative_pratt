import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ====== DIRECTORIES

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'

# ====== CONFIGS

elevation_mapping = {21 : 25.0,
                     22 : 12.5,
                     23 : 0.0,
                     24 : -12.5,
                     25 : -25.0}

# ====== CREATE A BIG DATAFRAME

def create_dataframe(RESULTS_DIR):
    subjects = [f for f in os.listdir(RESULTS_DIR) if f.startswith('Sub')]
    dfs = []
    for subject in subjects:
        dir = f'{RESULTS_DIR}/{subject}'
        files = os.listdir(dir)
        for file in files:
            temp_data = pandas.read_csv(f'{dir}/{file}', sep=',')
            dfs.append(temp_data)

    data = pandas.concat(dfs)
    data['elevation_ls'] = data['direction'].map(elevation_mapping)
    data['azimuth_ls'] = 0.0
    data['elevation_diff'] = data['elevation'] - data['elevation_ls']
    data.to_csv(f'{RESULTS_DIR}/data.csv', index=False)

# == FILTER DATA FOR SINGLE PARTICIPANT
def plot_boxplot(subject):
    data = pandas.read_csv(f'{RESULTS_DIR}/data.csv')
    sub_data = data[data['subject']==subject]


    # PLOTS
    # Do we want to plot only perceived elevation or difference?
    # Plot: boxplots for difference for frequencies

    conditions = sub_data['condition'].unique()

    # Set up the number of subplots
    num_conditions = len(conditions)
    fig, axes = plt.subplots(nrows=1, ncols=num_conditions, figsize=(5 * num_conditions, 6), sharey=True)

    for ax, condition in zip(axes, conditions):
        condition_data = sub_data[sub_data['condition'] == condition]

        ax.boxplot([condition_data[condition_data['midi_note'] == note]['elevation'] for note in
                    condition_data['midi_note'].unique()],
                   labels=condition_data['midi_note'].unique())

        ax.set_title(f"Condition: {condition}")
        ax.set_xlabel('MIDI Note')
        ax.set_ylabel('Elevation')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{PLOT_DIR}/{subject}_boxplot.png', dpi=300)

# Plot: difference in slope

def plot_slope(subject):
    data = pandas.read_csv(f'{RESULTS_DIR}/data.csv')
    sub_data = data[data['subject'] == subject]
    palette = sns.color_palette('Set1')
    mean_data = sub_data.groupby(['condition', 'midi_note']).agg({'elevation': 'mean'}).reset_index()
    conditions = sub_data['condition'].unique()
    plt.figure(figsize=(8, 6))

    for i, condition in enumerate(conditions):
        condition_data = mean_data[mean_data['condition'] == condition]

        plt.scatter(condition_data['midi_note'], condition_data['elevation'], label=condition, color=palette[i])

        plt.plot(condition_data['midi_note'], condition_data['elevation'], color=palette[i])

    # Add labels and legend
    plt.xlabel('MIDI Note')
    plt.ylabel('Mean Elevation')
    plt.legend(title='Condition')
    plt.tight_layout()

    plt.show()
    plt.savefig(f'{PLOT_DIR}/{subject}_slope.png', dpi=300)