import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from utils import notetofreq

# ====== DIRECTORIES

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'

subject = 'sub00'

# ====== CONFIGS

elevation_mapping = {21 : 25.0,
                     22 : 12.5,
                     23 : 0.0,
                     24 : -12.5,
                     25 : -25.0}

# ====== CREATE A BIG DATAFRAME

def create_dataframe(RESULTS_DIR):
    subjects = [f for f in os.listdir(RESULTS_DIR) if f.startswith('sub')]
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
    data = data.apply(lambda x: x.str.replace('\t', '') if x.dtype == "object" else x)
    data['midi_bin'] = ''
    data['frequency_bin'] = ''
    data = data.reset_index()
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
                                  numpy.nan if row['idx'] == 0
                                  else round((row['frequency_bin'] / data.loc[row.name - 1, 'frequency_bin']),2),
                                  axis=1)
    data.to_csv(f'{RESULTS_DIR}/data.csv', index=False)


# Plot - slopes
data = pandas.read_csv(f'{RESULTS_DIR}/data.csv')
sub_data = data[data['subject']==subject]
plt.figure(figsize=(12, 6))
g = sns.lmplot(x='frequency_bin', y='elevation', hue='condition', data=data, height=6, aspect=2)
g.set_axis_labels('Frequency (Hz)', 'Perceived Elevation (degrees)')
plt.show()
plt.savefig(f'{PLOT_DIR}/pratts_effect_plot.svg')


# Plot - boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='frequency_bin', y='elevation', hue='condition', data=data)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Perceived Elevation (degrees)')
plt.title('Boxplot of Elevation by Frequency Bin and Condition')
plt.show()
plt.savefig(f'{PLOT_DIR}/pratts_effect_boxplot.svg')

data_grouped = data.groupby(['condition', 'frequency_bin']).mean('elevation_diff')

plt.figure(figsize=(12, 6))
sns.boxplot(x='interval', y='elevation_diff', data=data[data['condition']=='flute'])
plt.xlabel('Interval')
plt.ylabel('Perceived Elevation (degrees)')
plt.title('Boxplot of Elevation by Interval and Condition')
plt.show()
plt.savefig(f'{PLOT_DIR}/relative_pratts_effect_boxplot.svg')

data_grouped = data.groupby(['condition', 'interval']).mean('elevation_diff')

###############################################
def calculate_slope(group):
    X = group['frequency']
    y = group['elevation']

    # Add a constant for the intercept in the model
    X = sm.add_constant(X)

    # Fit the OLS regression model
    model = sm.OLS(y, X).fit()

    # Return the slope of the regression
    return pandas.Series({'slope': model.params[1]})


slopes = data.groupby(['subject', 'condition']).apply(calculate_slope).reset_index()

# Plot using seaborn, with direction on the x-axis, slope on the y-axis, and hue representing condition
plt.figure(figsize=(12, 6))
sns.boxplot(x='condition', y='slope', data=slopes)

# Set axis labels and title
plt.xlabel('Direction')
plt.ylabel('Slope')
plt.title('Slope of Perceived Elevation vs Frequency by Direction and Condition')

# Show plot
plt.tight_layout()
plt.show()


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
    mean_data = sub_data.groupby(['condition', 'midi_note']).agg({'elevation_diff': 'mean'}).reset_index()
    conditions = sub_data['condition'].unique()
    plt.figure(figsize=(8, 6))

    for i, condition in enumerate(conditions):
        condition_data = mean_data[mean_data['condition'] == condition]

        plt.scatter(condition_data['midi_note'], condition_data['elevation_diff'], label=condition, color=palette[i])

        plt.plot(condition_data['midi_note'], condition_data['elevation_diff'], color=palette[i])

    # Add labels and legend
    plt.xlabel('MIDI Note')
    plt.ylabel('Mean Elevation')
    plt.legend(title='Condition')
    plt.tight_layout()

    plt.show()
    plt.savefig(f'{PLOT_DIR}/{subject}_slope.png', dpi=300)