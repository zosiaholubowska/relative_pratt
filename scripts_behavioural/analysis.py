import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_dataframe
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams['svg.fonttype'] = 'none'

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

palette = {
    'flute': '#e22c1f',           # red
    'complex': '#E5A09C',         # pink
    'viola': '#2e33a6',           # blue
    'viola_complex': '#9FA0CC'    # light blue
}

# ====== CREATE MAIN DATAFRAME

#create_dataframe(RESULTS_DIR, elevation_mapping)

# ====== DOWNLOAD DATA
data = pandas.read_csv(f'{RESULTS_DIR}/data.csv')
data = data[~data['subject'].str.contains('pilot')]
data = data[~data['subject'].str.contains('00')]

acoustic_features_temp = pandas.read_csv(f'{RESULTS_DIR}/acoustic_features_orthogonalised.csv', index_col=0)

acoustic_features = acoustic_features_temp.pivot_table(
    index=['stimulus', 'condition', 'midi', 'frequency'],
    columns='feature',
    values=['value_ortho_freq', 'value'],
    aggfunc='first'
).reset_index()
acoustic_features.columns = ['_'.join(filter(None, col)) for col in acoustic_features.columns]

acoustic_features['condition'] = acoustic_features['condition'].replace('harmoniccomplex', 'complex')

# ====== MATCH THE DATA WITH ACOUSTIC FEATURES

merged_data = data.merge(
    acoustic_features,
    left_on=['condition', 'midi_note'],
    right_on=['condition', 'midi'],
    how='left'
)
merged_data = merged_data.drop(columns=['frequency_y'])
merged_data = merged_data.rename(columns={'frequency_x': 'frequency'})

main_df = merged_data.copy()
main_df['frequency'] = main_df['frequency'].round(0)
main_df.to_csv(f'{RESULTS_DIR}/participants_data_with_acoustics.csv')

del merged_data, data, acoustic_features, acoustic_features_temp

# ====== BASE MODEL
# We want to confirm the Pratt's effect by predicting the elevation difference with f0.
# elevation_diff ~ frequency_bin

model_freq = smf.mixedlm('elevation_diff ~ frequency', main_df, groups='subject')
model_freq_fitted = model_freq.fit()
print(model_freq_fitted.summary())

# plot the results of the model

plt.figure(figsize=(10, 6))
sns.scatterplot(x='frequency', y='elevation_diff', data=main_df, alpha=0.3, color='lightblue')
sns.regplot(x='frequency', y='elevation_diff', data=main_df, scatter=False, color='red')
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.legend()

# ====== CONDITION EFFECT

model_cond = smf.mixedlm('elevation_diff ~ frequency * C(condition)', main_df, groups='subject')
model_cond_fitted = model_cond.fit()
print(model_cond_fitted.summary())

# plot the results of the model
palette = ['#e22c1f', '#ffce3a', '#2e33a6', '#5cb41b']
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.pointplot(x='frequency', y='elevation_diff', hue='condition', palette=palette, data=main_df)
plt.title('Elevation Difference vs. Frequency by Condition', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.subplot(2, 1, 2)
conditions = main_df['condition'].unique()
for idx, condition in enumerate(conditions):
    subset = main_df[main_df['condition'] == condition]
    sns.regplot(x='frequency', y='elevation_diff', data=subset, scatter=False, color=palette[idx], label=f'{condition} Trend')


plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.legend(title='Condition')
plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/elevation_diff.png')
plt.close()

main_df['music_based'] = numpy.where(
    main_df['condition'].str.contains('flute|viola'), 'musical', 'non_musical'
)

main_df['centroid_based'] = numpy.where(
    main_df['condition'].str.contains('viola|viola_complex'), 'high', 'low'
)

model_musical = smf.mixedlm('elevation_diff ~ frequency * C(music_based) + value_ortho_freq_centroid', main_df_filtered, groups='subject')
model_musical_fitted = model_musical.fit()
print(model_musical_fitted.summary())

model_centroid = smf.mixedlm('elevation_diff ~ frequency * C(centroid_based) + value_ortho_freq_centroid', main_df_filtered, groups='subject')
model_centroid_fitted = model_centroid.fit()
print(model_centroid_fitted.summary())

#### because we have the step for the last frequency bin, i would filter it out

main_df_filtered = main_df[main_df['frequency'] < 3000]
model_cond_filtered = smf.mixedlm('elevation_diff ~ frequency * C(condition)', main_df_filtered, groups='subject')
model_cond_filtered_fitted = model_cond_filtered.fit()
print(model_cond_filtered_fitted.summary())

plt.figure(figsize=(8, 10))
plt.subplot(2, 1, 1)
sns.pointplot(x='frequency', y='elevation_diff', hue='condition', palette=palette, data=main_df_filtered)
plt.title('Elevation Difference vs. Frequency by Condition', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.subplot(2, 1, 2)
conditions = main_df['condition'].unique()
for idx, condition in enumerate(conditions):
    subset = main_df_filtered[main_df_filtered['condition'] == condition]
    sns.regplot(x='frequency', y='elevation_diff', data=subset, color=palette[condition], scatter=False, label=f'{condition} Trend')


plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.legend(title='Condition')
plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/elevation_diff_filtered.svg')
plt.close()

# ====== ACOUSTIC FEATURES EFFECT

model_af = smf.mixedlm('elevation_diff ~ frequency * C(condition) + value_ortho_freq_centroid',
                        data=main_df_filtered,
                        groups=main_df_filtered['subject'])

model_af_fitted = model_af.fit()
print(model_af_fitted.summary())

model_af_inter = smf.mixedlm('elevation_diff ~ frequency * C(condition) + value_ortho_freq_centroid * C(condition)',
                        data=main_df_filtered,
                        groups=main_df_filtered['subject'])

model_af_inter_fitted = model_af_inter.fit()
print(model_af_inter_fitted.summary())

# correlation between centroid and elevation difference for each fundamental frequency separatelly

from scipy.stats import pearsonr

centroid_corrs = []

for freq, group in main_df_filtered.groupby('frequency_bin'):
    if len(group) >= 5:
        r, p = pearsonr(group['value_centroid'], group['elevation_diff'])
        centroid_corrs.append({'frequency_bin': freq, 'r': r, 'p_value': p, 'n': len(group)})

centroid_corrs_df = pandas.DataFrame(centroid_corrs)

print(centroid_corrs_df.sort_values('frequency_bin'))

main_df_filtered['freq_bin'] = main_df_filtered['frequency'].round(0).astype(int)

# Plot
g = sns.FacetGrid(main_df_filtered, col='frequency_bin', col_wrap=3, height=3.5, sharex=False, sharey=False)
g.map(sns.scatterplot, 'value_centroid', 'elevation_diff', alpha=0.3, color='grey')
g.map(sns.regplot, 'value_centroid', 'elevation_diff', scatter=False, color='black')

g.set_axis_labels('Spectral Centroid', 'Elevation Difference')
g.set_titles('f₀ = {col_name} Hz')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Centroid vs. Elevation Difference by Frequency')
plt.show()
plt.savefig(f'{PLOT_DIR}/centroid_vs_elevation_diff.svg')


# 4. Elevation Difference by Rolloff and Centroid
plt.figure(figsize=(12, 6))
plt.subplot(121)
conditions = main_df['condition'].unique()
for condition in conditions:
    subset = main_df[main_df['condition'] == condition]
    sns.regplot(x='value_ortho_freq_centroid', y='elevation_diff', data=subset, scatter=False, label=condition)
plt.title('Elevation Difference by Condition')
plt.xlabel('Centroid (orthogonalised)')
plt.ylabel('Elevation Difference')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()

plt.subplot(122)
for condition in conditions:
    subset = main_df[main_df['condition'] == condition]
    sns.regplot(x='value_ortho_freq_rolloff', y='elevation_diff', data=subset, scatter=False, label=condition)
plt.title('Elevation Difference by Condition')
plt.xlabel('Rolloff (orthogonalised)')
plt.ylabel('Elevation Difference')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/orthogonalised_acoustic_elevation_diff.png', dpi=300)
plt.close()

# -------------- #
# slope analysis #
# -------------- #

slope_results = []

for subject in main_df_filtered['subject'].unique():
    subject_data = main_df_filtered[main_df_filtered['subject'] == subject]

    # General frequency slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subject_data['frequency'],
        subject_data['elevation_diff']
    )

    slope_results.append({
        'subject': subject,
        'condition': 'overall',
        'x_variable': 'frequency',
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value
    })

grouped = main_df_filtered.groupby(['subject', 'condition'])

for (subject, condition), group in grouped:
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        group['frequency'],
        group['elevation_diff']
    )

    # Store results
    slope_results.append({
        'subject': subject,
        'condition': condition,
        'x_variable': 'frequency',
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value
    })

    # Convert results to DataFrame
slopes_df = pandas.DataFrame(slope_results)

statistical_results = {}

# Prepare data for ANOVA
slope_data_filtered = slopes_df[slopes_df['condition'] != 'overall']
x_var = 'frequency'
var_slopes = slope_data_filtered[slope_data_filtered['x_variable'] == x_var]
groups = [group['slope'].values for _, group in var_slopes.groupby('condition')]
f_stat, p_value = stats.f_oneway(*groups)

conditions = []
slopes = []
for condition in slopes_df['condition'].unique():
    condition_slopes = slopes_df[slopes_df['condition'] == condition]['slope']
    conditions.extend([condition] * len(condition_slopes))
    slopes.extend(condition_slopes)

    # One-way ANOVA
    groups = [var_slopes[var_slopes['condition'] == cond]['slope'] for cond in var_slopes['condition'].unique()]
    f_statistic, p_value = stats.f_oneway(*groups)

    # Tukey HSD for post-hoc pairwise comparisons
    tukey_results = pairwise_tukeyhsd(slopes, conditions)

    # Store results
    statistical_results[x_var] = {
        'anova': {
            'f_statistic': f_statistic,
            'p_value': p_value
        },
        'tukey_results': tukey_results
    }

print("\n--- Statistical Comparisons ---")
for x_var, results in statistical_results.items():
    print(f"\nX-Variable: {x_var}")
    print("ANOVA Results:")
    print(f"F-statistic: {results['anova']['f_statistic']:.4f}")
    print(f"p-value: {results['anova']['p_value']:.4f}")
    print("\nTukey HSD Pairwise Comparisons:")
    print(results['tukey_results'])

plt.figure(figsize=(15, 5))
for i, x_var in enumerate(x_variables, 1):
    plt.subplot(1, 3, i)
    var_slopes = slopes_df[slopes_df['x_variable'] == x_var]
    sns.boxplot(x='condition', y='slope', data=var_slopes)
    plt.title(f'Slopes for {x_var}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


### ======= RELATIVE EFFECT

main_df_filtered = main_df[main_df['frequency'] < 3000]
#main_df_filtered = main_df_filtered[main_df_filtered['frequency'] > 300]
model_interval_filtered = smf.mixedlm('elevation_diff ~ frequency + interval * condition', main_df_filtered, groups='subject')
model_interval_filtered_fitted = model_interval_filtered.fit()
print(model_interval_filtered_fitted.summary())

r = stats.pearsonr(main_df_filtered['frequency'], main_df_filtered['interval'])

plt.subplot()
sns.regplot(x='interval', y='elevation_diff', data=main_df_filtered, scatter=False, color='black')
plt.scatter(main_df_filtered['interval'], main_df_filtered['elevation_diff'], color='gray')
plt.savefig(f'{PLOT_DIR}/elevation_interval.svg', dpi=300)

plt.subplot()
conditions = main_df_filtered['condition'].unique()
for condition in conditions:
    subset = main_df_filtered[main_df_filtered['condition'] == condition]
    sns.regplot(x='interval', y='elevation_diff', data=subset, scatter=False, label=condition)
plt.title('Elevation Difference by Condition')
plt.xlabel('Interval')
plt.ylabel('Elevation Difference')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()



