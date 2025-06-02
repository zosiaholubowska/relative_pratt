import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_dataframe
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.pointplot(x='frequency', y='elevation_diff', hue='condition', data=main_df)
plt.title('Elevation Difference vs. Frequency by Condition', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.subplot(2, 1, 2)
conditions = main_df['condition'].unique()
for condition in conditions:
    subset = main_df[main_df['condition'] == condition]
    sns.regplot(x='frequency', y='elevation_diff', data=subset, scatter=False, label=f'{condition} Trend')


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

model_musical = smf.mixedlm('elevation_diff ~ frequency + value_ortho_freq_centroid + C(music_based)', main_df_filtered, groups='subject')
model_musical_fitted = model_musical.fit()
print(model_musical_fitted.summary())

#### because we have the step for the last frequency bin, i would filter it out

main_df_filtered = main_df[main_df['frequency'] < 3000]
model_cond_filtered = smf.mixedlm('elevation_diff ~ frequency * C(condition)', main_df_filtered, groups='subject')
model_cond_filtered_fitted = model_cond_filtered.fit()
print(model_cond_filtered_fitted.summary())

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
sns.pointplot(x='frequency', y='elevation_diff', hue='condition', data=main_df_filtered)
plt.title('Elevation Difference vs. Frequency by Condition', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.subplot(2, 1, 2)
conditions = main_df['condition'].unique()
for condition in conditions:
    subset = main_df_filtered[main_df_filtered['condition'] == condition]
    sns.regplot(x='frequency', y='elevation_diff', data=subset, scatter=False, label=f'{condition} Trend')


plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Elevation Difference', fontsize=12)
plt.legend(title='Condition')
plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/elevation_diff_filtered.png')
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
    # Skip centroid and rolloff for complex condition
    x_variables = ['frequency']
    if condition != 'complex':
        x_variables.extend(['value_ortho_freq_centroid', 'value_ortho_freq_rolloff'])

    for x_var in x_variables:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            group[x_var],
            group['elevation_diff']
        )

        # Store results
        slope_results.append({
            'subject': subject,
            'condition': condition,
            'x_variable': x_var,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value
        })

    # Convert results to DataFrame
slopes_df = pandas.DataFrame(slope_results)

statistical_results = {}
x_variables = ['frequency', 'value_ortho_freq_centroid', 'value_ortho_freq_rolloff']

for x_var in x_variables:
    # Filter for the specific x-variable
    var_slopes = slopes_df[slopes_df['x_variable'] == x_var]

    # Skip if no data (e.g., centroid/rolloff for complex)
    if len(var_slopes) == 0:
        continue

    # Prepare data for ANOVA
    conditions = []
    slopes = []
    for condition in var_slopes['condition'].unique():
        condition_slopes = var_slopes[var_slopes['condition'] == condition]['slope']
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
main_df_filtered = main_df_filtered[main_df_filtered['frequency'] > 300]
model_interval_filtered = smf.mixedlm('elevation_diff ~ interval * condition', main_df_filtered, groups='subject')
model_interval_filtered_fitted = model_interval_filtered.fit()
print(model_interval_filtered_fitted.summary())

r = stats.pearsonr(main_df_filtered['frequency'], main_df_filtered['interval'])
plt.scatter(main_df_filtered['elevation_diff'], main_df_filtered['interval'])

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



