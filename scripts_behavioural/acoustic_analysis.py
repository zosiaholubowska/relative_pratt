import os
from sklearn.linear_model import LinearRegression
import slab
import pandas
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['svg.fonttype'] = 'none'

slab.set_default_samplerate(44828)
# ====== DIRECTORIES

DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
PLOT_DIR = f'{DIR}/plots'
TONE_DIR = f'{DIR}/stimuli/tones'
STIM_DIR = f'{DIR}/stimuli'

conditions = ['flute', 'viola', 'harmoniccomplex', 'viola_complex']
features = ['centroid', 'rolloff']
# ====== Compute spectral features:
# spectral centroid, spectral roll-off

'''
acoustic_features_df = get_acoustic_features(conditions, features, TONE_DIR)

# get the midi values from the stimulus name
acoustic_features_df['midi'] = acoustic_features_df['stimulus'].apply(lambda x: int(re.search(r"stim_(\d+)_", x).group(1)))

acoustic_features_df['frequency'] = acoustic_features_df['midi'].apply(notetofreq)

acoustic_features_df.to_csv(f'{RESULTS_DIR}/acoustic_features.csv')
'''
acoustic_features = pandas.read_csv(f'{RESULTS_DIR}/acoustic_features.csv', index_col=0)
acoustic_features = acoustic_features.pivot_table(
    index=['stimulus', 'condition', 'midi', 'frequency'],
    columns='feature',
    values='value',
    aggfunc='first'
).reset_index()
palette = ['#e22c1f', '#E5A09C', '#2e33a6', '#9FA0CC']
g = sns.lmplot(data=acoustic_features, x="frequency", y="centroid", hue="condition", palette=palette, ci=None)
g.set_axis_labels("Frequency (Hz)", "Spectral Centroid (Hz)")
g.savefig(f'{PLOT_DIR}/centroid_stimuli.svg', dpi=300)
plt.close()

g = sns.barplot(data=acoustic_features, x="condition", y="centroid", hue="condition", palette=palette)
plt.ylim((0, 4500))
plt.yticks((0,1000, 2000, 3000, 4000))
plt.savefig(f'{PLOT_DIR}/centroid_stimuli_bar.svg', dpi=300)
plt.close()

from scipy.stats import f_oneway

groups = [group["centroid"].values for name, group in acoustic_features.groupby("condition")]
f_stat, p_val = f_oneway(*groups)

print(f"ANOVA F = {f_stat:.3f}, p = {p_val:.4f}")

import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=acoustic_features["centroid"],
                          groups=acoustic_features["condition"],
                          alpha=0.05)
print(tukey)

g = sns.lmplot(data=acoustic_features, x="frequency", y="rolloff", hue="condition", ci=None)
g.set_axis_labels("Frequency (Hz)", "Spectral Roll-Off (Hz)")
g.savefig(f'{PLOT_DIR}/rolloff_stimuli.png', dpi=300)
plt.close()

# ---- m o d e l s   c e n t r o i d ----

acoustic_features['condition_code'] = pandas.Categorical(acoustic_features['condition']).codes
dict(enumerate(pandas.Categorical(acoustic_features['condition']).categories))
# {0: 'flute', 1: 'harmoniccomplex', 2: 'viola', 3: 'viola_complex'}

X = sm.add_constant(acoustic_features[['centroid']])
y = acoustic_features['condition_code']

model_centroid = sm.MNLogit(y, X).fit()
print(model_centroid.summary())

# ---- m o d e l s   r o l l - o f f ----
X = sm.add_constant(acoustic_features[['rolloff']])
y = acoustic_features['condition_code']

model_rolloff = sm.MNLogit(y, X).fit()
print(model_rolloff.summary())

#### Orthogonalise the acoustic features
acoustic_features_df = pandas.read_csv(f'{RESULTS_DIR}/acoustic_features.csv', index_col=0)
result_df = acoustic_features_df.copy()
result_df['value_ortho_freq'] = None


for feature in features:
    mask = (result_df['feature'] == feature)

    if not any(mask):
        continue

    subset = result_df.loc[mask].copy()

    X = subset['frequency'].values.reshape(-1, 1)
    y = subset['value'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    orthogonalized = y - y_pred

    result_df.loc[mask, 'value_ortho_freq'] = orthogonalized


result_df.to_csv(f'{RESULTS_DIR}/acoustic_features_orthogonalised.csv')

#  plot the orthogonalised features
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

fig, axes = plt.subplots(2, len(conditions), figsize=(5 * len(conditions), 10), sharey='row')
feat = 'centroid'
for i, condition in enumerate(conditions):

    condition_data = result_df[(result_df['condition'] == condition) &
                               (result_df['feature'] == feat)]

    ax_top = axes[0, i]
    sns.scatterplot(
        x='frequency',
        y='value',
        data=condition_data,
        alpha=0.7,
        color='blue',
        ax=ax_top
    )
    x = condition_data['frequency'].values
    y = condition_data['value'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax_top.plot(x, p(x), 'r--', alpha=0.8)

    try:
        valid_data = condition_data.dropna(subset=['frequency', 'value'])
        if len(valid_data) > 1:
            corr = np.corrcoef(valid_data['frequency'], valid_data['value'])[0, 1]
            corr_text = f'r = {corr:.2f}'
        else:
            corr_text = 'insufficient data'
    except:
        corr_text = 'correlation error'

    ax_top.set_title(f'{condition}: Original {feat} \n{corr_text}')
    ax_top.set_xlabel('Frequency')
    ax_top.set_ylabel(f'{feat} (Hz)')

    ax_bottom = axes[1, i]
    sns.scatterplot(
        x='frequency',
        y='value_ortho_freq',
        data=condition_data,
        alpha=0.7,
        color='green',
        ax=ax_bottom
    )
    ax_bottom.axhline(y=0, color='r', linestyle='--', alpha=0.8)

    try:
        valid_data = condition_data.dropna(subset=['frequency', 'value_ortho_freq'])
        if len(valid_data) > 1:  # Ensure there are at least 2 points
            corr_ortho = valid_data['frequency'].corr(valid_data['value_ortho_freq'])
            corr_text = f'r = {corr_ortho:.2f}'
        else:
            corr_text = 'insufficient data'
    except:
        corr_text = 'correlation error'

    ax_bottom.set_title(f'{condition}: Orthogonalized\n{corr_text}')
    ax_bottom.set_xlabel('Frequency')
    ax_bottom.set_ylabel(f'Orthogonalized {feat}')

plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/stimuli/original_orthogonalised_{feat}.png', dpi=300)
plt.close()

### Plot features combined

feat = 'centroid'

# Filter the DataFrame for the specific feature
feat_df = result_df[result_df['feature'] == feat]

# Create the figure with subplots
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 12))

# Top subplot - Original feature
for i, condition in enumerate(conditions):
    condition_data = feat_df[feat_df['condition'] == condition]

    # Scatter plot
    sns.scatterplot(
        x='frequency',
        y='value',
        data=condition_data,
        alpha=0.7,
        label=condition,
        ax=ax_top
    )

    # Linear regression
    x = condition_data['frequency'].values
    y = condition_data['value'].values

    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax_top.plot(x, p(x), linestyle='--', alpha=0.8)

    # Correlation calculation
    try:
        valid_data = condition_data.dropna(subset=['frequency', 'value'])
        if len(valid_data) > 1:
            corr = np.corrcoef(valid_data['frequency'], valid_data['value'])[0, 1]
            print(f'{condition} Original {feat} Correlation: r = {corr:.2f}')
    except Exception:
        print(f'{condition} Original {feat} Correlation: error')

ax_top.set_title(f'Original {feat}')
ax_top.set_xlabel('Frequency')
ax_top.set_ylabel(f'{feat} (Hz)')
ax_top.legend()

# Bottom subplot - Orthogonalized feature
for i, condition in enumerate(conditions):
    condition_data = feat_df[feat_df['condition'] == condition]

    # Scatter plot
    sns.scatterplot(
        x='frequency',
        y='value_ortho_freq',
        data=condition_data,
        alpha=0.7,
        label=condition,
        ax=ax_bottom
    )

    ax_bottom.axhline(y=0, color='r', linestyle='--', alpha=0.8)

    # Correlation calculation for orthogonalized feature
    try:
        valid_data = condition_data.dropna(subset=['frequency', 'value_ortho_freq'])
        if len(valid_data) > 1:
            corr_ortho = valid_data['frequency'].corr(valid_data['value_ortho_freq'])
            print(f'{condition} Orthogonalized {feat} Correlation: r = {corr_ortho:.2f}')
    except Exception:
        print(f'{condition} Orthogonalized {feat} Correlation: error')

ax_bottom.set_title(f'Orthogonalized {feat}')
ax_bottom.set_xlabel('Frequency')
ax_bottom.set_ylabel(f'Orthogonalized {feat}')
ax_bottom.legend()

plt.tight_layout()
plt.show()
plt.savefig(f'{PLOT_DIR}/stimuli/original_orthogonalised_{feat}_combined.png', dpi=300)
plt.close()


