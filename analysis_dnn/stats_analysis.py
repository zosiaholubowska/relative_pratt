# %% IMPORT LIBRARIES
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.formula.api as smf


# %% SET PATHS
DIR = os.getcwd()
ACOUSTIC_DIR = os.path.join(os.getcwd(), "..", "acoustic_analysis")
DATA_DIR = os.path.join(DIR, "regression_output")
TRAINING_TONES_DIR = os.path.join(DIR, "..", "stimuli", "naturalsounds165_tones")
OUTPUT_DIR = os.path.join(DIR, "output_stats_analysis")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



# %% LOAD DATA
files = [f for f in os.listdir(DATA_DIR) if f.startswith("regression_predictions_")]

dfs = []
for file in files:
    df = pandas.read_csv(os.path.join(DATA_DIR, file))
    parts = file.removeprefix("regression_predictions_").removesuffix(".csv").split("_")
    df["tones"] = parts[-1] == "tones"
    if df["tones"].iloc[0]:
        parts = parts[:-1]
    df["stimuli"] = parts[0]
    df["model"] = "_".join(parts[1:])

    dfs.append(df)

data = pandas.concat(dfs)

centroids = pandas.read_csv(
    os.path.join(ACOUSTIC_DIR, "spectral_centroid_stimuli.csv")
).set_index("sound")[["centroid", "centroid_midi"]]
data["centroid"] = data["filename"].map(centroids["centroid"])

data['elev_error'] = data['pred_elev'] - data['true_elev']
data['elev_bin'] = data['true_elev'].round().astype(int)
data['log2c'] = numpy.log2(data['centroid'])


# %% 1. Compare models with different training - NATURAL SOUNDS

natural_sounds = data[data["stimuli"] == "naturalsounds"]
natural_sounds = natural_sounds.reset_index(drop=True)
natural_sounds.index.is_unique
natural_sounds['log2c_c'] = natural_sounds['log2c'] - natural_sounds['log2c'].mean()

unique_models = natural_sounds["model"].unique()
print(f"Unique models: {unique_models}")

# Fit the model

formula = "elev_error ~ 0 + model + model:log2c_c + model:C(elev_bin)"

md = smf.mixedlm(
    formula, natural_sounds,
    groups="filename",
    re_formula="1",
    vc_formula={"sound_model": "0 + C(model)"},
)
# lbfgs stops short on this model and leaves a Hessian that is not positive definite.
fit = md.fit(reml=True, method="powell")
print(fit.summary())
print(f"Converged: {fit.converged}")


# %% 1b. Centroid (frequency-elevation) effect: per-model slopes and direct comparisons
from scipy import stats
from statsmodels.stats.multitest import multipletests

fe_names = list(fit.fe_params.index)
fe_cov = fit.cov_params().loc[fe_names, fe_names].values
slope_names = {m: f"model[{m}]:log2c_c" for m in unique_models}


def linear_contrast(weights, res=fit):
    names = list(res.fe_params.index)
    cov = res.cov_params().loc[names, names].values
    c = numpy.zeros(len(names))
    for name, w in weights.items():
        c[names.index(name)] = w
    est = c @ res.fe_params.values
    se = numpy.sqrt(c @ cov @ c)
    return est, se


# Centroid range across sounds, in octaves
sound_log2c = natural_sounds.groupby("filename")["log2c"].first()
oct_5_95 = sound_log2c.quantile(0.95) - sound_log2c.quantile(0.05)
oct_sd = sound_log2c.std()
print(f"\nCentroid across {len(sound_log2c)} sounds: SD = {oct_sd:.2f} oct, "
      f"5th-95th percentile range = {oct_5_95:.2f} oct")

print("\nPer-model centroid slope (deg per octave)")
rows = []
for m, name in slope_names.items():
    est, se = linear_contrast({name: 1})
    z = est / se
    rows.append({
        "model": m,
        "slope": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "z": z,
        "p_two_sided": 2 * stats.norm.sf(abs(z)),
        "p_one_sided_positive": stats.norm.sf(z),
        "deg_per_sd": est * oct_sd,
        "deg_over_5_95_range": est * oct_5_95,
    })
slopes_table = pandas.DataFrame(rows).set_index("model")
print(slopes_table.round(4).to_string())

print("\nPairwise differences in centroid slope (deg per octave), Bonferroni-corrected")
pairs = [("prob_bias", "basic"), ("prob_bias", "flipped"), ("basic", "flipped")]
rows = []
for a, b in pairs:
    est, se = linear_contrast({slope_names[a]: 1, slope_names[b]: -1})
    z = est / se
    rows.append({
        "contrast": f"{a} - {b}",
        "diff": est,
        "se": se,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "z": z,
        "p": 2 * stats.norm.sf(abs(z)),
    })
contrast_table = pandas.DataFrame(rows).set_index("contrast")
contrast_table["p_adj"] = multipletests(contrast_table["p"], method="bonferroni")[1]
print(contrast_table.round(4).to_string())

# prob_bias vs. the average of the two unbiased models
est, se = linear_contrast({slope_names["prob_bias"]: 1,
                           slope_names["basic"]: -0.5,
                           slope_names["flipped"]: -0.5})
z = est / se
print(f"\nprob_bias - mean(basic, flipped): {est:.3f} deg/oct, SE = {se:.3f}, "
      f"95% CI [{est - 1.96 * se:.3f}, {est + 1.96 * se:.3f}], z = {z:.2f}, "
      f"p = {2 * stats.norm.sf(abs(z)):.2e}")

# Omnibus Wald test: all three slopes equal (2 df)
wald_matrix = numpy.zeros((2, len(fe_names)))
wald_matrix[0, fe_names.index(slope_names["prob_bias"])] = 1
wald_matrix[0, fe_names.index(slope_names["basic"])] = -1
wald_matrix[1, fe_names.index(slope_names["prob_bias"])] = 1
wald_matrix[1, fe_names.index(slope_names["flipped"])] = -1
diff = wald_matrix @ fit.fe_params.values
chi2 = diff @ numpy.linalg.solve(wald_matrix @ fe_cov @ wald_matrix.T, diff)
print(f"Omnibus test of equal slopes: chi2(2) = {chi2:.2f}, p = {stats.chi2.sf(chi2, 2):.2e}")


# %% 1c. Central tendency: elevation error per bin, gain and attractor elevation
bins = sorted(natural_sounds["elev_bin"].unique())
print(f"\nElevation bins: {bins} (reference = {bins[0]})")

print("\nModel-estimated mean error and predicted elevation per bin (deg, at mean centroid)")
rows = []
for m in unique_models:
    for b in bins:
        weights = {f"model[{m}]": 1}
        if b != bins[0]:
            weights[f"model[{m}]:C(elev_bin)[T.{b}]"] = 1
        est, se = linear_contrast(weights)
        rows.append({
            "model": m,
            "true_elev": b,
            "error": est,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "pred_elev": b + est,
        })
bin_table = pandas.DataFrame(rows).set_index(["model", "true_elev"])
print(bin_table.round(2).to_string())

# Linear summary: error = a + b * true_elev, so gain = 1 + b and errors cross zero at -a / b
gain_formula = "elev_error ~ 0 + model + model:true_elev + model:log2c_c"
gain_md = smf.mixedlm(
    gain_formula, natural_sounds,
    groups="filename",
    re_formula="1",
    vc_formula={"sound_model": "0 + C(model)"},
)
gain_fit = gain_md.fit(reml=True, method="powell")
print(f"\nLinear gain model converged: {gain_fit.converged}")

gain_names = list(gain_fit.fe_params.index)
gain_cov = gain_fit.cov_params().loc[gain_names, gain_names]
print("\nPer-model elevation gain (1 = veridical, 0 = constant output) and attractor elevation")
rows = []
for m in unique_models:
    a_name, b_name = f"model[{m}]", f"model[{m}]:true_elev"
    a, b = gain_fit.fe_params[a_name], gain_fit.fe_params[b_name]
    b_se = numpy.sqrt(gain_cov.loc[b_name, b_name])
    grad = numpy.array([-1 / b, a / b ** 2])
    sub_cov = gain_cov.loc[[a_name, b_name], [a_name, b_name]].values
    cross_se = numpy.sqrt(grad @ sub_cov @ grad)
    cross = -a / b
    rows.append({
        "model": m,
        "gain": 1 + b,
        "gain_ci_low": 1 + b - 1.96 * b_se,
        "gain_ci_high": 1 + b + 1.96 * b_se,
        "error_at_0": a,
        "attractor_elev": cross,
        "attractor_ci_low": cross - 1.96 * cross_se,
        "attractor_ci_high": cross + 1.96 * cross_se,
    })
gain_table = pandas.DataFrame(rows).set_index("model")
print(gain_table.round(3).to_string())

print("\nPairwise differences in gain, Bonferroni-corrected")
rows = []
for a, b in pairs:
    est, se = linear_contrast({f"model[{a}]:true_elev": 1, f"model[{b}]:true_elev": -1},
                              res=gain_fit)
    z = est / se
    rows.append({
        "contrast": f"{a} - {b}",
        "diff": est,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "z": z,
        "p": 2 * stats.norm.sf(abs(z)),
    })
gain_contrasts = pandas.DataFrame(rows).set_index("contrast")
gain_contrasts["p_adj"] = multipletests(gain_contrasts["p"], method="bonferroni")[1]
print(gain_contrasts.round(4).to_string())


# %% 1d. Centroid effect on untrained sounds: test set (ESC-50 environmental sounds + TinySOL instruments)
test_data = data[(data["stimuli"] == "test") & data["tones"]].reset_index(drop=True)
test_data["source"] = test_data["filename"].str.split("_").str[1]
test_data["log2c_c"] = test_data["log2c"] - test_data["log2c"].mean()
test_models = ["prob_bias", "basic"]
test_sources = ["esc50", "tinysol"]

test_formula = ("elev_error ~ 0 + model + model:source + model:source:log2c_c"
                " + model:C(elev_bin)")
test_md = smf.mixedlm(
    test_formula, test_data,
    groups="filename",
    re_formula="1",
    vc_formula={"sound_model": "0 + C(model)"},
)
test_fit = test_md.fit(reml=True, method="powell")


def test_slope(m, s):
    return f"model[{m}]:source[{s}]:log2c_c"


n_sounds = test_data.groupby("source")["filename"].nunique()
test_log2c = test_data.groupby("filename")[["log2c", "source"]].first()
test_ranges = {s: g["log2c"].quantile(0.95) - g["log2c"].quantile(0.05)
               for s, g in test_log2c.groupby("source")}
test_ranges["pooled"] = test_log2c["log2c"].quantile(0.95) - test_log2c["log2c"].quantile(0.05)
print(f"\n===== TEST SET (basic_tones vs. prob_bias_tones) =====")
print(f"Converged: {test_fit.converged}; sounds ESC-50 = {n_sounds['esc50']}, "
      f"TinySOL = {n_sounds['tinysol']}")
print("Centroid 5th-95th range (oct): "
      + ", ".join(f"{k} = {v:.2f}" for k, v in test_ranges.items()))

print("\nCentroid slope (deg per octave) by model and source, Bonferroni-corrected")
rows = []
for m in test_models:
    for s in test_sources + ["pooled"]:
        if s == "pooled":
            weights = {test_slope(m, src): 1 / len(test_sources) for src in test_sources}
        else:
            weights = {test_slope(m, s): 1}
        est, se = linear_contrast(weights, res=test_fit)
        z = est / se
        rows.append({
            "model": m,
            "source": s,
            "slope": est,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "z": z,
            "p": 2 * stats.norm.sf(abs(z)),
            "deg_over_5_95_range": est * test_ranges[s],
        })
test_slope_table = pandas.DataFrame(rows).set_index(["model", "source"])
test_slope_table["p_adj"] = multipletests(test_slope_table["p"], method="bonferroni")[1]
print(test_slope_table.round(4).to_string())

print("\nContrasts in centroid slope (deg per octave), Bonferroni-corrected")
test_contrasts = {
    "prob_bias - basic, esc50": {test_slope("prob_bias", "esc50"): 1,
                                 test_slope("basic", "esc50"): -1},
    "prob_bias - basic, tinysol": {test_slope("prob_bias", "tinysol"): 1,
                                   test_slope("basic", "tinysol"): -1},
    "prob_bias - basic, pooled": {test_slope("prob_bias", "esc50"): 0.5,
                                  test_slope("prob_bias", "tinysol"): 0.5,
                                  test_slope("basic", "esc50"): -0.5,
                                  test_slope("basic", "tinysol"): -0.5},
    "(prob_bias - basic): esc50 - tinysol": {test_slope("prob_bias", "esc50"): 1,
                                             test_slope("basic", "esc50"): -1,
                                             test_slope("prob_bias", "tinysol"): -1,
                                             test_slope("basic", "tinysol"): 1},
}
rows = []
for label, weights in test_contrasts.items():
    est, se = linear_contrast(weights, res=test_fit)
    z = est / se
    rows.append({
        "contrast": label,
        "diff": est,
        "ci_low": est - 1.96 * se,
        "ci_high": est + 1.96 * se,
        "z": z,
        "p": 2 * stats.norm.sf(abs(z)),
    })
test_contrast_table = pandas.DataFrame(rows).set_index("contrast")
test_contrast_table["p_adj"] = multipletests(test_contrast_table["p"], method="bonferroni")[1]
print(test_contrast_table.round(4).to_string())


# %% 2. Model generalizes the training: viola and flute tones
training_tones = set(os.listdir(TRAINING_TONES_DIR))
gen_formula = ("elev_error ~ 0 + model + model:training + model:training:log2c_c"
               " + model:C(elev_bin)")


def gen_slope(m, t):
    return f"model[{m}]:training[{t}]:log2c_c"


gen_fits, gen_slope_tables, gen_contrast_tables = {}, {}, {}
for instrument in ["viola", "flute"]:
    tones_data = data[(data["stimuli"] == instrument) & data["tones"]
                      & data["model"].isin(["basic", "prob_bias"])].reset_index(drop=True)
    tones_data["training"] = numpy.where(tones_data["filename"].isin(training_tones),
                                         "trained", "heldout")
    tones_data["log2c_c"] = tones_data["log2c"] - tones_data["log2c"].mean()

    gen_md = smf.mixedlm(
        gen_formula, tones_data,
        groups="filename",
        re_formula="1",
        vc_formula={"sound_model": "0 + C(model)"},
    )
    gen_fit = gen_md.fit(reml=True, method="powell")
    gen_fits[instrument] = gen_fit

    n_sounds = tones_data.groupby("training")["filename"].nunique()
    inst_log2c = tones_data.groupby("filename")["log2c"].first()
    inst_range = inst_log2c.quantile(0.95) - inst_log2c.quantile(0.05)
    print(f"\n===== {instrument.upper()} (basic_tones vs. prob_bias_tones) =====")
    print(f"Converged: {gen_fit.converged}; sounds trained = {n_sounds['trained']}, "
          f"held out = {n_sounds['heldout']}; centroid 5th-95th range = {inst_range:.2f} oct")

    print("\nCentroid slope (deg per octave) by model and training status, Bonferroni-corrected")
    rows = []
    for m in ["prob_bias", "basic"]:
        for t in ["trained", "heldout"]:
            est, se = linear_contrast({gen_slope(m, t): 1}, res=gen_fit)
            z = est / se
            rows.append({
                "model": m,
                "training": t,
                "slope": est,
                "ci_low": est - 1.96 * se,
                "ci_high": est + 1.96 * se,
                "z": z,
                "p": 2 * stats.norm.sf(abs(z)),
                "deg_over_5_95_range": est * inst_range,
            })
    gen_slope_tables[instrument] = pandas.DataFrame(rows).set_index(["model", "training"])
    gen_slope_tables[instrument]["p_adj"] = multipletests(
        gen_slope_tables[instrument]["p"], method="bonferroni")[1]
    print(gen_slope_tables[instrument].round(4).to_string())

    print("\nContrasts in centroid slope (deg per octave), Bonferroni-corrected")
    gen_contrasts = {
        "prob_bias - basic, trained": {gen_slope("prob_bias", "trained"): 1,
                                       gen_slope("basic", "trained"): -1},
        "prob_bias - basic, heldout": {gen_slope("prob_bias", "heldout"): 1,
                                       gen_slope("basic", "heldout"): -1},
        "prob_bias: trained - heldout": {gen_slope("prob_bias", "trained"): 1,
                                         gen_slope("prob_bias", "heldout"): -1},
        "(prob_bias - basic): trained - heldout": {gen_slope("prob_bias", "trained"): 1,
                                                   gen_slope("basic", "trained"): -1,
                                                   gen_slope("prob_bias", "heldout"): -1,
                                                   gen_slope("basic", "heldout"): 1},
    }
    rows = []
    for label, weights in gen_contrasts.items():
        est, se = linear_contrast(weights, res=gen_fit)
        z = est / se
        rows.append({
            "contrast": label,
            "diff": est,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "z": z,
            "p": 2 * stats.norm.sf(abs(z)),
        })
    contrasts = pandas.DataFrame(rows).set_index("contrast")
    contrasts["p_adj"] = multipletests(contrasts["p"], method="bonferroni")[1]
    gen_contrast_tables[instrument] = contrasts
    print(contrasts.round(4).to_string())

# %% 3. Comparison with human behaviour: viola and flute tones
# Humans were tested at -25 to 25 deg elevation and the models at 0 to 60 deg, so errors cannot be
# compared point to point; the comparison is on the centroid slope (deg per octave).
HUMAN_DIR = os.path.join(DIR, "..", "Results")
human_models = ["basic", "prob_bias"]

human = pandas.read_csv(os.path.join(HUMAN_DIR, "data.csv"))
human = human[~human["subject"].str.contains(r"pilot|00", regex=True, na=False)]
human = human.dropna(subset=["elevation_diff"]).copy()
human["filename"] = ("stim_" + human["midi_note"].astype(int).astype(str)
                     + "_" + human["condition"] + ".wav")
human["log2c"] = numpy.log2(human["filename"].map(centroids["centroid"]))

human_slope_tables = {}
for instrument in ["flute", "viola"]:
    hum = human[human["condition"] == instrument].reset_index(drop=True)
    stim_files = hum["filename"].unique()
    mod = data[(data["stimuli"] == instrument) & data["tones"]
               & data["model"].isin(human_models)
               & data["filename"].isin(stim_files)].reset_index(drop=True)
    stim_log2c = hum.groupby("filename")["log2c"].first()
    hum["log2c_c"] = hum["log2c"] - stim_log2c.mean()
    mod["log2c_c"] = mod["log2c"] - stim_log2c.mean()
    stim_range = stim_log2c.quantile(0.95) - stim_log2c.quantile(0.05)

    print(f"\n===== {instrument.upper()}: humans vs. basic_tones vs. prob_bias_tones =====")
    print(f"Subjects = {hum['subject'].nunique()}, stimuli = {len(stim_files)}, "
          f"centroid 5th-95th range = {stim_range:.2f} oct")

    # Centroid slope for humans: random intercept and slope per subject
    hum_fit = smf.mixedlm(
        "elevation_diff ~ log2c_c + C(elevation_ls)", hum,
        groups="subject", re_formula="1 + log2c_c",
    ).fit(reml=True, method="powell")
    h_est, h_se = hum_fit.fe_params["log2c_c"], hum_fit.bse_fe["log2c_c"]
    print(f"Converged: human = {hum_fit.converged}")

    # Model slopes on the same 12 stimuli
    mod_fit = smf.mixedlm(
        "elev_error ~ 0 + model + model:log2c_c + model:C(elev_bin)", mod,
        groups="filename", re_formula="1",
        vc_formula={"sound_model": "0 + C(model)"},
    ).fit(reml=True, method="powell")
    print(f"Converged: models = {mod_fit.converged}")
    mod_names = [f"model[{m}]:log2c_c" for m in human_models]
    m_est = mod_fit.fe_params[mod_names].values
    m_cov = mod_fit.cov_params().loc[mod_names, mod_names].values

    rows = [{"agent": "human", "slope": h_est,
             "ci_low": h_est - 1.96 * h_se, "ci_high": h_est + 1.96 * h_se,
             "deg_over_5_95_range": h_est * stim_range}]
    for i, m in enumerate(human_models):
        est, se = m_est[i], numpy.sqrt(m_cov[i, i])
        diff, diff_se = est - h_est, numpy.sqrt(m_cov[i, i] + h_se ** 2)
        rows.append({
            "agent": m,
            "slope": est,
            "ci_low": est - 1.96 * se,
            "ci_high": est + 1.96 * se,
            "deg_over_5_95_range": est * stim_range,
            "ratio_to_human": est / h_est,
            "diff_from_human": diff,
            "diff_ci_low": diff - 1.96 * diff_se,
            "diff_ci_high": diff + 1.96 * diff_se,
            "p_diff": 2 * stats.norm.sf(abs(diff / diff_se)),
        })
    slope_cmp = pandas.DataFrame(rows).set_index("agent")
    slope_cmp.loc[human_models, "p_diff_adj"] = multipletests(
        slope_cmp.loc[human_models, "p_diff"], method="bonferroni")[1]
    human_slope_tables[instrument] = slope_cmp
    print("\nCentroid slope (deg per octave) and difference from humans, Bonferroni-corrected")
    print(slope_cmp.round(4).to_string())

MODEL_LABELS = {"prob_bias": "World Bias", "human": "Human", "basic": "Basic"}
MODEL_COLORS = {"Basic": "#1a80bb", "Human": "black", "World Bias": "#a00000"}
fig, axes = plt.subplots(2, 1, figsize=(6, 3.5), sharex=True)
for ax, (instrument, slope_cmp) in zip(axes, human_slope_tables.items()):
    human_row = slope_cmp.loc["human"]
    ax.axvspan(human_row["ci_low"], human_row["ci_high"], color="grey", alpha=0.2)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    labels = []
    for y, (agent, row) in enumerate(slope_cmp.iterrows()):
        label = MODEL_LABELS[agent]
        ax.errorbar(row["slope"], y,
                    xerr=[[row["slope"] - row["ci_low"]], [row["ci_high"] - row["slope"]]],
                    fmt="o", capsize=3, color=MODEL_COLORS[label])
        labels.append(label)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.6, -0.6)
    if ax == axes[1]:
        ax.set_xlabel("Centroid slope (deg elevation error per octave)")
    ax.set_title(instrument.capitalize(), fontsize=14, pad=3)
fig.tight_layout(h_pad=0.5)
fig.savefig(os.path.join(OUTPUT_DIR, "human_vs_models_centroid_slopes.png"), dpi=200)
plt.show()

# %% 4. f0 vs. centroid analysis

# %% 5. test with other stimulus set
