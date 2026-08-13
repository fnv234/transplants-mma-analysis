"""Generate publication-ready PA renal random forest feature importance figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

BIOMARKER_LABELS = {
    'Arginine_Citrulline_ratio': 'Arginine/Citrulline ratio',
    'LN_TNFR1': 'TNFR1 (log)',
    'LN_TNFR2': 'TNFR2 (log)',
    'LN_KIM1': 'KIM-1 (log)',
    'LN_FGF21': 'FGF-21 (log)',
    'LN_GDF15': 'GDF-15 (log)',
    'LN_NGAL': 'NGAL (log)',
    'LN_RBP4': 'RBP4 (log)',
    'LN_TGFb1': 'TGF-β1 (log)',
    'Uric_Acid': 'Uric acid',
    'FEUA': 'FEUA',
    'PropOx_60': 'PropOx (60 min)',
    'PropOx_120': 'PropOx (120 min)',
}

TARGET_LABELS = {
    'Albumin_excretion_24h': 'Albumin excretion (24 h)',
    'Protein_Creatinine_Ratio': 'Protein/creatinine ratio',
    'Creatinine': 'Serum creatinine',
    'eGFR_SCr': 'eGFR (creatinine)',
    'eGFR_CystatinC': 'eGFR (cystatin C)',
    'Cystatin_C': 'Cystatin C',
}

PAPER_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def load_data():
    data_pa_non = pd.read_csv('Pa_renal_nontransplanted.csv')
    data_pa_post = pd.read_csv('pa_renal_post-transplanted.csv')
    data_all = pd.concat([data_pa_non, data_pa_post], ignore_index=True)
    data_all = data_all.replace(r'^\s*$', np.nan, regex=True).replace('', np.nan)
    data_all = data_all.rename(columns={
        'Arg_Citrulline_Ratio': 'Arginine_Citrulline_ratio',
        'eGFR_CyC': 'eGFR_CystatinC',
        'Screatinine_mg_dL': 'Creatinine',
        'Cystatin_mg_L': 'Cystatin_C',
    })
    for col in data_all.columns:
        data_all[col] = pd.to_numeric(data_all[col], errors='ignore')
    return data_all


def fit_model(data_all):
    dependent_vars = [
        'Protein_Creatinine_Ratio', 'Albumin_excretion_24h', 'Protein_excretion_24h',
        'Creatinine', 'Cystatin_C', 'eGFR_SCr', 'eGFR_CystatinC',
    ]
    independent_vars = [
        'Arginine_Citrulline_ratio', 'LN_TNFR1', 'LN_TNFR2', 'LN_KIM1', 'LN_FGF21',
        'LN_GDF15', 'LN_NGAL', 'LN_RBP4', 'LN_TGFb1', 'Uric_Acid', 'FEUA',
        'PropOx_60', 'PropOx_120',
    ]
    available_dependent = [v for v in dependent_vars if v in data_all.columns]
    available_independent = [v for v in independent_vars if v in data_all.columns]

    model_df = data_all[available_independent + available_dependent].copy()
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    target_missing = model_df[available_dependent].isna().mean()
    kept_targets = target_missing[target_missing <= 0.60].index.tolist()
    model_df = model_df.loc[model_df[kept_targets].notna().any(axis=1)].copy()
    x = model_df[available_independent]
    y = model_df[kept_targets]

    y_imputed = pd.DataFrame(
        SimpleImputer(strategy='median').fit_transform(y),
        columns=kept_targets,
        index=y.index,
    )
    x_preprocess = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), available_independent),
    ])
    pipeline = Pipeline([
        ('x_preprocess', x_preprocess),
        ('rf_multi', MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        )),
    ])
    pipeline.fit(x, y_imputed)
    return pipeline, kept_targets, available_independent


def save_figure(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG_DIR / f'{stem}.{ext}', bbox_inches='tight', facecolor='white')
    print(f'Saved {stem}.png and {stem}.pdf')


def main():
    pipeline, kept_targets, feature_names = fit_model(load_data())
    all_importances = np.array([
        est.feature_importances_ for est in pipeline.named_steps['rf_multi'].estimators_
    ])
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': all_importances.mean(axis=0),
        'std_importance': all_importances.std(axis=0),
    }).sort_values('mean_importance', ascending=False)
    importance_by_target_df = pd.DataFrame(all_importances, index=kept_targets, columns=feature_names)

    plot_df = importance_df.copy()
    plot_df['label'] = [BIOMARKER_LABELS.get(f, f) for f in plot_df['feature']]
    plot_df = plot_df.sort_values('mean_importance', ascending=True)

    heatmap_df = importance_by_target_df.copy()
    heatmap_df.index = [TARGET_LABELS.get(t, t) for t in heatmap_df.index]
    heatmap_df.columns = [BIOMARKER_LABELS.get(f, f) for f in heatmap_df.columns]
    heatmap_df = heatmap_df.loc[:, plot_df['label']]
    colors = sns.color_palette('colorblind', n_colors=len(plot_df))

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.barh(
            plot_df['label'], plot_df['mean_importance'], xerr=plot_df['std_importance'],
            color=colors, edgecolor='black', linewidth=0.4, capsize=2,
            error_kw={'elinewidth': 0.8, 'capthick': 0.8},
        )
        ax.set_xlabel('Mean decrease in impurity (Gini importance)')
        ax.set_title('A  Mean biomarker importance across renal outcomes', loc='left', fontweight='bold')
        ax.set_xlim(0, plot_df['mean_importance'].max() * 1.15)
        fig.tight_layout()
        save_figure(fig, 'pa_renal_rf_mean_feature_importance')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        sns.heatmap(
            heatmap_df, cmap='Blues', linewidths=0.4, linecolor='white',
            cbar_kws={'label': 'Gini importance'}, ax=ax,
        )
        ax.set_title('B  Biomarker importance by renal outcome', loc='left', fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        save_figure(fig, 'pa_renal_rf_feature_importance_heatmap')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1.05, 1.25]})
        axes[0].barh(
            plot_df['label'], plot_df['mean_importance'], xerr=plot_df['std_importance'],
            color=colors, edgecolor='black', linewidth=0.4, capsize=2,
            error_kw={'elinewidth': 0.8, 'capthick': 0.8},
        )
        axes[0].set_xlabel('Mean decrease in impurity')
        axes[0].set_title('A', loc='left', fontweight='bold')
        axes[0].set_xlim(0, plot_df['mean_importance'].max() * 1.15)
        sns.heatmap(
            heatmap_df, cmap='Blues', linewidths=0.4, linecolor='white',
            cbar_kws={'label': 'Gini importance'}, ax=axes[1],
        )
        axes[1].set_title('B', loc='left', fontweight='bold')
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        fig.suptitle(
            'Random forest feature importance for PA renal cohort biomarkers (n = 50)',
            y=1.03, fontsize=11,
        )
        fig.tight_layout()
        save_figure(fig, 'pa_renal_rf_feature_importance_combined')
        plt.close(fig)

    print(importance_df.to_string(index=False))


if __name__ == '__main__':
    main()
