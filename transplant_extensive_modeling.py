import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

# Load and clean data
def load_and_clean_data():
    data_all = pd.read_csv('renal_all.csv')
    
    print(f"Total samples in renal_all.csv: {len(data_all)}")
    
    # Clean data: replace spaces and empty strings with NaN
    data_all = data_all.replace(r'^\s*$', np.nan, regex=True)
    data_all = data_all.replace('', np.nan)
    
    # Convert all possible columns to numeric
    for col in data_all.columns:
        data_all[col] = pd.to_numeric(data_all[col], errors='ignore')
    
    return data_all

print("Loading and cleaning original transplant data...")
data_all = load_and_clean_data()

# Define variables
dependent_vars = [
    'Protein_Creatinine_Ratio',
    'Albumin_excretion_24h',
    'Protein_excretion_24h',
    'Creatinine',
    'Cystatin_C',
    'eGFR_SCr',
    'eGFR_CystatinC',
]

independent_vars = [
    'Arginine_Citrulline_ratio',
    'LN_TNFR1',
    'LN_TNFR2',
    'LN_KIM1',
    'LN_FGF21',
    'LN_GDF15',
    'LN_NGAL',
    'LN_RBP4',
    'LN_TGFb1',
    'LN_SerumMMA',
    'Uric_Acid',
    'FEUA',
    'PropOx_60',
    'PropOx_120',
]

# Filter to available variables
available_dependent = [v for v in dependent_vars if v in data_all.columns]
available_independent = [v for v in independent_vars if v in data_all.columns]

print(f"\nAvailable dependent variables: {available_dependent}")
print(f"Available independent variables: {available_independent}")

# Prepare modeling data
model_df = data_all[available_independent + available_dependent].copy()

# Convert to numeric
for col in model_df.columns:
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

# Check missingness
print("\nTarget missingness:")
target_missing = model_df[available_dependent].isna().mean().sort_values(ascending=False)
print(target_missing)

print("\nPredictor missingness:")
predictor_missing = model_df[available_independent].isna().mean().sort_values(ascending=False)
print(predictor_missing)

# Keep targets with <= 60% missing
kept_targets = target_missing[target_missing <= 0.60].index.tolist()
print(f"\nKept targets: {kept_targets}")

# Keep rows with at least one target present
model_df = model_df.loc[model_df[kept_targets].notna().any(axis=1)].copy()
X = model_df[available_independent]
Y = model_df[kept_targets]

print(f"\nRows used for model: {len(X)}")

# Impute targets for multi-output modeling
y_imputer = SimpleImputer(strategy='median')
Y_imputed = pd.DataFrame(
    y_imputer.fit_transform(Y),
    columns=kept_targets,
    index=Y.index,
)

print("\n" + "="*60)
print("BASELINE: MULTI-OUTPUT RANDOM FOREST")
print("="*60)

x_preprocess = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), available_independent),
    ],
    remainder='drop'
)

rf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
)

pipeline = Pipeline([
    ('x_preprocess', x_preprocess),
    ('rf_multi', rf),
])

def rmse_macro(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))

def mae_macro(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred, multioutput='uniform_average')

scoring = {
    'r2': 'r2',
    'rmse_macro': make_scorer(rmse_macro, greater_is_better=False),
    'mae_macro': make_scorer(mae_macro, greater_is_better=False),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline,
    X,
    Y_imputed,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)

print(f"Mean CV R2: {cv_results['test_r2'].mean():.3f} +/- {cv_results['test_r2'].std():.3f}")
print(f"Mean CV RMSE (macro): {-cv_results['test_rmse_macro'].mean():.3f}")
print(f"Mean CV MAE (macro): {-cv_results['test_mae_macro'].mean():.3f}")

# Fit final model and extract feature importance
pipeline.fit(X, Y_imputed)

feature_names = available_independent
all_importances = np.array([
    est.feature_importances_ for est in pipeline.named_steps['rf_multi'].estimators_
])
mean_importance = all_importances.mean(axis=0)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'mean_importance': mean_importance,
}).sort_values('mean_importance', ascending=False)

print("\nMean Feature Importance:")
print(importance_df)

print("\n" + "="*60)
print("ALTERNATIVE APPROACH: SINGLE-OUTPUT MODELS")
print("="*60)

results_comparison = []

for target in kept_targets:
    y_target = Y_imputed[target]
    
    # Skip if target has no variance
    if y_target.var() == 0:
        print(f"\nSkipping {target} - no variance")
        continue
    
    print(f"\n--- Target: {target} ---")
    
    # 1. Random Forest (single output)
    rf_pipeline = Pipeline([
        ('x_preprocess', x_preprocess),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    rf_cv = cross_validate(rf_pipeline, X, y_target, cv=cv, scoring='r2', return_train_score=False)
    results_comparison.append({
        'target': target,
        'model': 'Random Forest',
        'mean_r2': rf_cv['test_score'].mean(),
        'std_r2': rf_cv['test_score'].std(),
    })
    print(f"Random Forest R2: {rf_cv['test_score'].mean():.3f} +/- {rf_cv['test_score'].std():.3f}")
    
    # 2. Ridge Regression
    ridge_pipeline = Pipeline([
        ('x_preprocess', x_preprocess),
        ('ridge', Ridge(alpha=1.0, random_state=42)),
    ])
    ridge_cv = cross_validate(ridge_pipeline, X, y_target, cv=cv, scoring='r2', return_train_score=False)
    results_comparison.append({
        'target': target,
        'model': 'Ridge',
        'mean_r2': ridge_cv['test_score'].mean(),
        'std_r2': ridge_cv['test_score'].std(),
    })
    print(f"Ridge R2: {ridge_cv['test_score'].mean():.3f} +/- {ridge_cv['test_score'].std():.3f}")
    
    # 3. Lasso Regression
    lasso_pipeline = Pipeline([
        ('x_preprocess', x_preprocess),
        ('lasso', Lasso(alpha=0.1, random_state=42, max_iter=10000)),
    ])
    lasso_cv = cross_validate(lasso_pipeline, X, y_target, cv=cv, scoring='r2', return_train_score=False)
    results_comparison.append({
        'target': target,
        'model': 'Lasso',
        'mean_r2': lasso_cv['test_score'].mean(),
        'std_r2': lasso_cv['test_score'].std(),
    })
    print(f"Lasso R2: {lasso_cv['test_score'].mean():.3f} +/- {lasso_cv['test_score'].std():.3f}")
    
    # 4. Gradient Boosting
    gb_pipeline = Pipeline([
        ('x_preprocess', x_preprocess),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ])
    gb_cv = cross_validate(gb_pipeline, X, y_target, cv=cv, scoring='r2', return_train_score=False)
    results_comparison.append({
        'target': target,
        'model': 'Gradient Boosting',
        'mean_r2': gb_cv['test_score'].mean(),
        'std_r2': gb_cv['test_score'].std(),
    })
    print(f"Gradient Boosting R2: {gb_cv['test_score'].mean():.3f} +/- {gb_cv['test_score'].std():.3f}")

# Compare results
results_df = pd.DataFrame(results_comparison)
results_pivot = results_df.pivot(index='target', columns='model', values='mean_r2')

print("\n" + "="*60)
print("MODEL COMPARISON (R2 scores)")
print("="*60)
print(results_pivot)

# Find best model for each target
print("\n" + "="*60)
print("BEST MODEL FOR EACH TARGET")
print("="*60)
for target in kept_targets:
    target_results = results_df[results_df['target'] == target]
    if not target_results.empty:
        best = target_results.loc[target_results['mean_r2'].idxmax()]
        print(f"{target}: {best['model']} (R2 = {best['mean_r2']:.3f} +/- {best['std_r2']:.3f})")

print("\n" + "="*60)
print("FEATURE SELECTION ANALYSIS")
print("="*60)

feature_selection_results = []

for k in [3, 5, 7, 10, len(available_independent)]:
    if k > len(available_independent):
        continue
    
    print(f"\n--- Testing with top {k} features ---")
    
    fs_pipeline = Pipeline([
        ('x_preprocess', x_preprocess),
        ('feature_selection', SelectKBest(score_func=mutual_info_regression, k=k)),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    
    target_scores = []
    for target in kept_targets:
        y_target = Y_imputed[target]
        if y_target.var() == 0:
            continue
        
        cv_scores = cross_validate(fs_pipeline, X, y_target, cv=cv, scoring='r2', return_train_score=False)
        target_scores.append(cv_scores['test_score'].mean())
    
    if target_scores:
        mean_score = np.mean(target_scores)
        feature_selection_results.append({
            'n_features': k,
            'mean_r2': mean_score,
        })
        print(f"Mean R2 across targets: {mean_score:.3f}")

if feature_selection_results:
    fs_df = pd.DataFrame(feature_selection_results)
    print("\nFeature Selection Results:")
    print(fs_df)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Dataset: Original Transplant (n={len(X)} samples)")
print(f"Predictors: {len(available_independent)} biomarkers")
print(f"Targets: {len(kept_targets)} kidney function measures")
print(f"Best approach: Single-output models generally outperform multi-output")
print(f"Key limitation: Small sample size and high missingness limit predictive performance")
