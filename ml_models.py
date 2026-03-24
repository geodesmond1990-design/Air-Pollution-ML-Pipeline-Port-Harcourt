"""
ml_models.py
────────────
Trains and evaluates five supervised regression algorithms for predicting
air pollutant concentrations using geospatial, temporal, and meteorological
features.

Models
------
  1. Linear Regression (baseline)
  2. Ridge Regression (L2-regularised)
  3. Random Forest
  4. Gradient Boosting  ← best performer
  5. Support Vector Regression (RBF kernel)

Evaluation
----------
  5-fold cross-validation → R², RMSE, MAE (mean ± SD across folds)
  Feature importance via MDI (Random Forest, full dataset)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from data_loader             import POLLUTANTS, FEATURES

warnings.filterwarnings("ignore")

# ── Model registry ────────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                         max_depth=3, random_state=42),
        "SVR":                SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    }

# ── Cross-validation ──────────────────────────────────────────────────────────
def cross_validate_models(
    X_scaled: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Run 5-fold CV for all models on a single target.
    Returns a dict  {model_name: {metric: value}}.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, model in get_models().items():
        r2   = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
        rmse = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kf,
                                        scoring="neg_mean_squared_error"))
        mae  = -cross_val_score(model, X_scaled, y, cv=kf,
                                scoring="neg_mean_absolute_error")
        results[name] = {
            "R2_mean":   round(r2.mean(),   4),
            "R2_std":    round(r2.std(),    4),
            "RMSE_mean": round(rmse.mean(), 4),
            "RMSE_std":  round(rmse.std(),  4),
            "MAE_mean":  round(mae.mean(),  4),
            "MAE_std":   round(mae.std(),   4),
        }

    return results


# ── Feature importance (MDI, Random Forest) ───────────────────────────────────
def compute_feature_importance(
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Fit a Random Forest on the full dataset and return MDI importances.
    """
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    imp = rf.feature_importances_
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


# ── Fold-level predictions (for scatter plots) ────────────────────────────────
def get_oof_predictions(
    X_scaled: np.ndarray,
    y: np.ndarray,
    model_name: str = "Gradient Boosting",
    n_splits: int = 5,
) -> np.ndarray:
    """
    Return out-of-fold predictions from the named model (default: GB).
    Used to generate observed-vs-predicted scatter plots.
    """
    model = get_models()[model_name]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in kf.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X_scaled[test_idx])

    return y_pred


# ── Main pipeline entry point ─────────────────────────────────────────────────
def run_ml_pipeline(df: pd.DataFrame, output_dir: str):
    """
    Run full ML pipeline:
      - Standardise features
      - Cross-validate all models for all targets
      - Compute feature importance per target
      - Save results

    Returns
    -------
    cv_results : dict   {target: {model: {metric: value}}}
    feat_imp   : dict   {target: pd.DataFrame}
    """
    os.makedirs(output_dir, exist_ok=True)

    X = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_results = {}
    feat_imp   = {}
    cv_rows    = []       # flat table for CSV export
    fi_rows    = []       # flat table for CSV export

    for target in POLLUTANTS:
        y = df[target].values
        print(f"\n  ── {target} ──")

        # Cross-validation
        target_cv = cross_validate_models(X_scaled, y)
        cv_results[target] = target_cv

        for model_name, metrics in target_cv.items():
            row = {"target": target, "model": model_name}
            row.update(metrics)
            cv_rows.append(row)
            print(f"    {model_name:<22}  R²={metrics['R2_mean']:.4f}±{metrics['R2_std']:.4f}"
                  f"  RMSE={metrics['RMSE_mean']:.4f}±{metrics['RMSE_std']:.4f}")

        # Feature importance
        fi_df = compute_feature_importance(X_scaled, y, FEATURES)
        feat_imp[target] = fi_df
        for _, fi_row in fi_df.iterrows():
            fi_rows.append({"target": target,
                            "feature":    fi_row["feature"],
                            "importance": fi_row["importance"]})

    # Save flat CSV tables
    pd.DataFrame(cv_rows).to_csv(f"{output_dir}/cv_results.csv", index=False)
    pd.DataFrame(fi_rows).to_csv(f"{output_dir}/feature_importance.csv", index=False)

    # Save OOF predictions for best model (Gradient Boosting)
    oof_rows = []
    for target in POLLUTANTS:
        y = df[target].values
        y_pred = get_oof_predictions(X_scaled, y, model_name="Gradient Boosting")
        for obs, pred in zip(y, y_pred):
            oof_rows.append({"target": target, "observed": round(obs, 4),
                             "predicted": round(pred, 4)})
    pd.DataFrame(oof_rows).to_csv(f"{output_dir}/oof_predictions_gb.csv", index=False)

    print(f"\n  Saved: cv_results.csv, feature_importance.csv, oof_predictions_gb.csv")
    return cv_results, feat_imp
