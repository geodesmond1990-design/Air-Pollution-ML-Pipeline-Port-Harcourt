"""
visualization.py
────────────────
Generates all six publication-quality figures used in the Springer paper.

Figure 1 — Spearman correlation heatmap + monthly CO₂ box plots
Figure 2 — Random Forest feature importance (4-panel)
Figure 3 — Model comparison bar chart (R², 3 targets)
Figure 4 — Temporal trends with linear fit (4-panel)
Figure 5 — Observed vs predicted scatter plot (Gradient Boosting OOF)
Figure 6 — Geospatial distribution of mean pollutant concentrations
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics         import r2_score, mean_squared_error

from data_loader import POLLUTANTS, METEO, FEATURES, MONTH_ORDER

warnings.filterwarnings("ignore")

# ── Style constants ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
BLUE_PALETTE  = ["#4575b4", "#74add1", "#abd9e9", "#fee090", "#f46d43", "#d73027"]
CORR_CMAP     = LinearSegmentedColormap.from_list("corr", ["#2166ac", "#f7f7f7", "#b2182b"])
SHORT_LABELS  = ["Aug\n'23","Sep\n'23","Oct\n'23","Nov\n'23","Dec\n'23","Jan\n'24",
                 "Aug\n'24","Sep\n'24","Oct\n'24","Nov\n'24","Dec\n'24","Jan\n'25"]
POLL_LABELS   = {"CO2_": "CO₂ (ppm)", "N2O": "N₂O (ppm)",
                 "CH4": "CH₄ (ppm)", "O3": "O₃ (ppm)", "CO": "CO (ppm)"}


def _savefig(fig, path: str, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      Saved: {path}")


# ── Figure 1 ──────────────────────────────────────────────────────────────────
def fig1_correlation_and_boxplot(df: pd.DataFrame, fig_dir: str) -> None:
    """Spearman heatmap (left) + monthly CO₂ box plots (right)."""
    cols      = POLLUTANTS + METEO
    col_labels = ["CO₂","N₂O","CH₄","O₃","CO","Temp","Humidity","Wind\nSpeed","Rainfall"]
    corr      = df[cols].corr(method="spearman").values

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    # — Heatmap —
    im = axes[0].imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(col_labels)))
    axes[0].set_yticks(range(len(col_labels)))
    axes[0].set_xticklabels(col_labels, fontsize=8.5)
    axes[0].set_yticklabels(col_labels, fontsize=8.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7.5, color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=axes[0], shrink=0.82, label="Spearman r")
    axes[0].set_title("Spearman Correlation Matrix", fontsize=13, fontweight="bold", pad=10)

    # — Box plots —
    monthly_groups = [df[df["month"] == m]["CO2_"].values for m in MONTH_ORDER]
    bp = axes[1].boxplot(monthly_groups, labels=SHORT_LABELS, patch_artist=True)
    colors = ["#4575b4"] * 6 + ["#d73027"] * 6
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.72)
    axes[1].set_ylabel("CO₂ (ppm)", fontsize=11)
    axes[1].set_title("CO₂ Distribution by Month (2023–2025)", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, f"{fig_dir}/fig1_correlation_boxplot.png")


# ── Figure 2 ──────────────────────────────────────────────────────────────────
def fig2_feature_importance(df: pd.DataFrame, feat_imp: dict, fig_dir: str) -> None:
    """4-panel Random Forest feature importance bar charts."""
    targets = ["CO2_", "N2O", "CH4", "CO"]
    feat_labels = ["Latitude","Longitude","Month","Temperature","Humidity",
                   "Wind Speed","Rainfall"]
    bar_colors  = ["#1a9850","#91cf60","#d9ef8b","#fee08b","#fc8d59","#d73027","#4575b4"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor="white")
    for idx, target in enumerate(targets):
        fi     = feat_imp[target]
        sorted_fi = fi.sort_values("importance")
        bars = axes[idx].barh(
            sorted_fi["feature"].map(dict(zip(FEATURES, feat_labels))),
            sorted_fi["importance"],
            color=[bar_colors[FEATURES.index(f)] for f in sorted_fi["feature"]],
            edgecolor="white", linewidth=0.5
        )
        axes[idx].set_title(POLL_LABELS.get(target, target), fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Importance", fontsize=10)
        for bar, val in zip(bars, sorted_fi["importance"]):
            axes[idx].text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                           f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Random Forest Feature Importance (MDI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{fig_dir}/fig2_feature_importance.png")


# ── Figure 3 ──────────────────────────────────────────────────────────────────
def fig3_model_comparison(cv_results: dict, fig_dir: str) -> None:
    """Bar chart of 5-fold CV R² for three main pollutants."""
    targets    = ["CO2_", "CH4", "CO"]
    tgt_labels = ["CO₂", "CH₄", "CO"]
    model_keys = ["Linear Regression","Ridge Regression","Random Forest",
                  "Gradient Boosting","SVR"]
    model_names = ["Linear\nRegression","Ridge\nRegression","Random\nForest",
                   "Gradient\nBoosting","SVR"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    for pidx, (target, tlabel) in enumerate(zip(targets, tgt_labels)):
        r2s   = [cv_results[target][m]["R2_mean"] for m in model_keys]
        stds  = [cv_results[target][m]["R2_std"]  for m in model_keys]
        bars  = axes[pidx].bar(model_names, r2s, color=BLUE_PALETTE,
                               edgecolor="white", yerr=stds, capsize=5,
                               error_kw={"linewidth": 1.5})
        axes[pidx].set_title(f"{tlabel} Prediction", fontsize=13, fontweight="bold")
        axes[pidx].set_ylabel("R² Score (5-fold CV)", fontsize=10)
        axes[pidx].set_ylim(-0.15, 1.0)
        axes[pidx].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[pidx].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, r2s):
            if val > 0:
                axes[pidx].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                                f"{val:.3f}", ha="center", va="bottom",
                                fontsize=8, fontweight="bold")

    fig.suptitle("Model Comparison: 5-Fold Cross-Validation R²", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{fig_dir}/fig3_model_comparison.png")


# ── Figure 4 ──────────────────────────────────────────────────────────────────
def fig4_temporal_trends(df: pd.DataFrame, fig_dir: str) -> None:
    """4-panel temporal trend plots with linear regression overlay."""
    monthly = df.groupby("month")[POLLUTANTS].mean().reindex(MONTH_ORDER)
    targets = ["CO2_", "N2O", "CH4", "CO"]
    colors  = ["#d73027", "#4575b4", "#1a9850", "#756bb1"]
    x       = np.arange(len(MONTH_ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    axes = axes.flatten()

    for idx, (t, c) in enumerate(zip(targets, colors)):
        y = monthly[t].values
        axes[idx].plot(x, y, "o-", color=c, linewidth=2.5, markersize=7)
        z = np.polyfit(x, y, 1)
        axes[idx].plot(x, np.polyval(z, x), "--", color="grey",
                       linewidth=1.5, alpha=0.8, label=f"Trend (β={z[0]:.3f}/month)")
        axes[idx].set_xticks(range(len(SHORT_LABELS)))
        axes[idx].set_xticklabels(SHORT_LABELS, fontsize=8.5)
        axes[idx].set_ylabel(POLL_LABELS.get(t, t), fontsize=11)
        axes[idx].set_title(f"Temporal Trend: {POLL_LABELS.get(t,t).split('(')[0].strip()}",
                            fontsize=12, fontweight="bold")
        axes[idx].grid(alpha=0.3)
        axes[idx].legend(fontsize=9)
        # Shade periods
        axes[idx].axvspan(-0.5, 5.5, alpha=0.05, color="blue")
        axes[idx].axvspan(5.5,  11.5, alpha=0.05, color="red")
        ymax = axes[idx].get_ylim()[1]
        axes[idx].text(2.5, ymax * 0.97, "2023",    ha="center", fontsize=9, color="blue",  alpha=0.7)
        axes[idx].text(8.5, ymax * 0.97, "2024-25", ha="center", fontsize=9, color="red",   alpha=0.7)

    fig.suptitle("Temporal Trends in Air Pollutant Concentrations (2023–2025)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{fig_dir}/fig4_temporal_trends.png")


# ── Figure 5 ──────────────────────────────────────────────────────────────────
def fig5_observed_vs_predicted(df: pd.DataFrame, fig_dir: str) -> None:
    """Observed vs. OOF-predicted scatter plots (Gradient Boosting)."""
    targets     = ["CO2_", "CH4", "CO"]
    tgt_labels  = ["CO₂ (ppm)", "CH₄ (ppm)", "CO (ppm)"]
    X_scaled    = StandardScaler().fit_transform(df[FEATURES].values)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    for idx, (target, label) in enumerate(zip(targets, tgt_labels)):
        y      = df[target].values
        y_pred = np.zeros_like(y, dtype=float)
        model  = GradientBoostingRegressor(n_estimators=100, random_state=42)
        kf     = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_scaled):
            model.fit(X_scaled[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X_scaled[test_idx])

        r2   = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())

        axes[idx].scatter(y, y_pred, alpha=0.55, s=28, color="#4575b4",
                          edgecolors="white", linewidth=0.3)
        axes[idx].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="1:1 line")
        axes[idx].set_xlabel(f"Observed {label}", fontsize=11)
        axes[idx].set_ylabel(f"Predicted {label}", fontsize=11)
        axes[idx].set_title(f"{label.split()[0]}: Gradient Boosting\nR²={r2:.4f}, RMSE={rmse:.4f}",
                            fontsize=11, fontweight="bold")
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)

    fig.suptitle("Observed vs. Predicted — Gradient Boosting (5-Fold CV)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{fig_dir}/fig5_observed_vs_predicted.png")


# ── Figure 6 ──────────────────────────────────────────────────────────────────
def fig6_spatial_distribution(df: pd.DataFrame, fig_dir: str) -> None:
    """Proportional-symbol geospatial maps for CO₂, CH₄, CO."""
    spatial = (
        df.groupby(["station", "latitude", "longitude"])
          [["CO2_", "CH4", "CO"]]
          .mean()
          .reset_index()
    )
    targets  = ["CO2_", "CH4", "CO"]
    cmaps_sp = ["RdYlGn_r", "YlOrRd", "Blues"]
    labels   = ["CO₂ (ppm)", "CH₄ (ppm)", "CO (ppm)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    for idx, (t, cm, label) in enumerate(zip(targets, cmaps_sp, labels)):
        sc = axes[idx].scatter(
            spatial["longitude"], spatial["latitude"],
            c=spatial[t], s=110, cmap=cm,
            edgecolors="grey", linewidth=0.5, alpha=0.88
        )
        plt.colorbar(sc, ax=axes[idx], shrink=0.82, label=label)
        axes[idx].set_xlabel("Longitude (°E)", fontsize=10)
        axes[idx].set_ylabel("Latitude (°N)",  fontsize=10)
        axes[idx].set_title(f"Spatial Distribution: {label.split()[0]}",
                            fontsize=12, fontweight="bold")
        axes[idx].grid(alpha=0.3)
        # Annotate top 3 hotspots
        for _, row in spatial.nlargest(3, t).iterrows():
            axes[idx].annotate(row["station"], (row["longitude"], row["latitude"]),
                               fontsize=6.5, ha="left", va="bottom",
                               xytext=(3, 3), textcoords="offset points")

    fig.suptitle("Geospatial Distribution of Air Pollutants — Port Harcourt, Nigeria",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, f"{fig_dir}/fig6_spatial_distribution.png")


# ── Master entry point ────────────────────────────────────────────────────────
def generate_all_figures(
    df: pd.DataFrame,
    cv_results: dict,
    feat_imp: dict,
    output_dir: str,
) -> None:
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n  Saving figures to {fig_dir}/")
    fig1_correlation_and_boxplot(df, fig_dir)
    fig2_feature_importance(df, feat_imp, fig_dir)
    fig3_model_comparison(cv_results, fig_dir)
    fig4_temporal_trends(df, fig_dir)
    fig5_observed_vs_predicted(df, fig_dir)
    fig6_spatial_distribution(df, fig_dir)
    print("  All 6 figures generated.")
