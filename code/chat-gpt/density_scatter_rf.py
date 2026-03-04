# -*- coding: utf-8 -*-
"""
Density scatter (hexbin) for In-situ vs Predicted, with RMSE & R2 and a fitted line.

Input:
- data.csv : col0 = y (in-situ depth, m), col1..col4 = x1..x4 features

Workflow:
1) Randomly sample n=1000 rows as training set
2) Train RandomForestRegressor
3) Predict the full dataset
4) Plot density scatter (hexbin) with colorbar, axes 0–15
5) Add fitted line and show RMSE/R2 at top-left
6) Save figure

Note on font:
- The script sets Times New Roman first. If your environment lacks it, it falls back to a Times-like serif.
"""
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------- user parameters -------------------------
DATA_CSV = r"data.csv"
N_SAMPLE = 1000
OUT_PNG = r"density_scatter_insitu_vs_predicted.png"

XMIN, XMAX = 0, 15
YMIN, YMAX = 0, 15
GRIDSIZE = 80
# ------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA_CSV)
    y = df.iloc[:, 0].to_numpy(dtype=float)
    X = df.iloc[:, 1:].to_numpy(dtype=float)

    # Random sample for training
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
    X_train = X[idx]
    y_train = y[idx]

    # Model (robust for nonlinear tabular regression)
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )
    model.fit(X_train, y_train)

    # Predict full dataset
    y_pred = model.predict(X)

    rmse = math.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Fit line: Predicted = a*In-situ + b
    a, b = np.polyfit(y, y_pred, 1)

    # --------- style: Times New Roman ---------
    # If Times New Roman is not installed, matplotlib will fall back.
    mpl.rcParams['font.family'] = ['Times New Roman', 'Times', 'Nimbus Roman', 'serif']
    mpl.rcParams['font.size'] = 12

    # --------- plot ---------
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=300)

    hb = ax.hexbin(
        y, y_pred,
        gridsize=GRIDSIZE,
        extent=(XMIN, XMAX, YMIN, YMAX),
        mincnt=1,
        norm=LogNorm()
    )

    # fitted line
    xline = np.array([XMIN, XMAX], dtype=float)
    ax.plot(xline, a * xline + b, linewidth=2)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("In-situ(m)")
    ax.set_ylabel("Predicted")

    # metrics text in top-left
    ax.text(
        0.03, 0.97,
        f"RMSE = {rmse:.3f} m\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top", ha="left"
    )

    # colorbar
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Point density (log count)")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {OUT_PNG}")
    print(f"RMSE={rmse:.4f} m, R2={r2:.4f}, Fit: y={a:.4f}x+{b:.4f}")

if __name__ == "__main__":
    main()
