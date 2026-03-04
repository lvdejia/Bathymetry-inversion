# -*- coding: utf-8 -*-
"""
Harbor bathymetry inversion (Random Forest)
- Train on data.csv (y + 4 features)
- Predict on image data (mask, lon, lat, 4 features)
- Export: bathymetry map (Spectral, 0–15 m, masked pixels white) + Excel (lon, lat, depth)

Inputs expected:
1) data.csv:
   col0 = y (depth, m)
   col1..col4 = x1..x4 features

2) image_with_mask_pixel.xlsx (or your data/image.csv with same column order):
   col0 = mask (0–255)
   col1 = longitude (decimal degrees)
   col2 = latitude (decimal degrees)
   col3..end = spectral features (must be 4 columns to match the model)
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------- user parameters -------------------------
N_SAMPLE = 1000                 # random sample size for training
MASK_THRESHOLD = 128            # mask < threshold -> set to white (NaN) on map
GRID_H, GRID_W = 391, 376       # output raster size
VMIN, VMAX = 0, 15              # depth color scale (m)

DATA_CSV = r"data.csv"
IMAGE_TABLE = r"image_with_mask_pixel.xlsx"  # replace with r"data/image.csv" if you have it

OUT_MAP_PNG = r"harbor_bathymetry_map.png"
OUT_EXCEL = r"harbor_bathymetry_predictions.xlsx"
# ------------------------------------------------------------------

def deg_to_dms(deg: float, is_lon: bool = True) -> str:
    """Decimal degree -> DMS string with hemisphere suffix."""
    if np.isnan(deg):
        return ""
    hemi = "E" if is_lon else "N"
    if deg < 0:
        hemi = "W" if is_lon else "S"
        deg = abs(deg)
    d = int(deg)
    m_float = (deg - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return f"{d}°{m:02d}'{s:04.1f}\"{hemi}"

def main():
    # ---- 1) load training data ----
    data = pd.read_csv(DATA_CSV)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()

    # ---- 2) random sample for training ----
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data), size=min(N_SAMPLE, len(data)), replace=False)
    X_train = X[idx]
    y_train = y[idx]

    # ---- 3) model: Random Forest (robust for nonlinear relations) ----
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )
    model.fit(X_train, y_train)

    # Evaluate on full dataset prediction (as requested)
    y_pred_all = model.predict(X)
    rmse = math.sqrt(mean_squared_error(y, y_pred_all))
    r2 = r2_score(y, y_pred_all)
    print(f"[Training-data check] RMSE={rmse:.4f} m, R2={r2:.4f} (predicted on full data.csv)")

    # ---- 4) load image table ----
    if IMAGE_TABLE.lower().endswith(".csv"):
        img_df = pd.read_csv(IMAGE_TABLE)
    else:
        img_df = pd.read_excel(IMAGE_TABLE)

    mask = img_df.iloc[:, 0].astype(float).to_numpy()
    lon = img_df.iloc[:, 1].astype(float).to_numpy()
    lat = img_df.iloc[:, 2].astype(float).to_numpy()
    X_img = img_df.iloc[:, 3:].astype(float).to_numpy()

    # Sanity checks
    if X_img.shape[1] != X_train.shape[1]:
        raise ValueError(f"Feature mismatch: image has {X_img.shape[1]} features, model expects {X_train.shape[1]}.")
    if len(img_df) != GRID_H * GRID_W:
        raise ValueError(f"Cannot reshape: rows={len(img_df)} != {GRID_H}*{GRID_W}")

    # ---- 5) predict depth & apply mask ----
    pred = model.predict(X_img)
    valid = mask >= MASK_THRESHOLD
    pred_masked = pred.copy()
    pred_masked[~valid] = np.nan

    depth_grid = pred_masked.reshape(GRID_H, GRID_W)

    # plotting extent from valid pixels (avoid zeros from masked area)
    lon_valid = lon[valid]
    lat_valid = lat[valid]
    lon_min, lon_max = float(np.nanmin(lon_valid)), float(np.nanmax(lon_valid))
    lat_min, lat_max = float(np.nanmin(lat_valid)), float(np.nanmax(lat_valid))

    # ---- 6) map plot ----
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    cmap = plt.cm.get_cmap("Spectral").copy()
    cmap.set_bad(color="white")  # masked pixels appear white

    im = ax.imshow(
        depth_grid,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap,
        vmin=VMIN, vmax=VMAX,
        interpolation="nearest"
    )

    # DMS ticks
    xticks = np.linspace(lon_min, lon_max, 5)
    yticks = np.linspace(lat_min, lat_max, 5)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: deg_to_dms(x, is_lon=True)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: deg_to_dms(y, is_lon=False)))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (m)")
    cbar.set_ticks([0, 3, 6, 9, 12, 15])

    fig.tight_layout()
    fig.savefig(OUT_MAP_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {OUT_MAP_PNG}")

    # ---- 7) export excel (lon, lat, depth) for valid pixels only ----
    out_df = pd.DataFrame({
        "Longitude": lon[valid],
        "Latitude": lat[valid],
        "Depth": pred[valid]
    })
    out_df.to_excel(OUT_EXCEL, index=False)
    print(f"[Saved] {OUT_EXCEL}  rows={len(out_df)}")

if __name__ == "__main__":
    main()
