import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

# Set general font settings
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14

def main():
    # 1. Load Data
    file_path = 'd:/project/antigravity/waterdeep/data.csv'
    # Assuming no header or if there is a header, pandas will handle it.
    # Let's read first without header to inspect, but standard pd.read_csv usually works if there's a header.
    # Given the description, let's just use read_csv. If there's no header, it will use the first row as header.
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Assuming first column is y, and the rest are features
    y_col = df.columns[0]
    X_cols = df.columns[1:]
    
    # 2. Split Data
    n_samples = 1000
    if len(df) < n_samples:
        print(f"Warning: Dataset size ({len(df)}) is smaller than the requested sample size ({n_samples}). Using all data for training.")
        n_samples = len(df)
        
    train_df = df.sample(n=n_samples, random_state=42)
    
    X_train = train_df[X_cols]
    y_train = train_df[y_col]
    
    X_all = df[X_cols]
    y_all = df[y_col]

    # 3. Model Selection and Training (Random Forest)
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Prediction
    print("Predicting on entire dataset...")
    y_pred = model.predict(X_all)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_all, y_pred))
    r2 = r2_score(y_all, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # 5. Density calculation for scatter plot
    print("Calculating density...")
    # Calculate the point density
    xy = np.vstack([y_all, y_pred])
    z = gaussian_kde(xy)(xy)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_all_sorted, y_pred_sorted, z_sorted = y_all.iloc[idx], y_pred[idx], z[idx]

    # 6. Plotting
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(y_all_sorted, y_pred_sorted, c=z_sorted, s=20, cmap='jet', edgecolors='none', alpha=0.8)

    # Set axes limits
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    # 1:1 line
    ax.plot([0, 15], [0, 15], color='black', linestyle='--', linewidth=1.5, label='1:1 Line')

    # Fitting line
    # reshape for sklearn
    y_all_arr = y_all.values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(y_all_arr, y_pred)
    y_fit = lr.predict(y_all_arr)
    ax.plot(y_all.values, y_fit, color='red', linestyle='-', linewidth=1.5, label='Fit Line')

    # Text box for metrics
    textstr = f'$RMSE = {rmse:.4f}$\n$R^2 = {r2:.4f}$'
    # placing the text box in upper left
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, fontname='Times New Roman')

    # Labels
    ax.set_xlabel('In-situ(m)')
    ax.set_ylabel('Predicted')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Density', fontsize=16, fontname='Times New Roman')
    # The user asked for "不需要标题" (no title) so we leave ax.set_title out.

    plt.tight_layout()
    plt.savefig('d:/project/antigravity/waterdeep/density_scatter_plot.png', dpi=300)
    print("Plot saved to density_scatter_plot.png")
    
if __name__ == "__main__":
    main()
