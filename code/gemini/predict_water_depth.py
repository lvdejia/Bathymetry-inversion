import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

# Set font to Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12

def dms_formatter(x, pos):
    """Format decimal degrees to degrees, minutes, seconds"""
    d = int(x)
    m = int((x - d) * 60)
    s = (x - d - m/60) * 3600
    return f'{d}°{m}\'{s:.0f}"'

def main():
    print("1. Loading Data...")
    # Load training data
    train_file = 'data.csv'
    # y, x1, x2, x3, x4
    df_train = pd.read_csv(train_file, header=None, names=['y', 'x1', 'x2', 'x3', 'x4'])

    # Load image data
    # 6 columns: Lat, Lon, Band1, Band2, Band3, Band4
    # Note: User said 0-based index: 0=Mask(unused?), 1=Lon, 2=Lat... 
    # BUT inspection showed:
    # Line 301: 17.799..., 83.398..., 0.019..., ...
    # 17.8 is likely Lat (Visakhapatnam, India), 83.4 is Lon.
    # So we assume: Col 0 = Lat, Col 1 = Lon, Col 2-5 = Bands
    image_file = 'image.csv'
    # Reading without header
    df_img = pd.read_csv(image_file, header=None)
    
    # Truncate to expected size if larger
    expected_size = 391 * 376
    if len(df_img) > expected_size:
        print(f"Truncating input from {len(df_img)} to {expected_size} rows.")
        df_img = df_img.iloc[:expected_size]
    elif len(df_img) < expected_size:
        raise ValueError(f"Input data has {len(df_img)} rows, expected at least {expected_size}.")
    
    # Check if we need to filter out 0-filled rows for prediction to save time/errors
    # The file has 147016 rows.
    # We will predict on valid pixels only.
    
    # 2. Prepare Training Data
    print("2. Training Model...")
    n = 1000
    if len(df_train) < n:
        train_subset = df_train.copy()
    else:
        train_subset = df_train.sample(n=n, random_state=42)
        
    X_train = train_subset[['x1', 'x2', 'x3', 'x4']]
    y_train = train_subset['y']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. Prediction
    print("3. Predicting...")
    # Extract features from image dataframe
    # Assuming columns 2, 3, 4, 5 correspond to x1, x2, x3, x4
    # We need to handle the mask/zeros.
    # Let's identify valid rows where data is not all zeros.
    # Or just predict everything and mask later.
    
    # The user said: "第0列为mask信息，第1列为经度信息，第2列为纬度信息" in one prompt
    # But in the latest prompt: "第1列为经度信息，第2列为纬度信息，第3-最后一列对应不同的光谱反射率数据"
    # AND "image.csv文件中真实的的遥感反射率数据，它的大小为147,016行，6列"
    
    # Let's look at the data again from view_file output:
    # 301: 17.79999352,83.39820099,0.019655455,0.031388193,0.03664802,0.011966222
    # This looks like: Lat, Lon, B1, B2, B3, B4.
    # So I will stick to: Col 0=Lat, Col 1=Lon, Col 2=x1, Col 3=x2, Col 4=x3, Col 5=x4.
    
    X_img = df_img.iloc[:, 2:6]
    X_img.columns = ['x1', 'x2', 'x3', 'x4']
    
    # Predict
    # To avoid noise from 0-filled background, we can filter.
    # But to reshape easily to 391x376 (which is 147016 pixels), we should keep the shape.
    # 391 * 376 = 147016. Perfect.
    
    y_img_pred = rf.predict(X_img)
    
    # Mask out the background (where features are 0)
    # If sum of features is 0, depth is likely invalid/background.
    mask = (X_img.sum(axis=1) == 0)
    y_img_pred[mask] = np.nan  # Set background to NaN
    
    # 4. Generate Output
    print("4. Generating Outputs...")
    
    # Reshape to (391, 376)
    # We need to know the filling order. Usually row-major.
    # Let's verify with Lat/Lon.
    lats = df_img.iloc[:, 0].values
    lons = df_img.iloc[:, 1].values
    
    depth_grid = y_img_pred.reshape(391, 376)
    lat_grid = lats.reshape(391, 376)
    lon_grid = lons.reshape(391, 376)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use extent for proper axis labeling if using imshow, 
    # but pcolormesh is better for irregular grids or just to be safe.
    # However, for a regular grid, imshow is faster.
    # Let's check if lat/lon are regular.
    # Assuming they are regular enough for imshow or we use extent.
    
    # Extent: [lon_min, lon_max, lat_min, lat_max]
    # Note: Lat decreases in the file (17.799 -> 17.799...), so image might be flipped if we just reshape.
    # Let's use pcolormesh or just scatter if we want to be safe, but scatter is slow for 147k points.
    # Actually, let's use imshow and set extent, but we need to handle orientation.
    # If row 0 is top (Lat 17.799) and row N is bottom, imshow default origin='upper' works.
    
    # Filter NaNs for min/max
    valid_depths = y_img_pred[~np.isnan(y_img_pred)]
    
    # Plot
    # We need to create a masked array for the plot
    depth_grid_masked = np.ma.masked_invalid(depth_grid)
    
    # Extent
    # We need the corners.
    # Top-Left: (Lon[0,0], Lat[0,0])
    # Bottom-Right: (Lon[-1,-1], Lat[-1,-1])
    # But let's just use the min/max from the data columns
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    
    # Since there are zeros in the data (background), we should exclude them from min/max calculation
    # The zeros are likely in the masked area.
    # Let's take min/max of valid coordinates.
    valid_mask = ~mask
    if valid_mask.any():
        lon_min = lons[valid_mask].min()
        lon_max = lons[valid_mask].max()
        lat_min = lats[valid_mask].min()
        lat_max = lats[valid_mask].max()
    
    im = ax.imshow(depth_grid_masked, cmap='Spectral', vmin=0, vmax=15, 
                   extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)')
    
    # Format axes
    ax.xaxis.set_major_formatter(FuncFormatter(dms_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(dms_formatter))
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Predicted Water Depth')
    
    plt.tight_layout()
    plt.savefig('water_depth_map.png', dpi=300)
    print("Map saved to water_depth_map.png")
    
    # Export to Excel
    # Columns: Longitude, Latitude, Depth
    # Filter out the background (0s) to keep file size smaller? 
    # User asked for "generated data output to Excel", implying the result. 
    # Usually better to export valid points.
    
    out_df = pd.DataFrame({
        'Longitude': lons,
        'Latitude': lats,
        'Depth': y_img_pred
    })
    
    # Remove rows where Depth is NaN (background)
    out_df_clean = out_df.dropna()
    
    out_file = 'predicted_water_depth.xlsx'
    out_df_clean.to_excel(out_file, index=False)
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    main()
