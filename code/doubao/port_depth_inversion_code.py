
"""
港口水深反演完整代码
功能：
1. 数据加载与预处理
2. 随机森林回归模型训练（最优模型选择）
3. 遥感数据水深反演
4. 水深图绘制（含度分秒坐标）
5. 结果保存（Excel + 图像）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 全局设置
# ==============================
# 中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 图像尺寸配置
IMAGE_ROWS = 391
IMAGE_COLS = 376
MASK_THRESHOLD = 128  # mask<128为无效区域（白色）
DEPTH_RANGE = [0, 15]  # 水深显示范围（m）
TRAIN_SAMPLE_SIZE = 1000  # 训练集样本数量
RANDOM_STATE = 42  # 随机种子

# ==============================
# 工具函数
# ==============================
def decimal_to_dms(decimal):
    """
    将十进制经纬度转换为度分秒（°′″）格式
    参数：
        decimal: 十进制经纬度值
    返回：
        str: 度分秒格式字符串
    """
    if pd.isna(decimal) or decimal == 0:
        return "0°00'0.0""
    
    decimal_abs = abs(decimal)
    degrees = int(np.floor(decimal_abs))
    minutes = int(np.floor((decimal_abs - degrees) * 60))
    seconds = ((decimal_abs - degrees) * 60 - minutes) * 60
    
    return f"{degrees}°{minutes:02d}'{seconds:.1f}""

def load_depth_data(file_path):
    """
    加载水深特征数据
    参数：
        file_path: CSV文件路径
    返回：
        df: 预处理后的DataFrame（y=水深，x1-x4=特征）
    """
    df = pd.read_csv(file_path)
    df.columns = ['y', 'x1', 'x2', 'x3', 'x4']  # 标准化列名
    print(f"✅ 水深数据加载完成：{df.shape[0]}条样本，{df.shape[1]-1}个特征")
    print(f"   水深范围：{df['y'].min():.3f} - {df['y'].max():.3f} m")
    return df

def load_remote_data(file_path):
    """
    加载遥感反射率数据
    参数：
        file_path: Excel文件路径
    返回：
        df: 预处理后的DataFrame（mask+经纬度+光谱波段）
    """
    df = pd.read_excel(file_path)
    df.columns = ['mask', 'longitude', 'latitude', 'band1', 'band2', 'band3', 'band4']
    
    # 验证数据尺寸
    assert df.shape[0] == IMAGE_ROWS * IMAGE_COLS,         f"遥感数据尺寸错误！应为{IMAGE_ROWS*IMAGE_COLS}行，实际{df.shape[0]}行"
    
    print(f"✅ 遥感数据加载完成：{df.shape[0]}个像素点")
    print(f"   地理范围：经度 {df['longitude'].min():.6f} - {df['longitude'].max():.6f}")
    print(f"             纬度 {df['latitude'].min():.6f} - {df['latitude'].max():.6f}")
    return df

# ==============================
# 模型训练模块
# ==============================
def train_depth_model(depth_df):
    """
    训练随机森林回归模型（最优模型选择）
    参数：
        depth_df: 水深数据DataFrame
    返回：
        model: 训练好的随机森林模型
        scaler: 特征标准化器
    """
    print("\n📊 开始模型训练...")
    
    # 1. 数据分割（随机抽取训练集）
    train_df = depth_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)
    X_train = train_df[['x1', 'x2', 'x3', 'x4']]
    y_train = train_df['y']
    
    # 2. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 3. 训练随机森林模型（参数优化后）
    model = RandomForestRegressor(
        n_estimators=100,        # 决策树数量
        max_depth=10,            # 树最大深度
        min_samples_split=5,     # 分裂所需最小样本数
        min_samples_leaf=2,      # 叶节点最小样本数
        random_state=RANDOM_STATE,
        n_jobs=-1                # 并行计算
    )
    model.fit(X_train_scaled, y_train)
    
    # 4. 模型评估（使用全量数据）
    X_all_scaled = scaler.transform(depth_df[['x1', 'x2', 'x3', 'x4']])
    y_pred = model.predict(X_all_scaled)
    
    rmse = np.sqrt(mean_squared_error(depth_df['y'], y_pred))
    r2 = r2_score(depth_df['y'], y_pred)
    mae = mean_absolute_error(depth_df['y'], y_pred)
    
    print("✅ 模型训练完成！评估指标：")
    print(f"   RMSE（均方根误差）：{rmse:.4f} m")
    print(f"   R²（决定系数）：{r2:.4f}")
    print(f"   MAE（平均绝对误差）：{mae:.4f} m")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': ['x1', 'x2', 'x3', 'x4'],
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("\n🔍 特征重要性排序：")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return model, scaler

# ==============================
# 水深反演模块
# ==============================
def invert_depth(remote_df, model, scaler, output_excel_path):
    """
    遥感数据水深反演
    参数：
        remote_df: 遥感数据DataFrame
        model: 训练好的模型
        scaler: 特征标准化器
        output_excel_path: 结果输出Excel路径
    返回：
        result_df: 反演结果（经度、纬度、水深）
    """
    print("\n🌊 开始水深反演...")
    
    # 1. 准备特征数据（使用4个光谱波段）
    X_remote = remote_df[['band1', 'band2', 'band3', 'band4']].values
    X_remote_scaled = scaler.transform(X_remote)
    
    # 2. 水深预测
    depth_pred = model.predict(X_remote_scaled)
    depth_pred = np.clip(depth_pred, DEPTH_RANGE[0], DEPTH_RANGE[1])  # 限制水深范围
    
    # 3. 处理无效区域（mask<128设为NaN）
    depth_pred[remote_df['mask'] < MASK_THRESHOLD] = np.nan
    
    # 4. 保存结果
    result_df = pd.DataFrame({
        'longitude': remote_df['longitude'],
        'latitude': remote_df['latitude'],
        'depth': depth_pred
    })
    result_df.to_excel(output_excel_path, index=False)
    
    # 统计信息
    valid_count = np.sum(~np.isnan(depth_pred))
    invalid_count = np.sum(np.isnan(depth_pred))
    
    print(f"✅ 水深反演完成！")
    print(f"   有效水深区域：{valid_count:,}个像素")
    print(f"   无效区域（白色）：{invalid_count:,}个像素")
    print(f"   水深范围：{np.nanmin(depth_pred):.3f} - {np.nanmax(depth_pred):.3f} m")
    print(f"   结果已保存至：{output_excel_path}")
    
    return result_df

# ==============================
# 可视化模块
# ==============================
def plot_depth_map(result_df, remote_df, output_image_path):
    """
    绘制港口水深图（含度分秒坐标）
    参数：
        result_df: 反演结果DataFrame
        remote_df: 遥感数据DataFrame
        output_image_path: 图像输出路径
    """
    print("\n🎨 开始绘制水深图...")
    
    # 1. 数据重塑
    depth_data = result_df['depth'].values.reshape(IMAGE_ROWS, IMAGE_COLS)
    mask_data = remote_df['mask'].values.reshape(IMAGE_ROWS, IMAGE_COLS)
    lon_data = result_df['longitude'].values.reshape(IMAGE_ROWS, IMAGE_COLS)
    lat_data = result_df['latitude'].values.reshape(IMAGE_ROWS, IMAGE_COLS)
    
    # 2. 处理无效区域（白色）
    depth_data[mask_data < MASK_THRESHOLD] = np.nan
    
    # 3. 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(18, 15))
    
    # 4. 定义颜色映射（Spectral）
    cmap = plt.cm.Spectral
    new_cmap = LinearSegmentedColormap.from_list("custom_Spectral", cmap(np.arange(cmap.N)))
    new_cmap.set_bad(color='white')  # NaN值设为白色
    
    # 5. 绘制水深图像
    im = ax.imshow(
        depth_data,
        cmap=new_cmap,
        vmin=DEPTH_RANGE[0],
        vmax=DEPTH_RANGE[1],
        aspect='auto',
        extent=[0, IMAGE_COLS-1, 0, IMAGE_ROWS-1]
    )
    
    # 6. 设置经纬度刻度（度分秒格式）
    # 计算刻度位置（5个均匀分布点）
    lon_ticks_pos = np.linspace(0, IMAGE_COLS-1, 5, dtype=int)
    lat_ticks_pos = np.linspace(0, IMAGE_ROWS-1, 5, dtype=int)
    
    # 获取中心位置经纬度
    center_row = IMAGE_ROWS // 2
    center_col = IMAGE_COLS // 2
    lon_ticks_values = [lon_data[center_row, pos] for pos in lon_ticks_pos]
    lat_ticks_values = [lat_data[pos, center_col] for pos in lat_ticks_pos]
    
    # 转换为度分秒标签
    lon_ticks_labels = [decimal_to_dms(val) for val in lon_ticks_values]
    lat_ticks_labels = [decimal_to_dms(val) for val in lat_ticks_values]
    
    # 设置坐标轴
    ax.set_xticks(lon_ticks_pos)
    ax.set_xticklabels(lon_ticks_labels, fontsize=12, fontweight='bold', rotation=45)
    ax.set_yticks(lat_ticks_pos)
    ax.set_yticklabels(lat_ticks_labels, fontsize=12, fontweight='bold')
    
    # 7. 设置图形标签
    ax.set_xlabel('经度（度分秒）', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_ylabel('纬度（度分秒）', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_title(
        f'港口水深反演图\n（水深范围：{DEPTH_RANGE[0]}-{DEPTH_RANGE[1]}m，白色为无效区域）',
        fontsize=22, fontweight='bold', pad=30
    )
    
    # 8. 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label('水深（单位：米）', fontsize=14, fontweight='bold', rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(DEPTH_RANGE[0], DEPTH_RANGE[1], 7))
    cbar.set_ticklabels([f'{x:.0f} m' for x in np.linspace(DEPTH_RANGE[0], DEPTH_RANGE[1], 7)])
    
    # 9. 美化图形
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='gray')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # 10. 保存图像
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 水深图绘制完成！已保存至：{output_image_path}")

# ==============================
# 主函数（完整流程）
# ==============================
def main(
    depth_data_path="data.csv",
    remote_data_path="image_with_mask_pixel.xlsx",
    output_excel_path="port_depth_results.xlsx",
    output_image_path="port_depth_map.png"
):
    """
    港口水深反演完整流程
    参数：
        depth_data_path: 水深数据路径
        remote_data_path: 遥感数据路径
        output_excel_path: 结果Excel路径
        output_image_path: 水深图路径
    """
    print("="*60)
    print("          港口水深反演系统          ")
    print("="*60)
    
    # 1. 数据加载
    depth_df = load_depth_data(depth_data_path)
    remote_df = load_remote_data(remote_data_path)
    
    # 2. 模型训练
    model, scaler = train_depth_model(depth_df)
    
    # 3. 水深反演
    result_df = invert_depth(remote_df, model, scaler, output_excel_path)
    
    # 4. 可视化
    plot_depth_map(result_df, remote_df, output_image_path)
    
    print("\n" + "="*60)
    print("          所有任务完成！          ")
    print("="*60)
    print(f"📁 输出文件：")
    print(f"   1. {output_excel_path} - 水深反演结果（经度、纬度、水深）")
    print(f"   2. {output_image_path} - 港口水深图（Spectral色标）")

# ==============================
# 程序入口
# ==============================
if __name__ == "__main__":
    # 运行完整流程
    main()
