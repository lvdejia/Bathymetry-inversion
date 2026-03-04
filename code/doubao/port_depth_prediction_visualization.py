"""
水深预测结果可视化代码
功能：
1. 加载水深数据并分割训练集/预测集
2. 训练随机森林回归模型
3. 生成密度散点图（含拟合直线、性能指标、颜色映射）
4. 符合学术图表规范（Times New Roman字体、轴标签设置等）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

# ==============================
# 1. 全局设置（符合需求规范）
# ==============================
# 字体设置：Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 基础字体大小
plt.rcParams['axes.labelsize'] = 14  # 轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小

# 图表配置
FIG_SIZE = (10, 8)  # 图表尺寸
AXIS_RANGE = [0, 15]  # 横纵坐标范围
POINTS_ALPHA = 0.6  # 散点透明度
LINE_WIDTH = 2  # 拟合直线宽度
TEXT_FONT_SIZE = 14  # 性能指标文本大小

# 数据配置
TRAIN_SAMPLE_SIZE = 1000  # 训练集样本数
RANDOM_STATE = 42  # 随机种子（可复现）
DATA_PATH = '/mnt/data.csv'  # 数据路径
SAVE_PATH = '/mnt/depth_prediction_scatter.png'  # 图像保存路径


# ==============================
# 2. 数据加载与预处理
# ==============================
def load_and_preprocess_data(file_path):
    """加载并预处理水深数据"""
    # 读取数据，第一列为y（实际水深），后四列为x1-x4（特征）
    df = pd.read_csv(file_path)
    df.columns = ['y', 'x1', 'x2', 'x3', 'x4']

    # 数据分割：随机抽取训练集，全量为预测集
    train_df = df.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)
    predict_df = df  # 全量数据作为预测集

    # 特征与目标值分离
    X_train = train_df[['x1', 'x2', 'x3', 'x4']]
    y_train = train_df['y']
    X_predict = predict_df[['x1', 'x2', 'x3', 'x4']]
    y_true = predict_df['y']

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    print(f"数据加载完成：")
    print(f"  - 训练集：{len(train_df)}条样本")
    print(f"  - 预测集：{len(predict_df)}条样本")
    print(f"  - 实际水深范围：{y_true.min():.3f} - {y_true.max():.3f} m")

    return X_train_scaled, y_train, X_predict_scaled, y_true


# ==============================
# 3. 模型训练与预测
# ==============================
def train_and_predict(X_train, y_train, X_predict):
    """训练随机森林模型并进行预测"""
    # 训练随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_predict)

    # 计算性能指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n模型性能：")
    print(f"  - RMSE：{rmse:.4f} m")
    print(f"  - R²：{r2:.4f}")

    return y_pred, rmse, r2


# ==============================
# 4. 生成密度散点图
# ==============================
def plot_density_scatter(y_true, y_pred, rmse, r2, save_path):
    """
    绘制密度散点图
    参数：
        y_true: 实际水深值
        y_pred: 预测水深值
        rmse: 均方根误差
        r2: 决定系数
        save_path: 图像保存路径
    """
    # 创建画布
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 计算点的密度
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)

    # 按密度排序（确保高密度点在上方）
    idx = z.argsort()
    y_true_sorted, y_pred_sorted, z_sorted = y_true[idx], y_pred[idx], z[idx]

    # 绘制密度散点图
    scatter = ax.scatter(
        y_true_sorted,
        y_pred_sorted,
        c=z_sorted,
        cmap='viridis',  # 颜色映射
        alpha=POINTS_ALPHA,
        s=30,  # 点大小
        edgecolors='none'  # 无边框
    )

    # 添加拟合直线（y=x完美拟合线）
    ax.plot(
        AXIS_RANGE,
        AXIS_RANGE,
        'r-',
        linewidth=LINE_WIDTH,
        label='Fitting Line (y=x)'
    )

    # 设置坐标范围
    ax.set_xlim(AXIS_RANGE)
    ax.set_ylim(AXIS_RANGE)

    # 设置轴标签
    ax.set_xlabel('In-situ (m)', fontweight='bold')
    ax.set_ylabel('Predicted', fontweight='bold')

    # 添加网格（辅助读数）
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 添加性能指标文本（左上角）
    text_str = f'RMSE = {rmse:.4f} m\nR² = {r2:.4f}'
    ax.text(
        0.05, 0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=TEXT_FONT_SIZE,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # 添加右侧颜色映射条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Density', fontweight='bold', rotation=270, labelpad=20)

    # 调整布局
    plt.tight_layout()

    # 保存图像（高分辨率）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n密度散点图已保存至：{save_path}")


# ==============================
# 5. 主函数（完整流程）
# ==============================
if __name__ == "__main__":
    print("=" * 50)
    print("开始水深模型拟合与可视化流程")
    print("=" * 50)

    # 1. 数据加载与预处理
    X_train, y_train, X_predict, y_true = load_and_preprocess_data(DATA_PATH)

    # 2. 模型训练与预测
    y_pred, rmse, r2 = train_and_predict(X_train, y_train, X_predict)

    # 3. 生成密度散点图
    plot_density_scatter(y_true, y_pred, rmse, r2, SAVE_PATH)

    print("=" * 50)
    print("流程完成！")
    print("=" * 50)