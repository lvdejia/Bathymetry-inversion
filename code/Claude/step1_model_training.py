#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤1: 水深预测模型训练与验证
- 读取训练数据
- 对比多个机器学习模型
- 选择最佳模型
- 生成预测vs实际的密度散点图
- 保存训练好的模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde
from matplotlib import rcParams
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 设置字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']
rcParams['mathtext.fontset'] = 'dejavuserif'

# 文件路径
TRAINING_DATA = 'data.csv'  # 训练数据文件
OUTPUT_MODEL = 'bathymetry_model.joblib'  # 输出模型文件
OUTPUT_FIGURE = 'prediction_density_scatter.png'  # 输出图像文件

# 训练参数
RANDOM_SEED = 42
TRAIN_SAMPLE_SIZE = 1000  # 从数据中随机抽取的训练样本数

# ==================== 主程序 ====================

def main():
    """主程序入口"""
    
    print("=" * 80)
    print("            步骤1: 水深预测模型训练与验证")
    print("=" * 80)
    
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    
    # ========== 1. 读取训练数据 ==========
    print("\n【1】读取训练数据...")
    print("-" * 80)
    
    data = pd.read_csv(TRAINING_DATA, header=None)
    data.columns = ['y', 'x1', 'x2', 'x3', 'x4']
    
    print(f"✓ 数据形状: {data.shape}")
    print(f"✓ 目标值范围: {data['y'].min():.3f} - {data['y'].max():.3f} m")
    
    # 查看特征相关性
    correlation = data.corr()
    print(f"\n特征与目标值的相关性:")
    for col in ['x1', 'x2', 'x3', 'x4']:
        print(f"  {col}: {correlation['y'][col]:.4f}")
    
    # ========== 2. 准备训练数据 ==========
    print(f"\n【2】准备训练数据...")
    print("-" * 80)
    
    X = data[['x1', 'x2', 'x3', 'x4']].values
    y = data['y'].values
    
    # 随机抽取训练样本
    train_indices = np.random.choice(len(X), size=TRAIN_SAMPLE_SIZE, replace=False)
    X_train = X[train_indices]
    y_train = y[train_indices]
    
    print(f"✓ 训练集大小: {len(X_train)} 样本")
    print(f"✓ 完整数据集: {len(X)} 样本（用于预测评估）")
    
    # ========== 3. 训练多个模型并对比 ==========
    print(f"\n【3】训练并对比多个模型...")
    print("-" * 80)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, 
                                               random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                                       random_state=RANDOM_SEED),
        'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                          max_iter=500, random_state=RANDOM_SEED)
    }
    
    results = []
    best_model = None
    best_score = -float('inf')
    best_name = None
    
    print(f"\n{'模型名称':<25} {'RMSE (m)':<12} {'R²':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 在完整数据集上预测
        y_pred = model.predict(X)
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"{name:<25} {rmse:<12.4f} {r2:<10.4f}")
        
        results.append({
            'name': name,
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        })
        
        # 记录最佳模型
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
            best_y_pred = y_pred
    
    print("-" * 50)
    print(f"✓ 最佳模型: {best_name}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y, best_y_pred)):.4f} m")
    print(f"  R² = {r2_score(y, best_y_pred):.4f}")
    
    # ========== 4. 保存最佳模型 ==========
    print(f"\n【4】保存最佳模型...")
    print("-" * 80)
    
    joblib.dump(best_model, OUTPUT_MODEL)
    print(f"✓ 模型已保存至: {OUTPUT_MODEL}")
    
    # ========== 5. 绘制密度散点图 ==========
    print(f"\n【5】绘制预测结果密度散点图...")
    print("-" * 80)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y, best_y_pred))
    r2 = r2_score(y, best_y_pred)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 计算点的密度
    xy = np.vstack([y, best_y_pred])
    z = gaussian_kde(xy)(xy)
    
    # 按密度排序，使高密度点显示在上层
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = y[idx], best_y_pred[idx], z[idx]
    
    # 绘制散点图
    scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=15, 
                        cmap='jet', alpha=0.6)
    
    # 添加1:1参考线
    ax.plot([0, 15], [0, 15], 'k--', linewidth=1.5, label='1:1 line')
    
    # 添加线性拟合线
    lr = LinearRegression()
    lr.fit(y.reshape(-1, 1), best_y_pred.reshape(-1, 1))
    fit_line = lr.predict(np.array([[0], [15]]))
    ax.plot([0, 15], fit_line.flatten(), 'r-', linewidth=1.5, label='Fit line')
    
    # 设置坐标轴
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_xlabel('In-situ (m)', fontsize=15)
    ax.set_ylabel('Predicted (m)', fontsize=15)
    
    # 设置刻度字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(13)
    
    # 添加RMSE和R²文本到左上角
    textstr = f'RMSE = {rmse:.3f} m\n$R^2$ = {r2:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Density', fontsize=13)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(11)
    
    # 添加图例
    legend = ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 密度散点图已保存至: {OUTPUT_FIGURE}")
    
    # ========== 6. 总结 ==========
    print("\n" + "=" * 80)
    print("                        训练完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  1. {OUTPUT_MODEL} - 训练好的模型")
    print(f"  2. {OUTPUT_FIGURE} - 预测结果密度散点图")
    print(f"\n模型性能:")
    print(f"  • 最佳模型: {best_name}")
    print(f"  • RMSE: {rmse:.3f} m")
    print(f"  • R²: {r2:.3f}")
    print(f"  • 训练样本: {TRAIN_SAMPLE_SIZE}")
    print(f"  • 评估样本: {len(y)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
