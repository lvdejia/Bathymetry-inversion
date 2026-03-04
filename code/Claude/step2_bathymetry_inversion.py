#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步骤2: 港口水深反演
- 加载训练好的模型
- 读取遥感影像数据
- 使用mask识别水陆边界
- 进行水深反演
- 生成水深分布图
- 导出Excel数据
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
import openpyxl
from openpyxl.styles import Font, Alignment
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 设置字体
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']

# 文件路径
MODEL_PATH = 'bathymetry_model.joblib'  # 模型文件路径
DATA_PATH = 'image.csv'  # 输入遥感数据路径
MASK_PATH = 'mask_yd.png'  # mask文件路径（黑色=陆地，白色=水域）
OUTPUT_IMAGE = 'bathymetry_map.png'  # 输出水深图路径（带坐标）
OUTPUT_IMAGE_EXACT = 'bathymetry_391x376.png'  # 输出精确尺寸图（376×391像素）
OUTPUT_EXCEL = 'bathymetry_results.xlsx'  # 输出Excel路径

# 数据尺寸
ROWS = 391
COLS = 376

# ==================== 函数定义 ====================

def decimal_to_dms(decimal_degree):
    """
    将十进制度转换为度分秒格式的字符串
    
    参数:
        decimal_degree: 十进制度数
    
    返回:
        格式化的度分秒字符串
    """
    degrees = int(decimal_degree)
    minutes_decimal = (decimal_degree - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60
    return f"{degrees}°{minutes}'{seconds:.0f}\""


def dms_formatter_lon(x, pos):
    """经度格式化函数（用于matplotlib）"""
    return decimal_to_dms(x)


def dms_formatter_lat(x, pos):
    """纬度格式化函数（用于matplotlib）"""
    return decimal_to_dms(x)


def load_model(model_path):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型文件路径
    
    返回:
        加载的模型对象
    """
    print("=" * 70)
    print("步骤1: 加载模型...")
    model = joblib.load(model_path)
    print("✓ 模型加载成功")
    return model


def load_mask(mask_path, rows, cols):
    """
    加载mask文件
    
    参数:
        mask_path: mask图像路径
        rows: 期望的行数
        cols: 期望的列数
    
    返回:
        valid_mask: 布尔数组，True表示有效水域，False表示陆地
    """
    try:
        print("\n步骤2: 加载mask文件...")
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
        
        # 如果是RGB图像，转换为灰度
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # 检查尺寸
        if mask_array.shape != (rows, cols):
            print(f"  警告: Mask尺寸 {mask_array.shape} 与数据尺寸 ({rows}, {cols}) 不匹配")
            mask_img_resized = Image.fromarray(mask_array).resize((cols, rows), Image.Resampling.NEAREST)
            mask_array = np.array(mask_img_resized)
            print(f"  已调整Mask尺寸为: {mask_array.shape}")
        
        # 创建布尔mask：黑色(0)为无效，白色(255)为有效
        valid_mask = mask_array > 128
        print(f"✓ Mask加载成功")
        print(f"  有效像元数: {valid_mask.sum()}")
        print(f"  无效像元数: {(~valid_mask).sum()}")
        
        return valid_mask
    except FileNotFoundError:
        print(f"  警告: 未找到mask文件 {mask_path}，将使用所有非零像元")
        return None
    except Exception as e:
        print(f"  警告: 加载mask文件出错: {e}，将使用所有非零像元")
        return None


def load_and_process_data(data_path):
    """
    加载并处理遥感数据
    
    参数:
        data_path: 数据文件路径
    
    返回:
        lon: 经度数组
        lat: 纬度数组
        spectral_data: 光谱数据数组
    """
    print("\n步骤3: 读取遥感数据...")
    data = pd.read_csv(data_path, header=None)
    print(f"✓ 数据形状: {data.shape}")
    
    # 提取经纬度和光谱数据
    lon = data[0].values
    lat = data[1].values
    spectral_data = data.iloc[:, 2:6].values  # 第3-6列是光谱反射率数据
    
    print(f"✓ 经度范围: {lon[lon>0].min():.6f}° - {lon.max():.6f}°")
    print(f"✓ 纬度范围: {lat[lat>0].min():.6f}° - {lat.max():.6f}°")
    print(f"✓ 光谱数据形状: {spectral_data.shape}")
    
    return lon, lat, spectral_data


def predict_bathymetry(model, spectral_data):
    """
    使用模型预测水深
    
    参数:
        model: 训练好的模型
        spectral_data: 光谱数据
    
    返回:
        bathymetry: 预测的水深数组
    """
    print("\n步骤4: 进行水深反演预测...")
    
    # 只对非零数据进行预测
    valid_mask = (spectral_data != 0).any(axis=1)
    bathymetry = np.zeros(len(spectral_data))
    
    # 对有效数据进行预测
    if valid_mask.sum() > 0:
        bathymetry[valid_mask] = model.predict(spectral_data[valid_mask])
        print(f"✓ 预测完成，有效数据点: {valid_mask.sum()}")
        print(f"✓ 预测水深范围: {bathymetry[valid_mask].min():.3f} - {bathymetry[valid_mask].max():.3f} m")
    else:
        print("警告: 没有有效的光谱数据")
    
    # 将负值设为0（水深不能为负）
    bathymetry[bathymetry < 0] = 0
    
    return bathymetry


def reshape_data(lon, lat, bathymetry, rows, cols):
    """
    将一维数据重塑为二维网格
    
    参数:
        lon: 经度数组
        lat: 纬度数组
        bathymetry: 水深数组
        rows: 行数
        cols: 列数
    
    返回:
        lon_grid: 经度网格
        lat_grid: 纬度网格
        bathymetry_grid: 水深网格
    """
    print("\n步骤5: 重塑数据为二维网格...")
    
    # 如果数据长度是rows*cols+1，去掉第一行
    expected_length = rows * cols
    if len(lon) == expected_length + 1:
        lon = lon[1:]
        lat = lat[1:]
        bathymetry = bathymetry[1:]
    
    # 重塑为二维网格
    lon_grid = lon.reshape(rows, cols)
    lat_grid = lat.reshape(rows, cols)
    bathymetry_grid = bathymetry.reshape(rows, cols)
    
    print(f"✓ 数据重塑完成: {bathymetry_grid.shape}")
    print(f"✓ 有效水深点数: {(bathymetry_grid > 0).sum()}")
    
    return lon_grid, lat_grid, bathymetry_grid


def plot_bathymetry_map(lon_grid, lat_grid, bathymetry_grid, valid_mask, output_path):
    """
    绘制水深图
    
    参数:
        lon_grid: 经度网格
        lat_grid: 纬度网格
        bathymetry_grid: 水深网格
        valid_mask: 有效像元的布尔掩码（可以为None）
        output_path: 输出图像路径
    """
    print("\n步骤6: 绘制水深图...")
    
    # 如果没有提供mask，使用经纬度非零作为有效像元
    if valid_mask is None:
        valid_mask = (lon_grid != 0) & (lat_grid != 0)
        print("  使用经纬度非零判断有效像元")
    
    # 获取有效像元的经纬度范围
    valid_lon = lon_grid[valid_mask]
    valid_lat = lat_grid[valid_mask]
    
    print(f"  有效像元数: {valid_mask.sum()}")
    print(f"  经度范围: {valid_lon.min():.6f}° - {valid_lon.max():.6f}°")
    print(f"  纬度范围: {valid_lat.min():.6f}° - {valid_lat.max():.6f}°")
    
    # 将无效像元设为NaN以便在图中显示为空白
    bathymetry_plot = bathymetry_grid.copy()
    bathymetry_plot[~valid_mask] = np.nan
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 计算数据的纵横比（列/行）
    rows, cols = bathymetry_grid.shape
    data_aspect = cols / rows  # 376/391 ≈ 0.962
    
    # 绘制水深图（使用Spectral颜色映射，红色=浅水，蓝色=深水，范围0-15m）
    im = ax.imshow(bathymetry_plot, cmap='Spectral',
                   extent=[valid_lon.min(), valid_lon.max(), 
                          valid_lat.min(), valid_lat.max()],
                   origin='upper', interpolation='nearest',
                   vmin=0, vmax=15,  # 设置颜色映射范围为0-15m
                   aspect=data_aspect)  # 使用数据纵横比，避免拉伸变形
    
    # 设置坐标轴标签
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
    
    # 设置刻度格式为度分秒
    ax.xaxis.set_major_formatter(FuncFormatter(dms_formatter_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(dms_formatter_lat))
    
    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Water Depth (m)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 水深图已保存: {output_path}")
    
    # 输出统计信息
    valid_depths = bathymetry_grid[valid_mask & (bathymetry_grid > 0)]
    if len(valid_depths) > 0:
        print(f"\n水深统计信息:")
        print(f"  有效水深点数: {len(valid_depths)}")
        print(f"  最小水深: {valid_depths.min():.3f} m")
        print(f"  最大水深: {valid_depths.max():.3f} m")
        print(f"  平均水深: {valid_depths.mean():.3f} m")
        print(f"  中位水深: {np.median(valid_depths):.3f} m")


def plot_bathymetry_exact_size(bathymetry_grid, valid_mask, output_path):
    """
    生成精确尺寸的水深图（376×391像素）
    
    参数:
        bathymetry_grid: 水深网格（391×376）
        valid_mask: 有效像元的布尔掩码
        output_path: 输出图像路径
    """
    print("\n步骤6b: 生成精确尺寸图像（376×391像素）...")
    
    # 准备绘图数据
    bathymetry_plot = bathymetry_grid.copy()
    if valid_mask is not None:
        bathymetry_plot[~valid_mask] = np.nan
    
    rows, cols = bathymetry_grid.shape  # 391, 376
    
    # 创建精确尺寸的图像
    dpi = 100
    fig = plt.figure(figsize=(cols/dpi, rows/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    ax.imshow(bathymetry_plot, cmap='Spectral', aspect='auto',
              origin='upper', interpolation='nearest', vmin=0, vmax=15)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 验证尺寸
    from PIL import Image
    img_check = Image.open(output_path)
    print(f"✓ 精确尺寸图已保存: {output_path}")
    print(f"  图像尺寸: {img_check.size[0]}×{img_check.size[1]} (宽×高)")
    print(f"  数据尺寸: {rows}行×{cols}列")


def export_to_excel(lon, lat, bathymetry, output_path):
    """
    将结果导出为Excel文件
    
    参数:
        lon: 经度数组
        lat: 纬度数组
        bathymetry: 水深数组
        output_path: 输出Excel路径
    """
    print("\n步骤7: 导出Excel文件...")
    
    # 创建DataFrame
    result_df = pd.DataFrame({
        '经度 (Longitude)': lon,
        '纬度 (Latitude)': lat,
        '水深 (Water Depth, m)': bathymetry
    })
    
    # 保存为Excel
    result_df.to_excel(output_path, index=False, engine='openpyxl')
    
    # 美化Excel
    wb = openpyxl.load_workbook(output_path)
    ws = wb.active
    
    # 设置表头样式
    header_font = Font(bold=True, size=12)
    for cell in ws[1]:
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # 自动调整列宽
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(output_path)
    
    print(f"✓ Excel文件已保存: {output_path}")
    print(f"✓ 数据行数: {len(result_df)}")
    print(f"✓ 包含 {(result_df['水深 (Water Depth, m)'] > 0).sum()} 个有效水深值")


# ==================== 主程序 ====================

def main():
    """
    主程序入口
    """
    print("=" * 70)
    print("               步骤2: 港口水深反演")
    print("=" * 70)
    
    try:
        # 1. 加载模型
        model = load_model(MODEL_PATH)
        
        # 2. 加载mask（如果存在）
        valid_mask = load_mask(MASK_PATH, ROWS, COLS)
        
        # 3. 加载并处理数据
        lon, lat, spectral_data = load_and_process_data(DATA_PATH)
        
        # 4. 预测水深
        bathymetry = predict_bathymetry(model, spectral_data)
        
        # 5. 重塑数据
        lon_grid, lat_grid, bathymetry_grid = reshape_data(
            lon, lat, bathymetry, ROWS, COLS
        )
        
        # 6. 绘制水深图
        plot_bathymetry_map(lon_grid, lat_grid, bathymetry_grid, valid_mask, OUTPUT_IMAGE)
        
        # 6b. 生成精确尺寸的图像（376×391像素）
        plot_bathymetry_exact_size(bathymetry_grid, valid_mask, OUTPUT_IMAGE_EXACT)
        
        # 7. 导出Excel
        # 注意：这里使用一维数据导出
        lon_flat = lon if len(lon) == ROWS * COLS else lon[1:]
        lat_flat = lat if len(lat) == ROWS * COLS else lat[1:]
        bathymetry_flat = bathymetry if len(bathymetry) == ROWS * COLS else bathymetry[1:]
        export_to_excel(lon_flat, lat_flat, bathymetry_flat, OUTPUT_EXCEL)
        
        print("\n" + "=" * 70)
        print("               水深反演完成！")
        print("=" * 70)
        print(f"\n输出文件:")
        print(f"  1. {OUTPUT_IMAGE} - 水深分布图（带坐标）")
        print(f"  2. {OUTPUT_IMAGE_EXACT} - 精确尺寸图（376×391像素）")
        print(f"  3. {OUTPUT_EXCEL} - 水深数据表")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
