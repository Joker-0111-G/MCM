import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
data = pd.read_excel('2022-2024 aqi (1).xlsx', sheet_name='Sheet2')

date_column = '日期'
value_column = 'AQI'

# 将日期列转换为日期类型并提取年月信息
data[date_column] = pd.to_datetime(data[date_column], format='%Y.%m.%d')
data['年份'] = data[date_column].dt.year
data['月份'] = data[date_column].dt.month
data['年月'] = data.apply(lambda x: f"{x['年份']}-{x['月份']:02d}", axis=1)

# 按时间顺序排序并分组（确保年月顺序正确）
sorted_months = data['年月'].unique()
sorted_months = sorted(sorted_months, key=lambda x: pd.to_datetime(x, format='%Y-%m'))  # 按时间顺序排序

# 过滤有效月份数据（保留有数据的月份）
monthly_data = {month: data[data['年月'] == month][value_column].dropna().tolist()
                for month in sorted_months if not data[data['年月'] == month][value_column].isna().all()}

valid_months = list(monthly_data.keys())

# 创建分组结构（按年份分组，便于后续标签布局）
year_groups = data['年份'].unique()
grouped_months = {year: [f"{year}-{month:02d}" for month in range(1, 13)
                         if f"{year}-{month:02d}" in valid_months]
                  for year in year_groups}

# 绘制箱线图（调整画布尺寸适应竖排标签）
plt.figure(figsize=(15, 8))  # 加宽画布适应更多月份

# 生成按时间顺序排列的数据列表
boxplot_data = [monthly_data[month] for month in valid_months]

# 绘制箱线图
boxplot = plt.boxplot(boxplot_data, labels=valid_months, widths=0.6)

# 定义渐变色（从白色到钢蓝色的渐变）
cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#FFFFFF', '#4682B4'])

# 为每个箱体添加渐变填充（优化填充逻辑）
for i, patch in enumerate(boxplot['boxes']):
    # 获取箱体顶点坐标
    vertices = patch.get_path().vertices
    x = vertices[:, 0]
    y = vertices[:, 1]

    # 创建渐变填充区域
    gradient = np.linspace(0, 1, 100)
    for j, g in enumerate(gradient):
        y1 = y[0] + j * (y[2] - y[0]) / len(gradient)
        y2 = y[0] + (j + 1) * (y[2] - y[0]) / len(gradient)
        plt.fill_between([x[0], x[1]], y1, y2, color=cmap(g), alpha=0.8, edgecolor='none')

# 设置标签竖排显示
plt.xticks(rotation=45, fontsize=12)  # 标签旋转90度，字体缩小避免重叠
plt.title('各年月AQI分布箱线图', fontsize=14)
plt.xlabel('年-月', fontsize=12)
plt.ylabel('AQI值', fontsize=12)

# 调整布局防止标签截断
plt.tight_layout(pad=3)

plt.savefig("AQI_Time_Distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# 输出统计信息（保持原有逻辑）
for month in valid_months:
    column_data = monthly_data[month]
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = [x for x in column_data if x < lower_bound or x > upper_bound]

    print(f"月份 {month} 的统计信息:")
    print(f"第一四分位数 (Q1): {Q1:.2f}")
    print(f"第三四分位数 (Q3): {Q3:.2f}")
    print(f"四分位距 (IQR): {IQR:.2f}")
    print(f"异常值数量: {len(outliers)}")
    print(f"异常值: {outliers}\n")