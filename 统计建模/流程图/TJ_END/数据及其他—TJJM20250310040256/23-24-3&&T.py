import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pylab import mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取Excel数据
data = pd.read_excel('地铁1-6号线路23-24统计.xlsx', sheet_name='Sheet4')

# 假设数据中有两列：'日期' 和 '西安地铁客运量（万人次）'
# 如果列名不同，请根据实际情况调整
date_column = '日期'  # 假设日期列名为 '年份'
value_column = '三号线'  # 假设值列名为 '西安地铁客运量（万人次）'

# 将日期列转换为日期类型
data[date_column] = pd.to_datetime(data[date_column], format='%Y.%m.%d')

# 提取月份和年份
data['月份'] = data[date_column].dt.month
data['年份'] = data[date_column].dt.year
data['年月'] = data['年份'].astype(str) + "-" + data['月份'].apply(lambda x: f"{x:02d}")

# 按月份分组
monthly_data = {}
for month in data['年月'].unique():
    monthly_data[month] = pd.to_numeric(data[data['年月'] == month][value_column], errors='coerce').dropna().tolist()

# 如果某个月份没有有效数据，则跳过
valid_months = [month for month, values in monthly_data.items() if values]

# 绘制箱线图
plt.figure(figsize=(12, 8))

# 创建一个自定义的箱线图
boxplot = plt.boxplot([monthly_data[month] for month in valid_months], labels=valid_months)

# 定义渐变色
cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#FFFFFF', '#4682B4'])  # 从白色到钢蓝色的渐变

# 为每个箱体添加渐变填充
for i, (patch, month_data) in enumerate(zip(boxplot['boxes'], [monthly_data[month] for month in valid_months])):
    # 获取箱体的边界点
    box = patch.get_path().vertices
    box = box[(box[:, 0] != 0) & (box[:, 1] != 0)]  # 过滤掉无效点

    # 创建渐变填充
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    colors = cmap(gradient)

    # 创建自定义路径
    path = Path([
        (box[0, 0], box[0, 1]),
        (box[1, 0], box[1, 1]),
        (box[2, 0], box[2, 1]),
        (box[3, 0], box[3, 1]),
        (box[0, 0], box[0, 1])
    ])

    # 创建渐变填充的Patch
    patch = PathPatch(path, facecolor='none', edgecolor='black')
    plt.gca().add_patch(patch)

    # 绘制渐变填充
    for j in range(len(colors)):
        plt.fill_between(
            [box[0, 0], box[1, 0]],
            [box[0, 1] + j * (box[2, 1] - box[0, 1]) / len(colors), box[0, 1] + j * (box[2, 1] - box[0, 1]) / len(colors)],
            [box[0, 1] + (j + 1) * (box[2, 1] - box[0, 1]) / len(colors), box[0, 1] + (j + 1) * (box[2, 1] - box[0, 1]) / len(colors)],
            color=colors[j],
            alpha=0.8
        )

# 设置标题和标签
plt.xticks(rotation=45, fontsize=12)
plt.title('3号线客流量')
plt.xlabel('月份')
plt.ylabel('客流量')

plt.savefig("23-24-3&&T.png", dpi=300, bbox_inches='tight')
# 显示图形
plt.show()

# 输出各个月份的统计信息
for month in valid_months:
    column_data = monthly_data[month]
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = [x for x in column_data if x < lower_bound or x > upper_bound]

    print(f"月份 {month} 的统计信息:")
    print(f"第一四分位数 (Q1): {Q1}")
    print(f"第三四分位数 (Q3): {Q3}")
    print(f"四分位距 (IQR): {IQR}")
    print(f"异常值: {outliers}\n")