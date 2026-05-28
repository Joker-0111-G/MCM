import pandas as pd

# 定义温度区间和对应的赋分
score_intervals = [
    (-float('inf'), -10, 3),
    (-10, 0, 3),
    (0, 15, 1),
    (15, 25, 0),
    (25, 30, 2),
    (30, float('inf'), 4)
]


def calculate_score(temperature):
    for lower, upper, score in score_intervals:
        if lower <= temperature < upper:
            return score
    return None


# 从 Excel 文件读取数据
file_path = '23-24西安温度统计.xlsx'
data = pd.read_excel(file_path)

# 提取温度数值并计算平均温度
data['最高温'] = data['最高温'].str.rstrip('°').astype(float)
data['最低温'] = data['最低温'].str.rstrip('°').astype(float)
data['平均温度'] = (data['最高温'] + data['最低温']) / 2

# 计算温度赋分
data['温度赋分'] = data['平均温度'].apply(calculate_score)

# 保存结果到新的 Excel 文件
data.to_excel('temperature_score_result.xlsx', index=False)
