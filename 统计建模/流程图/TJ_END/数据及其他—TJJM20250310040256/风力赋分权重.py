import pandas as pd
import re


# 定义一个函数来提取风力等级
def extract_wind_level(wind_info):
    # 使用正则表达式查找风力等级
    match = re.search(r'(\d+)级', wind_info)
    if match:
        return int(match.group(1))
    return None


# 从 Excel 文件读取数据
file_path = '23-24西安风力统计.xlsx'
data = pd.read_excel(file_path)

# 计算风力赋分
data['风力赋分'] = data['风力风向'].apply(extract_wind_level)

# 保存结果到新的 Excel 文件
data.to_excel('wind_score_result.xlsx', index=False)
