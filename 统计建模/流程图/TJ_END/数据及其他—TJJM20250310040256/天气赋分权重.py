import pandas as pd

# 定义天气评分映射
weather_score_mapping = {
    '晴': 0,
    '多云': 1,
    '阴': 2,
    '小雨': 3,
    '中雨': 4,
    '大雨': 4,
    '雪': 5,
    '雾': 5
}


def calculate_weather_score(weather):
    # 处理复合天气
    if '~' in weather:
        weather_types = weather.split('~')
        scores = [weather_score_mapping.get(wt.strip()) for wt in weather_types]
        # 过滤掉可能不存在的天气类型
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            return sum(valid_scores) / len(valid_scores)
        return None
    # 处理单一天气
    return weather_score_mapping.get(weather.strip())


# 从 Excel 文件读取数据
file_path = '23-24西安天气数据.xlsx'
data = pd.read_excel(file_path)

# 计算天气评分
data['天气评分'] = data['天气'].apply(calculate_weather_score)

# 保存结果到新的 Excel 文件
data.to_excel('weather_score_result.xlsx', index=False)
