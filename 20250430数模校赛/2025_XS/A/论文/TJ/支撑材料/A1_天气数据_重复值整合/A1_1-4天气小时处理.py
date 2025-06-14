import pandas as pd
import numpy as np

def process_weather_hourly(file_path):
    # 读取数据
    df = pd.read_excel(file_path, parse_dates=['时间'])

    # 保留所需列
    df = df[['时间', '当前温度', '最高温度', '最低温度', '天气', '风向', '风速', '湿度', '日出时间', '日落时间']]

    # 提取日期
    df['日期'] = df['时间'].dt.date

    # 提取每个日期中第一次出现的日出/日落时间
    sunrise_sunset = df.groupby('日期')[['日出时间', '日落时间']].first().reset_index()

    # 时间四舍五入到整点（分钟 >=30 进1小时）
    df['时间'] = df['时间'].dt.round('h')

    # 去重：防止同一小时有多条记录
    df = df.drop_duplicates('时间').set_index('时间')

    # 构造完整小时时间索引
    time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    df_full = df.reindex(time_range)

    # 数值列插值
    numeric_cols = ['当前温度', '最高温度', '最低温度', '风速', '湿度']
    df_full[numeric_cols] = df_full[numeric_cols].interpolate(method='linear')

    # 类别列用前向填充
    df_full['天气'] = df_full['天气'].ffill()
    df_full['风向'] = df_full['风向'].ffill()

    # 重置索引
    df_full.reset_index(inplace=True)
    df_full.rename(columns={'index': '时间'}, inplace=True)

    # 添加日期列用于合并
    df_full['日期'] = df_full['时间'].dt.date

    # 合并日出/日落时间
    df_full = df_full.merge(sunrise_sunset, on='日期', how='left')

    # 删除辅助列
    df_full.drop(columns=['日期'], inplace=True)

    return df_full

# 使用示例
result = process_weather_hourly('电站4天气数据整合.xlsx')
result.to_excel('电站4天气数据整合_每小时汇总.xlsx', index=False)

print("整理完成，已保存为 '电站4天气数据整合_每小时汇总.xlsx'")
