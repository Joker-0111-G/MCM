import pandas as pd
import numpy as np
from scipy import stats
from openpyxl.styles import PatternFill
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 夜间时段定义
NIGHT_HOURS = (0, 5)  # 0:00-5:00视为夜间
NIGHT_FILL = PatternFill(start_color='FFF2CC', end_color='FFF2CC', fill_type='solid')


def load_and_clean_data(power_file, weather_file, irrad_file):
    """
    加载并清洗三个数据文件
    :return: 清洗后的发电数据、气象数据、辐照数据
    """
    # ========== 1. 发电数据处理 ==========
    df_power = pd.read_excel(
        power_file,
        parse_dates=['时间'],
        usecols=['时间', '当日累计发电量kwh']
    ).rename(columns={'当日累计发电量kwh': '累计发电量'})

    df_power = df_power.sort_values('时间').set_index('时间')
    hourly_power = df_power.resample('1h').last().ffill()  # 按小时重采样并填充
    hourly_power['小时发电量'] = hourly_power.groupby(hourly_power.index.date)['累计发电量'].diff().fillna(0)
    hourly_power['小时发电量'] = hourly_power['小时发电量'].clip(lower=0)  # 清除负值

    # ========== 2. 气象数据处理 ==========
    df_weather = pd.read_excel(
        weather_file,
        parse_dates=['时间'],
        usecols=['时间', '当前温度', '天气', '风向', '风速', '湿度']
    )

    # 检查并去除重复的时间戳
    df_weather = df_weather.drop_duplicates(subset='时间', keep='first')

    # 按小时重新索引并插值
    full_idx = pd.date_range(
        start=df_weather['时间'].min().floor('h'),  # 使用小写 'h'
        end=df_weather['时间'].max().ceil('h'),  # 使用小写 'h'
        freq='h'
    )
    df_weather = (
        df_weather.set_index('时间')
        .reindex(full_idx)
        .rename_axis('时间')
    )

    # 数值型插值
    num_cols = ['当前温度', '风速', '湿度']
    df_weather[num_cols] = df_weather[num_cols].interpolate(method='linear')

    # 类别型填充
    cat_cols = ['天气', '风向']
    df_weather[cat_cols] = df_weather[cat_cols].ffill().fillna('未知')

    # ========== 3. 辐照数据处理 ==========
    df_irrad = pd.read_excel(
        irrad_file,
        parse_dates=['时间'],
        usecols=['时间', '辐照强度w/m2']
    ).rename(columns={'辐照强度w/m2': '辐照强度'})

    # 按小时重采样并计算均值
    df_irrad = df_irrad.set_index('时间').resample('1h').mean()

    # ========== 4. 异常值处理（Z-Score） ==========
    numeric_cols = ['当前温度', '风速', '湿度']
    z_scores = df_weather[numeric_cols].apply(stats.zscore)
    df_weather = df_weather[(z_scores.abs() < 3).all(axis=1)]  # Z-score 大于 3 的数据被去除

    # ========== 5. 高级清洗（夜间修正数据）==========
    # 确保夜间数据的布尔索引与数据的长度一致
    night_mask_power = (hourly_power.index.hour >= NIGHT_HOURS[0]) & (hourly_power.index.hour < NIGHT_HOURS[1])
    hourly_power.loc[night_mask_power, '小时发电量'] = 0

    night_mask_irrad = (df_irrad.index.hour >= NIGHT_HOURS[0]) & (df_irrad.index.hour < NIGHT_HOURS[1])
    df_irrad.loc[night_mask_irrad, '辐照强度'] = 0

    # 同样处理气象数据
    night_mask_weather = (df_weather.index.hour >= NIGHT_HOURS[0]) & (df_weather.index.hour < NIGHT_HOURS[1])
    df_weather.loc[night_mask_weather, ['当前温度', '风速', '湿度']] = 0

    # ========== 6. 数据返回 ==========
    return hourly_power, df_weather, df_irrad


def export_to_excel(df_power, df_weather, df_irrad, output_power_path, output_weather_path, output_irrad_path):
    """分别将清洗后的数据存储到不同的Excel文件"""
    # 发电数据导出
    if df_power.empty:
        print("发电数据为空，无法导出。")
    else:
        df_power.to_excel(output_power_path, index=True)
        print(f"发电数据已保存到 {output_power_path}")

    # 气象数据导出
    if df_weather.empty:
        print("气象数据为空，无法导出。")
    else:
        df_weather.to_excel(output_weather_path, index=True)
        print(f"气象数据已保存到 {output_weather_path}")

    # 辐照数据导出
    if df_irrad.empty:
        print("辐照数据为空，无法导出。")
    else:
        df_irrad.to_excel(output_irrad_path, index=True)
        print(f"辐照数据已保存到 {output_irrad_path}")


if __name__ == "__main__":
    input_files = {
        'power_file': '电站1发电数据.xlsx',  # 确保文件路径正确
        'weather_file': '电站1天气数据.xlsx',  # 确保文件路径正确
        'irrad_file': '电站1环境检测仪数据.xlsx'  # 确保文件路径正确
    }

    try:
        # 清洗数据并返回
        df_power, df_weather, df_irrad = load_and_clean_data(**input_files)

        # 导出数据到不同的Excel文件
        export_to_excel(
            df_power, df_weather, df_irrad,
            '清洗后的发电数据.xlsx',
            '清洗后的气象数据.xlsx',
            '清洗后的辐照数据.xlsx'
        )

        # 生成趋势图
        plt.figure(figsize=(15, 5))
        plt.plot(df_power.index, df_power['小时发电量'], label='发电量')
        plt.plot(df_irrad.index, df_irrad['辐照强度'], label='辐照强度', alpha=0.7)
        plt.title('发电量与辐照强度趋势')
        plt.legend()
        plt.savefig('趋势对比图.png')
        plt.show()

    except Exception as e:
        print(f"处理失败: {str(e)}")
