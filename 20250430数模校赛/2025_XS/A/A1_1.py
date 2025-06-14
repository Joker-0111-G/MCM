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

def load_and_clean_data(power_file, irrad_file):
    """
    加载并清洗发电和辐照数据
    :return: 清洗后的发电数据、辐照数据
    """
    # ========== 1. 发电数据处理 ==========
    df_power = pd.read_excel(
        power_file,
        parse_dates=['时间'],
        usecols=['时间', '当日累计发电量kwh']
    ).rename(columns={'当日累计发电量kwh': '累计发电量'})

    df_power = df_power.sort_values('时间').set_index('时间')
    hourly_power = df_power.resample('1h').last().ffill()
    hourly_power['小时发电量'] = hourly_power.groupby(hourly_power.index.date)['累计发电量'].diff().fillna(0)
    hourly_power['小时发电量'] = hourly_power['小时发电量'].clip(lower=0)

    # ========== 2. 辐照数据处理 ==========
    df_irrad = pd.read_excel(
        irrad_file,
        parse_dates=['时间'],
        usecols=['时间', '辐照强度w/m2']
    ).rename(columns={'辐照强度w/m2': '辐照强度'})

    df_irrad = df_irrad.set_index('时间').resample('1h').mean()

    # ========== 3. 高级清洗（夜间修正数据）==========
    # 处理发电数据夜间值
    night_mask_power = (hourly_power.index.hour >= NIGHT_HOURS[0]) & (hourly_power.index.hour < NIGHT_HOURS[1])
    hourly_power.loc[night_mask_power, '小时发电量'] = 0

    # 处理辐照数据夜间值
    night_mask_irrad = (df_irrad.index.hour >= NIGHT_HOURS[0]) & (df_irrad.index.hour < NIGHT_HOURS[1])
    df_irrad.loc[night_mask_irrad, '辐照强度'] = 0

    return hourly_power, df_irrad

def export_to_excel(df_power, df_irrad, output_power_path, output_irrad_path):
    """数据导出到Excel"""
    # 发电数据导出
    if not df_power.empty:
        df_power.to_excel(output_power_path, index=True)
        print(f"发电数据已保存到 {output_power_path}")
    else:
        print("警告：发电数据为空，未生成文件")

    # 辐照数据导出
    if not df_irrad.empty:
        df_irrad.to_excel(output_irrad_path, index=True)
        print(f"辐照数据已保存到 {output_irrad_path}")
    else:
        print("警告：辐照数据为空，未生成文件")

if __name__ == "__main__":
    input_files = {
        'power_file': '电站4发电数据.xlsx',
        'irrad_file': '电站4环境监测仪数据.xlsx'
    }

    try:
        # 清洗数据
        df_power, df_irrad = load_and_clean_data(
            input_files['power_file'],
            input_files['irrad_file']
        )

        # 导出数据
        export_to_excel(
            df_power,
            df_irrad,
            '清洗后的发电数据.xlsx',
            r'A1_辐射数据_缺失补充\电站4环境检测仪数据_每小时汇总.xlsx'
        )

        # 生成趋势图
        plt.figure(figsize=(15, 5))
        plt.plot(df_power.index, df_power['小时发电量'], label='发电量')
        plt.plot(df_irrad.index, df_irrad['辐照强度'], label='辐照强度', alpha=0.7)
        plt.title('发电量与辐照强度趋势')
        plt.legend()
        plt.savefig('电站4趋势对比图.png')
        plt.show()

    except Exception as e:
        print(f"处理失败: {str(e)}")