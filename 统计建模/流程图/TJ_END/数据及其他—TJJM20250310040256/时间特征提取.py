import pandas as pd
import numpy as np
from datetime import datetime


def create_date_range(start_year, end_year):
    """生成指定年份范围的日期序列"""
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return dates.to_frame(name='date')


def basic_time_features(df):
    """提取基础时间特征"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday  # 0=周一, 6=周日
    df['is_weekend'] = np.where(df['weekday'].isin([5, 6]), 1, 0)  # 周末为周六(5)和周日(6)
    return df


def fill_holidays(df):
    """填充中国2023-2024年法定节假日（修正节假日合并错误）"""
    # 2023年节假日（需根据实际调整）
    holidays_2023 = [
        '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
        '2023-01-25', '2023-01-26', '2023-04-05', '2023-04-29', '2023-04-30',
        '2023-05-01', '2023-05-02', '2023-06-22', '2023-06-23', '2023-06-24',
        '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03',
        '2023-10-04', '2023-10-05', '2023-10-06'
    ]
    # 2024年节假日（示例，实际需更新，修正之前的重复合并错误）
    holidays_2024 = [
        '2024-01-01', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13',
        '2024-02-14', '2024-02-15', '2024-02-16', '2024-04-04', '2024-04-05',
        '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-6-10',
        '2024-6-11', '2024-6-12', '2024-9-15', '2024-9-16', '2024-10-01',
        '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06'
    ]

    all_holidays = holidays_2023 + holidays_2024  # 修正为合并2023+2024
    df['is_holiday'] = df['date'].dt.strftime('%Y-%m-%d').isin(all_holidays).astype(int)
    return df


def periodic_encoding(df, period, feature, name_prefix):
    """通用周期性编码函数"""
    df[f'{name_prefix}_sin'] = np.sin(2 * np.pi * df[feature] / period)
    df[f'{name_prefix}_cos'] = np.cos(2 * np.pi * df[feature] / period)
    return df


def season_mapping(month):
    """月份映射到季节（返回季节名称）"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:  # 9-11月
        return 'fall'


def seasonal_onehot_encoding(df):
    """季节独热编码（强制转换为0/1整数类型）"""
    df['season'] = df['month'].apply(season_mapping)  # 先映射到季节名称
    # 独热编码并显式转换为int类型（解决False/True问题）
    season_dummies = pd.get_dummies(df['season'], prefix='season').astype(int)
    df = pd.concat([df, season_dummies], axis=1)  # 合并到原始数据
    return df


def main():
    # 1. 生成日期范围
    dates_df = create_date_range(2023, 2024)

    # 2. 提取基础时间特征
    df = basic_time_features(dates_df)

    # 3. 填充节假日（修正后的节假日数据）
    df = fill_holidays(df)

    # 4. 周次周期性编码（基于weekday，周期7）
    df = periodic_encoding(df, period=7, feature='weekday', name_prefix='week')

    # 5. 月份周期性编码（周期12）
    df = periodic_encoding(df, period=12, feature='month', name_prefix='month')

    # 6. 季节编码（关键修正：强制转换为int类型）
    df = seasonal_onehot_encoding(df)

    # 7. 保存到Excel
    output_file = 'time_features_2023-2024.xlsx'
    df.to_excel(output_file, index=False)
    print(f"数据已成功保存到 {output_file}")
    print("季节特征已修正为0/1编码（原False/True问题已解决）")


if __name__ == "__main__":
    main()
