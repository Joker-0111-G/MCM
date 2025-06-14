import pandas as pd

# 要处理的四个修复结果文件名
input_files = [
    "电站1发电数据_修复结果.xlsx",
    "电站2发电数据_修复结果.xlsx",
    "电站3发电数据_修复结果.xlsx",
    "电站4发电数据_修复结果.xlsx"
]

for file in input_files:
    try:
        # 读取数据
        df = pd.read_excel(file, parse_dates=['时间'])
        df = df.sort_values('时间').reset_index(drop=True)

        # 计算每日归零的修复累计发电量
        df['修复累计发电量（每日归零）'] = 0.0
        df['日期'] = df['时间'].dt.date
        for date, group in df.groupby('日期'):
            cumulative = group['修复功率'].cumsum()
            df.loc[group.index, '修复累计发电量（每日归零）'] = cumulative
        df.drop(columns=['日期'], inplace=True)

        # 按小时汇总
        hourly_df = df.resample('1h', on='时间').agg({
            '功率': 'sum',
            '拟合功率': 'sum',
            '修复功率': 'sum',
            '累计发电量kwh': 'last',
            '修复累计发电量': 'last',
            '修复累计发电量（每日归零）': 'last'
        }).reset_index()

        # 输出文件名（不覆盖）
        new_filename = file.replace("修复结果.xlsx", "每小时汇总.xlsx")
        hourly_df.to_excel(new_filename, index=False)
        print(f"✅ 已保存：{new_filename}")

    except Exception as e:
        print(f"❌ 处理失败：{file}，错误：{e}")
