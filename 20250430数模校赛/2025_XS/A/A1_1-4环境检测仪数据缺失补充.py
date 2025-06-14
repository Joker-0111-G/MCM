import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 高斯模型
def gaussian(x, A, mu, sigma):
    return A * np.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))

def repair_radiation_data(input_file, output_file):
    df = pd.read_excel(input_file)
    df.columns = ['时间', '辐照强度']
    df['时间'] = pd.to_datetime(df['时间']).dt.floor('min')
    df = df.drop_duplicates('时间').set_index('时间').sort_index()

    # 重建完整时间索引（按分钟）
    full_index = pd.date_range(df.index.min(), df.index.max(), freq='1min')
    df = df.reindex(full_index)
    df.index.name = '时间'
    df['原始'] = df['辐照强度'].notna()

    days = pd.unique(df.index.date)
    all_dfs = []

    for day in days:
        day_df = df.loc[str(day)].copy()
        day_df['分钟'] = (day_df.index - day_df.index.normalize()).total_seconds() / 60

        valid_data = day_df.loc[day_df['辐照强度'].notna(), ['分钟', '辐照强度']]
        if len(valid_data) >= 10:
            try:
                x = valid_data['分钟'].values
                y = valid_data['辐照强度'].values
                p0 = [y.max(), x[np.argmax(y)], (x.max() - x.min()) / 4]
                popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=8000)
                day_df['拟合值'] = gaussian(day_df['分钟'], *popt).clip(lower=0)
            except Exception as e:
                print(f"拟合失败：{day}，使用滑动平均。错误：{e}")
                day_df['拟合值'] = valid_data['辐照强度'].rolling(6, min_periods=1).mean().reindex(day_df.index, method='nearest').fillna(0)
        else:
            day_df['拟合值'] = valid_data['辐照强度'].rolling(6, min_periods=1).mean().reindex(day_df.index, method='nearest').fillna(0)

        day_df['修复强度'] = day_df['辐照强度']
        day_df.loc[day_df['辐照强度'].isna(), '修复强度'] = day_df.loc[day_df['辐照强度'].isna(), '拟合值']

        all_dfs.append(day_df)


    final_df = pd.concat(all_dfs)
    final_df[['辐照强度', '拟合值', '修复强度', '原始']].to_excel(output_file)

    # 可视化最近一天
    last_day = final_df.index.max().normalize()
    plot_df = final_df.loc[last_day:last_day + pd.Timedelta(days=1)]
    plt.figure(figsize=(15, 5))
    plt.plot(plot_df.index, plot_df['辐照强度'], label='原始值', alpha=0.6)
    plt.plot(plot_df.index, plot_df['修复强度'], '--', label='修复后')
    plt.title("辐照强度拟合修复示例（最近一天）")
    plt.xlabel("时间")
    plt.ylabel("W/m²")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("辐照强度_修复示意图.png")
    plt.close()

    print("✅ 修复完成，结果已保存：", output_file)

# 执行示例
repair_radiation_data("电站1环境检测仪数据.xlsx", "电站1_辐照强度_修复结果.xlsx")
