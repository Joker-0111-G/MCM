import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======== 高斯函数 ========
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# ======== 单文件修复 ========
def process_power_data(input_path: str, output_path: str):
    """
    input_path : 原始 Excel（含“时间、当日累计发电量kwh”）
    output_path: 修复后 Excel 保存路径
    """
    # 1) 若目标文件夹不存在，自动创建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2) 读取 & 生成完整 5 min 索引
    raw_df = pd.read_excel(input_path, parse_dates=['时间'])
    df = raw_df.drop_duplicates('时间').set_index('时间').sort_index()
    full_index = pd.date_range(df.index.min().floor('5min'),
                               df.index.max().ceil('5min'),
                               freq='5min', name='时间')
    df = df.reindex(full_index)
    df['原始标记'] = df['当日累计发电量kwh'].notna()

    # 3) 功率、累计差
    df['累计发电量kwh'] = df['当日累计发电量kwh'].ffill()
    df['功率'] = df['累计发电量kwh'].diff().clip(lower=0).fillna(0)

    # 4) 按天处理
    all_dfs = []
    for day, day_df in df.groupby(df.index.date):
        day_df = day_df.copy()
        day_df['当日分钟'] = (day_df.index - day_df.index.normalize()).total_seconds() / 60

        # —— 高斯拟合
        valid = (day_df['功率'] > 0) & day_df['原始标记']
        if valid.sum() > 10:
            x, y = day_df.loc[valid, ['当日分钟', '功率']].values.T
            try:
                dur = x.max() - x.min()
                popt, _ = curve_fit(gaussian, x, y, p0=[y.max(), x.mean(), dur / 2], maxfev=5000)
                day_df['拟合功率'] = gaussian(day_df['当日分钟'], *popt).clip(lower=0)
            except Exception:
                day_df['拟合功率'] = day_df['功率'].rolling(6, min_periods=1).mean()
        else:
            day_df['拟合功率'] = day_df['功率']

        # —— 修复规则
        day_df['累计差'] = day_df['累计发电量kwh'].diff().fillna(0)
        day_df['修复功率'] = day_df['功率']

        # 1) 功率=0 & 累计差=0
        flat = (day_df['功率'] == 0) & (day_df['累计差'] == 0)
        missing = flat & (~day_df['原始标记'])          # 真缺失
        keep0   = flat & day_df['原始标记']            # 夜间/停机
        day_df.loc[missing, '修复功率'] = day_df.loc[missing, '拟合功率']
        day_df.loc[keep0,   '修复功率'] = 0

        # 2) 22 点-次日 2 点 强制 0
        night = (day_df.index.hour >= 22) | (day_df.index.hour < 2)
        day_df.loc[night, '修复功率'] = 0

        # 3) 0 → 大跳增
        jump_thr = 400
        prev = day_df['功率'].shift(1).fillna(0)
        jump_mask = (prev == 0) & (day_df['功率'] > jump_thr)
        day_df.loc[jump_mask, '修复功率'] = day_df.loc[jump_mask, '拟合功率']

        # 4) 累计突增前段回填
        big = day_df['累计差'] > 300
        for ts in day_df.index[big]:
            for i in range(1, 13):
                idx = ts - pd.Timedelta(minutes=5 * i)
                if idx not in day_df.index:
                    break
                if day_df.at[idx, '功率'] > 0 or day_df.at[idx, '累计差'] != 0:
                    break
                if day_df.at[idx, '修复功率'] == 0:
                    day_df.at[idx, '修复功率'] = day_df.at[idx, '拟合功率']

        all_dfs.append(day_df)

    merged = pd.concat(all_dfs)
    merged['修复累计发电量'] = merged['修复功率'].fillna(0).cumsum() + merged['累计发电量kwh'].iloc[0]

    merged[['累计发电量kwh', '功率', '拟合功率', '修复功率', '修复累计发电量']].to_excel(output_path)
    print(f"✅ {os.path.basename(input_path)} 处理完成 → {output_path}")


# ======== 批量调用 ========
if __name__ == '__main__':
    # 待处理的电站 Excel 文件名
    stations = ["电站1发电数据.xlsx",
                "电站2发电数据.xlsx",
                "电站3发电数据.xlsx",
                "电站4发电数据.xlsx"]

    # 输出目录
    out_dir = r"A1_发电数据_缺失值补充"
    os.makedirs(out_dir, exist_ok=True)

    for fname in stations:
        in_path  = fname
        base     = os.path.splitext(fname)[0] + "_修复结果.xlsx"
        out_path = os.path.join(out_dir, base)
        process_power_data(in_path, out_path)
