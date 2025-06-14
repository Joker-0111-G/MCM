
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局配置
stations = {
    '电站1': {'capacity': 4998.30, 'tilt': 15, 'clean_date': '2024-12-05'},
    '电站2': {'capacity': 5581.00, 'tilt': 0, 'clean_date': '2025-01-02'},
    '电站3': {'capacity': 4456.00, 'tilt': 0, 'clean_date': '2025-01-02'},
    '电站4': {'capacity': 1794.61, 'tilt': 0, 'clean_date': '2025-01-02'}
}

def compute_theoretical_power(row, eta, alpha1, beta, tref, h0, h1, ifactor):
    G = row['辐照强度']
    T_env = row['环境温度']
    H = row['湿度']
    W = row['风速']
    if G <= 0:
        return 0
    T_back = T_env + G / (h0 + h1 * W)
    P_pred = eta * G * (1 - alpha1 * (T_back - tref)) * (1 - beta * H) * ifactor
    return max(P_pred, 0)

def sliding_statistics(series, window=168):
    return series.rolling(window=window, min_periods=1).mean(), series.rolling(window=window, min_periods=1).std()

def dynamic_cleaning_decision(df, cap, p_electric=0.582, theta_safe=20):
    C_clean = 2 * cap
    df = df.copy()
    df['发电损失'] = df['DI'] / 100 * df['实际功率'] * p_electric
    df['累积损失'] = df['发电损失'].cumsum()

    df['清洗建议'] = False
    if (df['累积损失'] >= C_clean).any():
        t_clean = df[df['累积损失'] >= C_clean].index[0]
        df.loc[t_clean:, '清洗建议'] = True
    elif (df['DI均值'] > theta_safe).any():
        t_clean = df[df['DI均值'] > theta_safe].index[0]
        df.loc[t_clean:, '清洗建议'] = True
    return df


def process_station(station_name, config):
    print(f"\n>>> 正在处理：{station_name}")

    # 文件路径
    power_file = f"A2/A2_发电数据/{station_name}发电数据_每小时汇总.xlsx"
    env_file = f"A2/A2_辐射数据/{station_name}环境检测仪数据_每小时汇总.xlsx"
    weather_file = f"A2/A2_天气数据/{station_name}天气数据整合_每小时汇总.xlsx"

    # 读取数据
    power_df = pd.read_excel(power_file, parse_dates=['时间'])[['时间', '修复累计发电量（每日归零）']].rename(
        columns={'修复累计发电量（每日归零）': '累计发电量'})
    env_df = pd.read_excel(env_file, parse_dates=['时间'])[['时间', '辐照强度']]
    weather_df = pd.read_excel(weather_file, parse_dates=['时间'])[['时间', '当前温度', '风速', '湿度']].rename(
        columns={'当前温度': '环境温度'})

    # 合并数据
    df = power_df.merge(env_df, on='时间').merge(weather_df, on='时间')
    df = df.set_index('时间').sort_index()
    df['实际功率'] = df['累计发电量'].diff().fillna(0).clip(lower=0)

    # 参数
    cap = config['capacity']
    beta = config['tilt']
    area = cap / 0.18 / 1000
    eta = cap / area / 1000
    tref = 25
    alpha1 = 0.004
    beta_h = 0.02
    k = 0.2 if beta > 0 else 0
    ifactor = 1 - k * np.sin(np.radians(beta))
    clean_date = pd.to_datetime(config['clean_date'])

    # 拟合模型
    train_df = df[(df.index.date >= clean_date.date()) & (df.index.date <= (clean_date + timedelta(days=2)).date())]
    valid = train_df[(train_df['辐照强度'] > 200) & (train_df['实际功率'] > 0)]

    def model_func(xdata, h0, h1):
        G, T_env, H, W = xdata
        T_back = T_env + G / (h0 + h1 * W)
        return eta * G * (1 - alpha1 * (T_back - tref)) * (1 - beta_h * H) * ifactor

    xdata = np.array([valid['辐照强度'], valid['环境温度'], valid['湿度'], valid['风速']])
    ydata = valid['实际功率'].values
    popt, _ = curve_fit(model_func, xdata, ydata, p0=[20, 5], maxfev=10000)
    h0_fit, h1_fit = popt

    # 理论功率与DI
    df['理论功率'] = df.apply(lambda row: compute_theoretical_power(row, eta, alpha1, beta_h, tref, h0_fit, h1_fit, ifactor), axis=1)
    df['DI'] = (df['理论功率'] - df['实际功率']) / df['理论功率'].replace(0, np.nan) * 100
    df['DI'] = df['DI'].clip(lower=0)
    df['DI均值'], df['DI波动'] = sliding_statistics(df['DI'])

    # 添加清洗时间决策建议
    df = dynamic_cleaning_decision(df, cap)

    # 输出结果
    os.makedirs("A3/ANS_1", exist_ok=True)
    df.reset_index().to_excel(f"A3/ANS_1/{station_name}_DI分析结果_含清洗建议.xlsx", index=False)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['DI'], label='DI(%)', alpha=0.4)
    plt.plot(df.index, df['DI均值'], label='DI均值', color='blue')
    plt.axhline(20, color='green', linestyle='--', label='安全预警阈值20%')
    plt.fill_between(df.index, 0, df['DI'], where=df['清洗建议'], color='red', alpha=0.3, label='清洗建议时段')
    plt.title(f"{station_name} 积灰指数趋势及清洗建议")
    plt.xlabel("时间")
    plt.ylabel("DI (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"A3/ANS_1/{station_name}_清洗决策趋势图.png")
    plt.close()
    print(f">>> 处理完成：{station_name}")

# 主程序
for name, config in stations.items():
    process_station(name, config)
