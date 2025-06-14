import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta

# 电站配置信息
stations = {
    '电站1': {'capacity': 4998.3, 'tilt': 15, 'clean_date': '2024-12-05', 'lat': 41.1},
    '电站2': {'capacity': 5581.0, 'tilt': 0, 'clean_date': '2025-01-02', 'lat': 24.88},
    '电站3': {'capacity': 4456.0, 'tilt': 0, 'clean_date': '2025-01-02', 'lat': 25.03},
    '电站4': {'capacity': 1794.61, 'tilt': 0, 'clean_date': '2025-01-02', 'lat': 38.93},
}

# 理论功率计算函数
def compute_theoretical_power(row, eta, alpha1, beta, tref, h0, h1, ifactor):
    G = row['辐照强度']
    T_env = row['当前温度']
    H = row['湿度']
    W = row['风速']
    if pd.isna(G) or G <= 0:
        return 0
    T_back = T_env + G / (h0 + h1 * W)
    P_pred = eta * G * (1 - alpha1 * (T_back - tref)) * (1 - beta * H) * ifactor
    return max(P_pred, 0)

# 滑动窗口统计
def sliding_statistics(series, window=168):
    return series.rolling(window=window, min_periods=1).mean(), series.rolling(window=window, min_periods=1).std()

# 主处理函数
def process_station(station_name, config):
    print(f"处理 {station_name} ...")

    # 路径拼接
    base_path = "A2"
    power_path = os.path.join(base_path, "A2_发电数据", f"{station_name}发电数据_每小时汇总.xlsx")
    env_path = os.path.join(base_path, "A2_辐射数据", f"{station_name}环境检测仪数据_每小时汇总.xlsx")
    weather_path = os.path.join(base_path, "A2_天气数据", f"{station_name}天气数据整合_每小时汇总.xlsx")

    # 数据读取与合并
    power_df = pd.read_excel(power_path, parse_dates=['时间']).set_index('时间')
    env_df = pd.read_excel(env_path, parse_dates=['时间']).set_index('时间')
    weather_df = pd.read_excel(weather_path, parse_dates=['时间']).set_index('时间')
    df = power_df.join(env_df, how='outer').join(weather_df, how='outer').sort_index()

    # 实际功率计算
    df['实际功率'] = df['累计发电量'].diff().fillna(0).clip(lower=0)

    # 参数准备
    cap = config['capacity']
    beta = config['tilt']
    area = cap / 0.18 / 1000
    eta = cap / area / 1000
    alpha1 = 0.004
    beta_h = 0.02
    tref = 25
    k = 0.2 if beta > 0 else 0
    ifactor = 1 - k * np.sin(np.radians(beta))

    # 清洗后建模数据
    clean_date = pd.to_datetime(config['clean_date'])
    train_df = df[(df.index.date >= clean_date.date()) & (df.index.date <= (clean_date + timedelta(days=2)).date())]
    valid = train_df[(train_df['辐照强度'] > 200) & (train_df['实际功率'] > 0)]

    def model_func(xdata, h0_fit, h1_fit):
        G, T_env, H, W = xdata
        T_back = T_env + G / (h0_fit + h1_fit * W)
        return eta * G * (1 - alpha1 * (T_back - tref)) * (1 - beta_h * H) * ifactor

    # 模型拟合
    xdata = np.array([valid['辐照强度'], valid['当前温度'], valid['湿度'], valid['风速']])
    ydata = valid['实际功率'].values
    popt, _ = curve_fit(model_func, xdata, ydata, p0=[20, 5], maxfev=10000)
    h0_fit, h1_fit = popt

    # 理论功率 & DI 计算
    df['理论功率'] = df.apply(lambda row: compute_theoretical_power(row, eta, alpha1, beta_h, tref, h0_fit, h1_fit, ifactor), axis=1)
    df['DI'] = (df['理论功率'] - df['实际功率']) / df['理论功率'].replace(0, np.nan) * 100
    df['DI'] = df['DI'].clip(lower=0)
    df['DI均值'], df['DI波动'] = sliding_statistics(df['DI'])

    # 阈值设定
    if station_name == '电站1':
        theta1, theta2 = 12, 4
    else:
        theta1, theta2 = 10, 5
    df['预警'] = (df['DI均值'] > theta1) & (df['DI波动'] < theta2)

    # 导出
    df.to_excel(f"{station_name}_DI分析结果.xlsx")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['DI'], label='DI (%)', alpha=0.5)
    plt.plot(df.index, df['DI均值'], label='DI均值', linewidth=2)
    plt.axhline(theta1, color='red', linestyle='--', label='预警阈值')
    plt.title(f"{station_name} 积灰指数趋势")
    plt.ylabel("DI (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{station_name}_DI趋势图.png")
    plt.close()

# 执行处理
for name, conf in stations.items():
    process_station(name, conf)
