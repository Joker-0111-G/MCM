import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载
data = pd.read_excel('D:/BBTJJM/现有数据/XGBoost/processed_metro_data.xlsx', parse_dates=['date'])

# 日期特征提取
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter
data['season'] = data['month'] % 12 // 3 + 1

# 时间编码（正弦余弦转换）
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# 滞后特征（前一天客流量）
data['lag_1'] = data.groupby('line_number')['passenger_flow'].shift(1)

# 填充缺失值
data.bfill(inplace=True)

# 特征选择
features = ['aqi', 'wind_force', 'temperature', 'wind_speed', 'month_sin', 'month_cos', 'week_sin', 'week_cos', 'season', 'lag_1']
target = 'passenger_flow'

# 特征与目标变量
X = data[features]
y = data[target]

# 数据划分（训练集：2023-01-01 至 2024-03-31，验证集：2024-04-01 至 2024-04-30）
train_data = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2024-03-31')]
val_data = data[(data['date'] >= '2024-04-01') & (data['date'] <= '2024-04-30')]

X_train = train_data[features]
y_train = train_data[target]
X_val = val_data[features]
y_val = val_data[target]

# 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 参数设置
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.7,
    #'gamma': 0
    'colsample_bytree': 0.8,
    'reg_alpha': 0,
    'reg_lambda': 1
}

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], early_stopping_rounds=10)

# 预测与评估
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f'验证集 RMSE: {rmse:.4f}')
print(f'验证集 MAE: {mae:.4f}')
print(f'验证集 MSE: {mse:.4f}')
print(f'验证集 R²: {r2:.4f}')

# 特征重要性分析
importance = model.get_score(importance_type='weight')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
labels, scores = zip(*importance)

plt.figure(figsize=(10, 6))
plt.barh(labels, scores)
plt.xlabel('特征重要性')
plt.title('XGBoost 特征重要性')
plt.show()

# 残差分析
residuals = y_val - y_pred

# 残差与拟合值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('残差与拟合值散点图')
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.show()

# 残差直方图
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', color='skyblue')
plt.title('残差直方图')
plt.xlabel('残差')
plt.ylabel('频率')
plt.show()
