import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================== 层次分析法（AHP）权重计算 ==================
def ahp_weights(criteria_names, comparison_matrix):
    """基于判断矩阵计算AHP权重"""
    n = len(criteria_names)
    # 计算特征向量
    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, max_eigenvalue_index].real
    weights = weights / weights.sum()

    # 一致性检验
    CI = (eigenvalues[max_eigenvalue_index].real - n) / (n - 1)
    RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45][n - 1]
    CR = CI / RI
    if CR < 0.1:
        print(f"一致性检验通过 (CR={CR:.3f} < 0.1)")
    else:
        print(f"警告：一致性检验未通过 (CR={CR:.3f} >= 0.1)")

    return {name: weight for name, weight in zip(criteria_names, weights)}


# ================== 主程序 ==================
# 读取数据并分配满分
data = pd.read_excel("附件1wl.xlsx")
full_marks = {"T11": 6, "T13": 9, "T14": 14, "T15": 16}

# 收集所有子对象和算法的指标数据
metrics_list = []

for subj in full_marks.keys():
    subset = data[data['子对象编码'] == subj]
    human = subset['人工测试最终成绩']
    ai1 = subset['智能测试1']
    ai2 = subset['智能测试2']

    # 计算AI1的指标
    mae_ai1 = np.mean(np.abs(ai1 - human))
    pearson_ai1, _ = pearsonr(ai1, human)
    std_ai1 = ai1.std()
    mean_error_ai1 = np.mean(ai1 - human)

    # 计算AI2的指标
    mae_ai2 = np.mean(np.abs(ai2 - human))
    pearson_ai2, _ = pearsonr(ai2, human)
    std_ai2 = ai2.std()
    mean_error_ai2 = np.mean(ai2 - human)

    # 计算一致性（AI1与AI2的相关性）
    consistency, _ = pearsonr(ai1, ai2)

    # 添加到列表（每个子对象生成两条记录：AI1和AI2）
    metrics_list.append({
        '子对象': subj, '算法': 'AI1',
        'MAE': mae_ai1, 'Pearson': pearson_ai1,
        'Std': std_ai1, 'Consistency': consistency,
        'AbsMeanError': abs(mean_error_ai1)
    })
    metrics_list.append({
        '子对象': subj, '算法': 'AI2',
        'MAE': mae_ai2, 'Pearson': pearson_ai2,
        'Std': std_ai2, 'Consistency': consistency,
        'AbsMeanError': abs(mean_error_ai2)
    })

# 转换为DataFrame
metrics_df = pd.DataFrame(metrics_list)

# ================== 指标标准化 ==================
scaler = MinMaxScaler(feature_range=(0, 1))
X = metrics_df[['MAE', 'Pearson', 'Std', 'Consistency', 'AbsMeanError']]

# 处理方向性（MAE、Std、AbsMeanError越小越好）
X_normalized = pd.DataFrame()
X_normalized['MAE'] = 1 - scaler.fit_transform(X[['MAE']]).ravel()  # 关键修改点
X_normalized['Pearson'] = scaler.fit_transform(X[['Pearson']]).ravel()
X_normalized['Std'] = 1 - scaler.fit_transform(X[['Std']]).ravel()
X_normalized['Consistency'] = scaler.fit_transform(X[['Consistency']]).ravel()
X_normalized['AbsMeanError'] = 1 - scaler.fit_transform(X[['AbsMeanError']]).ravel()

# ================== 定义AHP判断矩阵 ==================
# (此处保持原有判断矩阵不变)
criteria_names = ['MAE', 'Pearson', 'Std', 'Consistency', 'AbsMeanError']
judgement_matrix = np.array([
    [1, 1 / 3, 5, 3, 5],
    [3, 1, 7, 5, 7],
    [1 / 5, 1 / 7, 1, 1 / 3, 1],
    [1 / 3, 1 / 5, 3, 1, 3],
    [1 / 5, 1 / 7, 1, 1 / 3, 1]
])

# 计算AHP权重
weights = ahp_weights(criteria_names, judgement_matrix)
print("\nAHP权重分配：")
for k, v in weights.items():
    print(f"{k}: {v:.4f}")

# ================== 计算综合得分 ==================
metrics_df['综合得分'] = (X_normalized * pd.Series(weights)).sum(axis=1)

# 按算法分组计算最终得分
final_scores = metrics_df.groupby('算法')['综合得分'].mean()
print("\n算法最终得分：")
print(final_scores)

# ================== 可视化对比 ==================
plt.figure(figsize=(10, 6))

# 综合得分柱状图
plt.subplot(1, 2, 1)
final_scores.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('AI算法综合评价对比')
plt.ylabel('综合得分')
plt.ylim(0.5, 0.8)
plt.grid(axis='y', linestyle='--')

# 各维度雷达图
plt.subplot(1, 2, 2, polar=True)
categories = criteria_names
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# AI1数据
values_ai1 = X_normalized[metrics_df['算法'] == 'AI1'].mean().values.tolist()
values_ai1 += values_ai1[:1]
angles += angles[:1]

# AI2数据
values_ai2 = X_normalized[metrics_df['算法'] == 'AI2'].mean().values.tolist()
values_ai2 += values_ai2[:1]

# 绘图
plt.plot(angles, values_ai1, 'o-', linewidth=2, label='AI1')
plt.fill(angles, values_ai1, alpha=0.25)
plt.plot(angles, values_ai2, 'o-', linewidth=2, label='AI2')
plt.fill(angles, values_ai2, alpha=0.25)

# 标签设置
plt.thetagrids(np.degrees(angles[:-1]), categories)
plt.title('各维度表现雷达图')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()