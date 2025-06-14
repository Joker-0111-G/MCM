import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据并分配满分值
data = pd.read_excel("附件1wl.xlsx")
full_marks = {"T11": 6, "T13": 9, "T14": 14, "T15": 16}
total_full = sum(full_marks.values())
weights = {k: v / total_full for k, v in full_marks.items()}

# 初始化存储结果的DataFrame
metrics = {
    'MAE_AI1': [], 'MAE_AI2': [],
    'Pearson_AI1': [], 'Pearson_AI2': [],
    'Std_AI1': [], 'Std_AI2': [],
    'Consistency_AI1_AI2': [],
    'MeanError_AI1': [], 'MeanError_AI2': []
}

# 按子对象计算指标
for subj in full_marks.keys():
    subset = data[data['子对象编码'] == subj]
    human = subset['人工测试最终成绩']
    ai1 = subset['智能测试1']
    ai2 = subset['智能测试2']
    weight = weights[subj]

    # MAE
    mae_ai1 = np.mean(np.abs(ai1 - human))
    mae_ai2 = np.mean(np.abs(ai2 - human))
    metrics['MAE_AI1'].append(mae_ai1 * weight)
    metrics['MAE_AI2'].append(mae_ai2 * weight)

    # Pearson相关系数
    r_ai1, _ = pearsonr(ai1, human)
    r_ai2, _ = pearsonr(ai2, human)
    metrics['Pearson_AI1'].append(r_ai1 * weight)
    metrics['Pearson_AI2'].append(r_ai2 * weight)

    # 标准差
    std_ai1 = ai1.std() * weight
    std_ai2 = ai2.std() * weight
    metrics['Std_AI1'].append(std_ai1)
    metrics['Std_AI2'].append(std_ai2)

    # AI间一致性
    r_consistency, _ = pearsonr(ai1, ai2)
    metrics['Consistency_AI1_AI2'].append(r_consistency * weight)

    # 平均误差
    mean_error_ai1 = np.mean(ai1 - human) * weight
    mean_error_ai2 = np.mean(ai2 - human) * weight
    metrics['MeanError_AI1'].append(mean_error_ai1)
    metrics['MeanError_AI2'].append(mean_error_ai2)

# 汇总加权结果
final_metrics = {
    'AI1': {
        'MAE': np.sum(metrics['MAE_AI1']),
        'Pearson': np.sum(metrics['Pearson_AI1']),
        'Std': np.sum(metrics['Std_AI1']),
        'Consistency': np.sum(metrics['Consistency_AI1_AI2']),
        'MeanError': np.sum(metrics['MeanError_AI1'])
    },
    'AI2': {
        'MAE': np.sum(metrics['MAE_AI2']),
        'Pearson': np.sum(metrics['Pearson_AI2']),
        'Std': np.sum(metrics['Std_AI2']),
        'Consistency': np.sum(metrics['Consistency_AI1_AI2']),
        'MeanError': np.sum(metrics['MeanError_AI2'])
    }
}

# 输出结果
print("加权指标汇总：")
for ai in ['AI1', 'AI2']:
    print(f"\n{ai}:")
    for key, value in final_metrics[ai].items():
        print(f"{key}: {value:.4f}")

# 标准化处理（假设以AI1和AI2的最大值作为基准）
max_mae = max(final_metrics['AI1']['MAE'], final_metrics['AI2']['MAE'])
max_std = max(final_metrics['AI1']['Std'], final_metrics['AI2']['Std'])
max_me = max(abs(final_metrics['AI1']['MeanError']), abs(final_metrics['AI2']['MeanError']))


# 计算标准化得分
def normalize(value, max_value, reverse=False):
    if reverse:
        return 1 - (value / max_value)  # 越小越好
    else:
        return value  # 越大越好


scores = {
    'AI1': {
        'MAE_score': normalize(final_metrics['AI1']['MAE'], max_mae, reverse=True),
        'Pearson_score': normalize(final_metrics['AI1']['Pearson'], 1),
        'Std_score': normalize(final_metrics['AI1']['Std'], max_std, reverse=True),
        'Consistency_score': normalize(final_metrics['AI1']['Consistency'], 1),
        'Bias_score': normalize(1 - abs(final_metrics['AI1']['MeanError']) / max_me, 1)
    },
    'AI2': {
        'MAE_score': normalize(final_metrics['AI2']['MAE'], max_mae, reverse=True),
        'Pearson_score': normalize(final_metrics['AI2']['Pearson'], 1),
        'Std_score': normalize(final_metrics['AI2']['Std'], max_std, reverse=True),
        'Consistency_score': normalize(final_metrics['AI2']['Consistency'], 1),
        'Bias_score': normalize(1 - abs(final_metrics['AI2']['MeanError']) / max_me, 1)
    }
}

# 权重分配
#weights = {
#    'Accuracy': 0.4,  # MAE + Pearson
#    'Robustness': 0.2,  # Std
#    'Consistency': 0.2,  # AI间一致性
#    'Bias': 0.2  # MeanError
#}

weights={
    'Accuracy': 0.2469+0.4195,  # MAE + Pearson
    'Robustness': 0.0164,  # Std
    'Consistency':0.0490 ,  # AI间一致性
    'Bias':0.2683  # MeanError
}

# 计算综合得分
for ai in ['AI1', 'AI2']:
    accuracy = (scores[ai]['MAE_score'] + scores[ai]['Pearson_score']) / 2
    robustness = scores[ai]['Std_score']
    consistency = scores[ai]['Consistency_score']
    bias = scores[ai]['Bias_score']

    total_score = (
            accuracy * weights['Accuracy'] +
            robustness * weights['Robustness'] +
            consistency * weights['Consistency'] +
            bias * weights['Bias']
    )
    print(f"{ai} 综合得分: {total_score:.4f}")

# 绘制综合得分对比
labels = ['准确性', '鲁棒性', '一致性', '偏差']
ai1_scores = [
    (scores['AI1']['MAE_score'] + scores['AI1']['Pearson_score']) / 2,
    scores['AI1']['Std_score'],
    scores['AI1']['Consistency_score'],
    scores['AI1']['Bias_score']
]
ai2_scores = [
    (scores['AI2']['MAE_score'] + scores['AI2']['Pearson_score']) / 2,
    scores['AI2']['Std_score'],
    scores['AI2']['Consistency_score'],
    scores['AI2']['Bias_score']
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, ai1_scores, width, label='AI1')
rects2 = ax.bar(x + width/2, ai2_scores, width, label='AI2')

ax.set_ylabel('得分')
ax.set_title('AI算法各维度表现对比')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()