import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# 设置Pandas显示选项以确保完整输出
pd.set_option('display.max_columns', None)     # 显示所有列
pd.set_option('display.expand_frame_repr', False)  # 禁止自动换行
pd.set_option('display.max_colwidth', None)    # 显示完整列内容
pd.set_option('display.precision', 6)          # 设置小数精度为6位

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设数据已加载为DataFrame `data`
data = pd.read_excel('附件1wl.xlsx')

subjects = data['子对象编码'].unique()

results = []
for subj in subjects:
    subset = data[data['子对象编码'] == subj]
    human = subset['人工测试最终成绩']
    ai1 = subset['智能测试1']
    ai2 = subset['智能测试2']

    # 计算统计量
    stats = {
        '子对象': subj,
        '人工均值': human.mean(),
        '人工标准差': human.std(),
        '人工偏度': skew(human),
        'AI1均值': ai1.mean(),
        'AI1标准差': ai1.std(),
        'AI1偏度': skew(ai1),
        'AI2均值': ai2.mean(),
        'AI2标准差': ai2.std(),
        'AI2偏度': skew(ai2)
    }
    results.append(stats)

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(human, kde=True, label='人工', color='blue')
    sns.histplot(ai1, kde=True, label='AI1', color='green')
    sns.histplot(ai2, kde=True, label='AI2', color='red')
    plt.title(f'{subj}分数分布')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(data=[human, ai1, ai2], palette="Set2")
    plt.xticks([0, 1, 2], ['人工', 'AI1', 'AI2'])
    plt.title(f'{subj}箱线图')
    plt.tight_layout()
    plt.show()

# 转换为DataFrame输出结果
results_df = pd.DataFrame(results)
print(results_df)