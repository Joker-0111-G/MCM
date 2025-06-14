import pandas as pd

# 读取原始Excel文件
df = pd.read_excel('A1_天气数据_重复值整合\电站1天气数据整合.xlsx')

# 将时间列转换为datetime类型
df['时间'] = pd.to_datetime(df['时间'])

# 筛选出所有时间重复的行并按时间排序
duplicate_rows = df[df.duplicated('时间', keep=False)].sort_values('时间')

# 保存到新的Excel文件
duplicate_rows.to_excel('output.xlsx', index=False)

print("处理完成，结果已保存至output.xlsx")

