import pandas as pd


def check_all_hours(data):
    # 将时间列转换为 datetime 类型
    data['时间'] = pd.to_datetime(data['时间'])
    # 提取小时信息
    hours = data['时间'].dt.hour.unique()
    # 判断是否包含所有 24 个小时
    return set(hours) == set(range(24))


# 假设数据存储在 Excel 文件中，文件名为 'radiation_data.xlsx'
# 你需要将文件名替换为实际的文件名
try:
    df = pd.read_excel('电站4环境监测仪数据.xlsx')
    result = check_all_hours(df)
    if result:
        print("数据包含所有小时段。")
    else:
        print("数据未包含所有小时段。")
except FileNotFoundError:
    print("错误：未找到指定的 Excel 文件。")
except Exception as e:
    print(f"发生未知错误：{e}")
