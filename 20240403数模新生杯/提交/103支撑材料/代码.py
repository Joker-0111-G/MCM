# 导入所需库
import pulp

prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)


# 定义利润、机床工时和销售上限
Profits = [10, 9, 3, 5, 11, 9, 8]
MachineTimes = [
    [0.6, 0.1, 0.2, 0.05, 0],
    [0.7, 0.1, 0, 0.08, 0],
    [0, 0, 0.4, 0, 0.01],
    [0.3, 0.3, 0, 0.06, 0],
    [0.6, 0, 0.2, 0.1, 0.05],
    [0, 0.6, 0, 0, 0.08],
    [0.5, 0, 0.6, 0.08, 0.05]
]
SalesLimits = [
    [600, 800, 200, 0, 700, 300, 200],
    [500, 600, 300, 300, 500, 200, 250],
    [200, 500, 400, 200, 500, 0, 300],
    [300, 400, 0, 400, 300, 500, 100],
    [0, 200, 300, 200, 900, 200, 0],
    [400, 300, 100, 300, 800, 400, 100]
]
NumMachines = [6,3,4,2,1]  # 五种机床各自的数量

# 定义产品数量、月份和机床类型等常量
num_products = 7
num_months = 6
num_machine_types = len(NumMachines)  # 假设NumMachines是一个定义了机床类型数量的列表

# 定义机器设备数量和每月工作天数
working_days_per_month = 22  # 每月工作天数
working_hours_per_day = 8  # 每天工作小时数
total_months = 6  # 总月数

# 定义每个月需要维修的设备
MaintenanceMonths = {
    1: {1: 2},  # 一月：2台机床1整修
    2: {2: 1},  # 二月：1台机床2整修
    3: {4: 1},  # 三月：1台机床4整修
    4: {3: 1},  # 四月：1台机床3整修
    5: {1: 1, 2: 1},  # 五月：1台机床1和1台机床2整修
    6: {3: 1, 5: 1}  # 六月：1台机床3和1台机床5整修
}

# 计算每月每种机床的可用工时
available_machine_hours = {
    month: {
        machine_type: (NumMachines[machine_type - 1] - maint.get(machine_type, 0)) * working_days_per_month * working_hours_per_day
        for machine_type in range(1, len(NumMachines) + 1)
    } for month, maint in MaintenanceMonths.items()
}

# 填充没有整修数据的月份
for month in range(1, 7):  # 假设有6个月，从1月到6月
    if month not in available_machine_hours:
        available_machine_hours[month] = {
            machine_type: NumMachines[machine_type - 1] * machine_hours_per_month
            for machine_type in range(1, len(NumMachines) + 1)
        }

# 初始化库存量（一月初所有零件库存量为0）
Inventory = [[0] * 7 for _ in range(6)]
# 库存目标（六月底每种零件的库存量为60个）
InventoryTargets = [60] * 7

# 设置线性规划问题
prob = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

# 定义决策变量
# 每个产品每月的生产数量
product_vars = pulp.LpVariable.dicts("Product",((month, product) for month in range(1, num_months + 1) for product in range(num_products)),lowBound=0,cat='Continuous')
# 添加库存决策变量
InventoryVars= pulp.LpVariable.dicts("Inventory",((month, product) for month in range(1, num_months + 1)for product in range(num_products)),lowBound=0,cat='Continuous')
# 创建销售量变量字典，一月到六月，七种零件
SalesVars = pulp.LpVariable.dicts("Sales",((i, j) for i in range(1, num_months + 1) for j in range(1, num_products + 1)),lowBound=0, cat='Integer')
# 目标函数：最大化总利润
prob += pulp.lpSum([Profits[product] * product_vars[(month, product)]
                    for month in range(1, num_months + 1)
                    for product in range(num_products)])

# 约束条件
# 1. 库存限制条件
for month in range(1, num_months + 1):
    for product in range(num_products):
        if month < 6:  # 对于一月到五月
            prob += InventoryVars[(month, product)] <= 100, f"InventoryLimit_{month}_{product}"
        else:  # 对于六月
            prob += InventoryVars[(month, product)] == 60, f"InventoryLimit_{month}_{product}"

# 2. 销售上限约束
for month in range(1, num_months + 1):
    for product in range(num_products):
        prob += product_vars[(month, product)] <= SalesLimits[month-1][product], f"SalesLimit_{month}_{product}"

# 3. 机床工时约束
for month in range(1, num_months + 1):
    for machine_type in range(1, num_machine_types + 1):
        prob += pulp.lpSum([MachineTimes[product][machine_type-1] * product_vars[(month, product)]
                            for product in range(num_products)]) <= available_machine_hours[month][machine_type], f"MachineTime_{month}_{machine_type}"

# 求解问题
prob.solve()

# 输出结果
print("生产决策：")
for month in range(1, num_months + 1):
    print(f"月份 {month}:")
    for product in range(num_products):
        print(f"  产品 {product+1}: {product_vars[(month, product)].value()} 个")
print(f"总利润：{pulp.value(prob.objective)}")
