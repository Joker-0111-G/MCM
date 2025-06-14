import numpy as np
import matplotlib.pyplot as plt

# 示例数据
x = np.array([250, 275, 300, 325, 350])
y1 = np.array([1.4, 3.4, 6.7, 19.3, 43.6])
y2 = np.array([6.32, 8.25, 12.28, 25.97, 41.08])

# 分别对y1和y2进行二次拟合
coefficients1 = np.polyfit(x, y1, 2)  # 将阶数更改为2
coefficients2 = np.polyfit(x, y2, 2)  # 将阶数更改为2

# 提取拟合得到的系数（注意现在是三个系数：a, b, c）
a1, b1, c1 = coefficients1
a2, b2, c2 = coefficients2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 打印拟合得到的系数
print(f'拟合得到的二次方程为: y1 = {a1:.2f}x^2 + {b1:.2f}x + {c1:.2f}')
print(f'拟合得到的二次方程为: y2 = {a2:.2f}x^2 + {b2:.2f}x + {c2:.2f}')

# 使用matplotlib绘制原始数据点和拟合的曲线
plt.scatter(x, y1, color='blue', label='乙醇转化率原始数据')
plt.scatter(x, y2, color='green', label='C4烯烃选择性原始数据')  # 使用不同颜色区分y1和y2

# 生成拟合曲线的x值范围
x_fit1 = np.linspace(min(x), max(x), 100)
y_fit1 = a1 * x_fit1 ** 2 + b1 * x_fit1 + c1
x_fit2 = np.linspace(min(x), max(x), 100)
y_fit2 = a2 * x_fit2 ** 2 + b2 * x_fit2 + c2

# 绘制拟合曲线，使用不同颜色区分
plt.plot(x_fit1, y_fit1, color='red', label='乙醇转化率拟合曲线')
plt.plot(x_fit2, y_fit2, color='purple', label='C4烯烃选择性拟合曲线')  # 使用不同颜色区分拟合曲线

plt.title('B1')
plt.legend()

# 显示图形
plt.show()