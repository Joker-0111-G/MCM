history_data = [170 185 155 165 200 238 267 291 326 298 315 390 406 469 530
515 520 600 630 700 759];
n = length(history_data);
X0 = history_data&apos;;% 累加生成序列
history_data_agg = cumsum(history_data);
X1 = history_data_agg&apos;;% 计算数据矩阵 B 和数据向量 Y
B = zeros(n - 1, 2);
Y = zeros(n - 1, 1);
for i = 1:n – 1
B(i, 1) = -0.5 * (X1(i) + X1(i + 1));
B(i, 2) = 1;
Y(i) = X0(i + 1);
end% 计算 GM(1,1)微分方程的参数 a 和 u
A = (B&apos; * B) \ B&apos; * Y;
a = A(1);
u = A(2);% 建立灰色预测模型
XX0 = zeros(n, 1);
XX0(1) = X0(1);
for i = 2:n
XX0(i) = (X0(1) - u / a) * (1 - exp(a)) * exp(-a * (i - 1));
end% 模型精度的后验差检验
e = mean(X0 - XX0);% 求历史数据平均值
aver = mean(X0);% 求历史数据方差
s12 = var(X0);% 求残差方差
s22 = var(X0 - XX0 - e);% 求后验差比值
C = s22 / s12;% 求小误差概率
count = sum(abs((X0 - XX0) - e) < 0.6754 * sqrt(s12));
P = count / n;
if (C < 0.35 && P > 0.95)% 预测精度为一级
m = 5; % 请输入需要预测的年数
fprintf(&apos;往后 %d 各年负荷为：\n&apos;, m);
f = zeros(m, 1);
for i = 1:m
f(i) = (X0(1) - u / a) * (1 - exp(a)) * exp(-a * (i + n - 1));
disp(f(i));
end
else
disp(&apos;灰色预测法不适用&apos;);
end