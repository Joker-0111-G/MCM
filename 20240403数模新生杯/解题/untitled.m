% 零件利润和设备工时
profit = [10; 9; 3; 5; 11; 9; 8];
workhours = [0.6 0.7 0 0.3 0.6 0 0.5;
             0.1 0.1 0 0.3 0 0.6 0;
             0.2 0 0.4 0 0.2 0 0.6;
             0.05 0.08 0 0.06 0.1 0 0.08;
             0 0 0.01 0 0.05 0.08 0.05];

% 销售上限
sales_limit = [600 800 200 0 700 300 200;
               500 600 300 300 500 200 250;
               200 500 400 200 500 0 300;
               300 400 0 400 300 500 100;
               0 200 300 200 900 200 0;
               400 300 100 300 800 400 100];

% 月份和设备整修计划
months = 6;
repair_plan = [2 0 0 0 0;
               0 1 0 0 0;
               0 0 0 1 0;
               0 0 1 0 0;
               1 1 0 0 0;
               0 0 1 0 1];

% 变量个数
num_parts = size(profit, 1);
num_machines = size(workhours, 2);
num_variables = num_parts * months + num_machines * months;

% 目标函数系数
f = -[profit; zeros(num_machines * months, 1)];

% 线性不等式约束矩阵
A = zeros(num_parts * months + num_machines * months, num_variables);
b = zeros(num_parts * months + num_machines * months, 1);

% 零件销售约束
for i = 1:num_parts
    A(i, (i-1)*months+1:i*months) = -1;
    b(i) = -100;
end

% 设备工时约束
for i = 1:num_machines
    A(num_parts*months+(i-1)*months+1:num_parts*months+i*months, :) = ...
        [zeros(months, (i-1)*months), repmat(workhours(:, i)', months, 1), ...
        zeros(months, (num_machines-i)*months)];
    b(num_parts*months+(i-1)*months+1:num_parts*months+i*months) = ...
        repmat([0; 0; 0; 0; 0; 0], months, 1);
end

% 线性不等式约束上下界
lb = zeros(num_variables, 1);
ub = [repmat(sales_limit(:), months, 1); repmat([100; 100; 100; 100; 100], months, 1)];

% 求解线性规划问题
options = optimoptions('linprog', 'Algorithm', 'interior-point', 'Display', 'off');
[x, fval, exitflag] = linprog(f, A, b, [], [], lb, ub, options);

% 提取结果
production_plan = reshape(x(1:num_parts*months), num_parts, months);
inventory = sum(production_plan, 2) - sales_limit(:, 1);
profit_total = -fval;

% 显示结果
disp('各种零件的生产计划：');
disp(production_plan);
disp('各种零件的库存量：');
disp(inventory);
disp('总利润：');
disp(profit_total);
