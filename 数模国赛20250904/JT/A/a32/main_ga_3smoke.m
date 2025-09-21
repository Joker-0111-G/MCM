% --- 1. 初始化和预计算 ---
% (与之前的方案二类似，初始化全局变量，预计算导弹轨迹和目标采样点)
V_drone_max = 150; % 设定无人机最大速度
global drone_max_speed;
drone_max_speed = V_drone_max;

% --- 2. 设置遗传算法 ---
nvars = 12; % 12个决策变量

% 变量边界 [t_deploy1, Px1, Py1, t_delay1,  t_deploy2, Px2, Py2, t_delay2, ...]
lb = [1,  -5000, -5000, 1,   2, -5000, -5000, 1,   3, -5000, -5000, 1];
ub = [10, 5000,  5000, 8,   20, 5000,  5000, 8,   30, 5000,  5000, 8];
% 注意：边界(lb, ub)需要根据实际情况合理设置，以缩小搜索空间

% 定义目标函数
objectiveFunc = @objectiveFunction_3smoke;
% 定义非线性约束函数
constraintFunc = @constraintFunction_3smoke;

% 设置GA选项
options = optimoptions('ga', 'Display', 'iter', 'UseParallel', true, ...
                       'NonlinearConstraintAlgorithm', 'auglag', ...
                       'PopulationSize', 200, 'MaxGenerations', 100);

% --- 3. 运行GA ---
[best_params, max_obscured_time_neg] = ga(objectiveFunc, nvars, [], [], [], [], lb, ub, constraintFunc, options);

% --- 4. 显示结果 ---
% (根据 best_params 解析并展示所有结果)