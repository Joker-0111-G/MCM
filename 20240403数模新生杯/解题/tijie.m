% 零件利润（元/个）
profits = [10, 9, 3, 5, 11, 9, 8];

% 机床设备工时（小时/个）
machine_times = [
    0.6, 0.7, 0,   0.3, 0.6, 0,   0.5;
    0.1, 0.1, 0,   0.3, 0,   0.6, 0;
    0.2, 0,   0.4, 0,   0.2, 0,   0.6;
    0.05,0.08, 0,   0.06,0.1, 0,   0.08;
    0,   0,   0.01,0,   0.05,0.08,0.05
];

% 每月需要整修的设备
maintenance_schedule = [
    2, 0, 0, 0, 0, 0;
    0, 1, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0;
    0, 0, 1, 0, 0, 0;
    1, 1, 0, 0, 0, 0;
    0, 0, 1, 0, 1, 0
];

% 销售上限（个/月）
sales_limits = [
    600, 800, 200, 0,   700, 300, 200;
    500, 600, 300, 300, 500, 200, 250;
    200, 500, 400, 200, 500, 0,   300;
    300, 400, 0,   400, 300, 500, 100;
    0,   200, 300, 200, 900, 200, 0;
    400, 300, 100, 300, 800, 400, 100
];

% 初始库存量
initial_inventory = zeros(1, 7);

% 每个零件的库存成本（元/个/月）
inventory_cost_per_unit = 0.5;

% 最大库存量
max_inventory = 100;

% 机床工作天数
working_days_per_month = 22;

% 解决线性规划
for month = 1:6
    f = -profits'; % 最大化利润
    Aeq = machine_times;
    beq = sales_limits(month, :)';
    A = [];
    b = [];
    lb = zeros(7, 1);
    ub = repmat(max_inventory, 7, 1) - initial_inventory';
    
    % 考虑库存成本
    f = f + inventory_cost_per_unit * ones(7, 1);
    
    % 考虑整修
    if any(maintenance_schedule(month, :))
        machine_indices = find(maintenance_schedule(month, :));
        for machine_index = machine_indices
            f = f - profits(machine_index); % 考虑整修损失
            Aeq = [Aeq; zeros(1, 7)];
            Aeq(end, machine_index) = 1;
            beq = [beq; 0];
        end
    end
    
    % 线性规划求解
    [x, fval] = linprog(f, A, b, Aeq, beq, lb, ub);
    
    % 更新库存
    initial_inventory = initial_inventory + machine_times * x * working_days_per_month;
    
    % 输出结果
    fprintf('Month %d:\n', month);
    fprintf('Produced quantities:\n');
    disp(x');
    fprintf('Ending inventory:\n');
    disp(initial_inventory');
    fprintf('Profit for this month: %.2f\n', -fval);
    fprintf('----------------------------------\n');
end
