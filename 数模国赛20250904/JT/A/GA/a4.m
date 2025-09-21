%% 三维空间导弹-烟雾遮蔽模拟 - 三架无人机协同干扰 (遗传算法完整版 v2)
%
% ***********************************
% 描述:
% 本脚本使用遗传算法 (Genetic Algorithm, GA) 来优化三架独立无人机(FY1, FY2, FY3)
% 的协同干扰方案。每架无人机有4个独立参数，总计12个优化变量。
%
% v2版新增功能：在找到最优解后，会额外计算并显示每一枚烟雾弹单独的有效
% 遮蔽时长，以分析其在协同策略中的各自贡献。
% ***********************************

% ***********************************
% 1. 参数定义和初始化
% ***********************************
clear; clc;
tic; % 开始计时

% 使用全局变量
global M0 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt g;
global A0_1 A0_2 A0_3; % 三架无人机的初始位置

% 导弹初始信息
M0 = [20000, 0, 2000];   % 导弹初始坐标 (x, y, z)
v_m = 300;               % 导弹速度 (m/s)
O = [0, 0, 0];           % 原点坐标

% 三架无人机初始信息
A0_1 = [17800, 0, 1800];      % FY1 无人机初始坐标
A0_2 = [12000, 1400, 1400];   % FY2 无人机初始坐标
A0_3 = [6000, -3000, 700];    % FY3 无人机初始坐标

% 真实目标参数 (圆柱体)
T_c = [0, 200, 0];       % 圆柱体中心
r_t = 7;                 % 圆柱体半径 (m)
h_t = 10;                % 圆柱体高度 (m)

% 干扰弹和烟雾参数
v_down = 3;              % 烟雾云团下沉速度 (m/s)
r_s = 10;                % 烟雾有效遮蔽半径 (m)
t_smoke_effective = 20;  % 烟雾有效遮蔽时长 (s)
g = 9.8;                 % 重力加速度 (m/s^2)

% 模拟时间参数
t_start = 0;
t_end = 70;              % 适当延长模拟总时长
dt = 0.01;               % 时间步长 (s)
t_vec_full = t_start:dt:t_end;

% ***********************************
% 1.5. 预计算与优化
% ***********************************
fprintf('进行预计算以加速优化过程...\n');
missile_traj = getMissilePos_vec(t_vec_full, M0, v_m, O);
num_samples = 50;
theta_vec = linspace(0, 2*pi, num_samples);
target_points_bottom = [T_c(1) + r_t * cos(theta_vec); T_c(2) + r_t * sin(theta_vec); repmat(T_c(3) - h_t/2, 1, num_samples)];
target_points_top = [T_c(1) + r_t * sin(theta_vec); T_c(2) + r_t * cos(theta_vec); repmat(T_c(3) + h_t/2, 1, num_samples)];
all_target_points = [target_points_bottom, target_points_top];


% ***********************************
% 2. 三无人机协同干扰优化 (遗传算法)
% ***********************************
fprintf('开始进行三无人机协同干扰方案优化 (遗传算法)...\n');

% --- 遗传算法参数设置 ---
population_size = 900;      % 种群大小
num_generations = 350;      % 迭代代数
num_genes = 12;             % 基因数量 (3架无人机 x 4个参数)
crossover_rate = 0.8;       % 交叉概率
mutation_rate = 0.25;        % 变异概率
elitism_count = 2;          % 精英个体数量
tournament_size = 3;        % 锦标赛选择规模

% --- 定义变量的搜索边界 ---
% 染色体编码: [v1, t_dep1, theta1, t_del1, v2, t_dep2, theta2, t_del2, v3, t_dep3, theta3, t_del3]
bounds.lower = [70, 0, 0, 0,  70, 0, 0, 0,  70, 0, 0, 0];
bounds.upper = [140, 15, 2*pi, 8, 140, 15, 2*pi, 8, 140, 15, 2*pi, 8];

% --- 初始化种群 ---
population = zeros(population_size, num_genes);
for i = 1:num_genes
    population(:, i) = bounds.lower(i) + (bounds.upper(i) - bounds.lower(i)) * rand(population_size, 1);
end

% 初始化最优解记录
best_fitness_history = zeros(num_generations, 1);
global_best_fitness = -1;
global_best_individual = zeros(1, num_genes);



% --- 遗传算法主循环 ---
for gen = 1:num_generations
    % 1. 适应度评估
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        ind = population(i, :);
        
        % 解码染色体并打包参数
        params.p1 = struct('v_fy', ind(1), 't_deploy', ind(2), 'theta_yaw', ind(3), 't_delay', ind(4), 'A0', A0_1);
        params.p2 = struct('v_fy', ind(5), 't_deploy', ind(6), 'theta_yaw', ind(7), 't_delay', ind(8), 'A0', A0_2);
        params.p3 = struct('v_fy', ind(9), 't_deploy', ind(10), 'theta_yaw', ind(11), 't_delay', ind(12), 'A0', A0_3);

        % 计算总遮蔽时间
        fitness(i) = calculateObscuredTime_3UAVs(params, missile_traj, all_target_points, t_vec_full);
    end
    
    % 更新全局最优解
    [max_fitness_current_gen, idx] = max(fitness);
    if max_fitness_current_gen > global_best_fitness
        global_best_fitness = max_fitness_current_gen;
        global_best_individual = population(idx, :);
    end
    best_fitness_history(gen) = global_best_fitness;
    fprintf('第 %d 代: 当前最优遮蔽时间 = %.4f s, 全局最优 = %.4f s\n', gen, max_fitness_current_gen, global_best_fitness);

    % 2. 生成下一代
    new_population = zeros(size(population));
    [~, sorted_indices] = sort(fitness, 'descend');
    new_population(1:elitism_count, :) = population(sorted_indices(1:elitism_count), :);
    
    for i = (elitism_count + 1):2:population_size
        parent1 = population(tournament_selection(fitness, tournament_size), :);
        parent2 = population(tournament_selection(fitness, tournament_size), :);
        
        child1 = parent1; child2 = parent2;
        if rand < crossover_rate
            alpha = rand;
            child1 = alpha * parent1 + (1 - alpha) * parent2;
            child2 = alpha * parent2 + (1 - alpha) * parent1;
        end
        
        new_population(i, :) = mutate(child1, mutation_rate, bounds);
        if i+1 <= population_size
           new_population(i+1, :) = mutate(child2, mutation_rate, bounds);
        end
    end


    population = new_population;
end


% --- 输出最终优化结果 ---
fprintf('\n遗传算法优化完成！\n');

% 解码最优个体
best_ind = global_best_individual;
best_params.p1 = struct('v_fy', best_ind(1), 't_deploy', best_ind(2), 'theta_yaw', best_ind(3), 't_delay', best_ind(4), 'A0', A0_1);
best_params.p2 = struct('v_fy', best_ind(5), 't_deploy', best_ind(6), 'theta_yaw', best_ind(7), 't_delay', best_ind(8), 'A0', A0_2);
best_params.p3 = struct('v_fy', best_ind(9), 't_deploy', best_ind(10), 'theta_yaw', best_ind(11), 't_delay', best_ind(12), 'A0', A0_3);

if global_best_fitness > 0
    fprintf('最大总遮蔽时长: %.2f s\n\n', global_best_fitness);
    
    uavs = {'FY1', 'FY2', 'FY3'};
    ps = {best_params.p1, best_params.p2, best_params.p3};
    
    for i = 1:3
        p = ps{i};
        deploy_v_vec = [p.v_fy * cos(p.theta_yaw), p.v_fy * sin(p.theta_yaw), 0];
        deploy_point = p.A0 + deploy_v_vec * p.t_deploy;
        exp_point = deploy_point + deploy_v_vec * p.t_delay + [0, 0, -0.5 * g * p.t_delay^2];
        
        % *** 新增：计算当前这枚烟雾弹的单独贡献 ***
        individual_obscured_time = calculateIndividualObscuredTime(p, missile_traj, all_target_points, t_vec_full);

        fprintf('--- %s 无人机最优参数 ---\n', uavs{i});
        fprintf('投放后飞行速度: %.2f m/s\n', p.v_fy);
        fprintf('飞行及投放时间: %.2f s\n', p.t_deploy);
        fprintf('飞行方向 (偏航角): %.2f rad (%.2f 度)\n', p.theta_yaw, rad2deg(p.theta_yaw));
        fprintf('延迟引爆时间: %.2f s\n', p.t_delay);
        fprintf('投放点: (%.2f, %.2f, %.2f)\n', deploy_point);
        fprintf('起爆点: (%.2f, %.2f, %.2f)\n', exp_point);
        fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', individual_obscured_time); % *** 新增输出 ***
    end
else
    fprintf('在搜索范围内未找到有效的遮蔽方案。\n');
end
toc;

% 绘制进化过程图
figure;
plot(1:num_generations, best_fitness_history, 'b-', 'LineWidth', 2);
title('遗传算法进化过程 (三无人机协同)');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大遮蔽时长 s)');
grid on;

filename = 'a4.png'; % 定义图片文件名
saveas(gcf, filename); % gcf 获取当前图形句柄并保存

% ***********************************
% 3. 局部函数定义
% ***********************************

% 辅助函数：计算三架无人机协同方案的遮蔽时长
function obscured_time = calculateObscuredTime_3UAVs(params, missile_traj, all_target_points, t_vec_full)
    global v_down r_s t_smoke_effective dt g;

    % 分别计算三枚烟幕弹的爆炸时间和爆炸点
    p = {params.p1, params.p2, params.p3};
    t_exp = zeros(1, 3);
    smoke_exp_pos = zeros(3, 3);
    
    for i = 1:3
        deploy_v_vec = [p{i}.v_fy * cos(p{i}.theta_yaw), p{i}.v_fy * sin(p{i}.theta_yaw), 0];
        deploy_pos = p{i}.A0 + deploy_v_vec * p{i}.t_deploy;
        
        t_exp(i) = p{i}.t_deploy + p{i}.t_delay;
        smoke_exp_pos(i, :) = deploy_pos + deploy_v_vec*p{i}.t_delay + [0, 0, -0.5*g*p{i}.t_delay^2];
    end

    % 确定整个模拟需要检查的时间范围
    min_start_time = min(t_exp);
    max_end_time = max(t_exp) + t_smoke_effective;
    start_idx = floor(min_start_time / dt) + 1;
    end_idx = min(floor(max_end_time / dt) + 1, length(t_vec_full));
    
    if start_idx > end_idx || start_idx < 1
        obscured_time = 0;
        return;
    end
    
    obscured_steps = 0;
    
    for i = start_idx:end_idx
        t = t_vec_full(i);
        missile_pos = missile_traj(i, :);
        
        % 收集当前时间点所有有效的烟雾云团
        active_smoke_centers = [];
        for j = 1:3
            if t >= t_exp(j) && t <= (t_exp(j) + t_smoke_effective)
                delta_t = t - t_exp(j);
                current_smoke_pos = smoke_exp_pos(j, :) + [0, 0, -v_down * delta_t];
                active_smoke_centers = [active_smoke_centers; current_smoke_pos];
            end
        end
        
        if isempty(active_smoke_centers)
            continue;
        end
        
        % 检查目标是否被当前活动的烟雾云团组合完全遮蔽
        if isTargetFullyObscured_MultiSphere(missile_pos, all_target_points, active_smoke_centers, r_s)
            obscured_steps = obscured_steps + 1;
        end
    end
    
    obscured_time = obscured_steps * dt;
end

% *** 新增函数：计算单枚烟幕弹的独立遮蔽时间 ***
function obscured_time = calculateIndividualObscuredTime(grenade_params, missile_traj, all_target_points, t_vec_full)
    global v_down r_s t_smoke_effective dt g;
    
    p1 = grenade_params;
    
    % 计算爆炸时间和位置
    deploy_v_vec = [p1.v_fy * cos(p1.theta_yaw), p1.v_fy * sin(p1.theta_yaw), 0];
    deploy_pos = p1.A0 + deploy_v_vec * p1.t_deploy;
    t_exp1 = p1.t_deploy + p1.t_delay;
    smoke_exp_pos1 = deploy_pos + deploy_v_vec*p1.t_delay + [0, 0, -0.5*g*p1.t_delay^2];
    
    % 确定此烟雾弹的有效时间窗口
    t_sim_start1 = t_exp1;
    t_sim_end1 = t_exp1 + t_smoke_effective;
    
    start_idx = floor(t_sim_start1 / dt) + 1;
    end_idx = min(floor(t_sim_end1 / dt) + 1, length(t_vec_full));
    
    if start_idx > end_idx || start_idx < 1; obscured_time = 0; return; end
    
    obscured_steps = 0;
    for i = start_idx:end_idx
        t = t_vec_full(i);
        missile_pos = missile_traj(i, :);
        
        % 计算当前时刻的烟雾中心
        current_smoke_center = smoke_exp_pos1 + [0, 0, -v_down * (t - t_exp1)];
        
        % 检查这一个烟雾弹是否完全遮蔽目标
        if isTargetFullyObscured_MultiSphere(missile_pos, all_target_points, current_smoke_center, r_s)
            obscured_steps = obscured_steps + 1;
        end
    end
    obscured_time = obscured_steps * dt;
end

% 新函数：检查目标是否被多个球体完全遮蔽
function is_fully_obscured = isTargetFullyObscured_MultiSphere(missile_pos, target_points, smoke_centers, r)
    num_target_points = size(target_points, 2);
    num_smoke_spheres = size(smoke_centers, 1);
    
    % 必须遮蔽所有目标采样点
    for i = 1:num_target_points
        target_point = target_points(:, i)';
        
        is_this_point_obscured = false;
        % 只要有一个烟雾弹遮挡了该点即可
        for j = 1:num_smoke_spheres
            smoke_center = smoke_centers(j, :);
            if checkSingleRaySphereIntersection(missile_pos, target_point, smoke_center, r)
                is_this_point_obscured = true;
                break; % 该目标点被遮蔽，跳出内层循环，检查下一个目标点
            end
        end
        
        % 如果我们找到了一个没有被任何烟雾遮蔽的目标点，那么目标就不是完全遮蔽的
        if ~is_this_point_obscured
            is_fully_obscured = false;
            return;
        end
    end
    
    % 如果循环完成，意味着所有目标点都被至少一个烟雾球遮蔽
    is_fully_obscured = true;
end

% 检查单条射线（线段）与单个球体是否相交
function does_intersect = checkSingleRaySphereIntersection(P, Q, C, r)
    vec_ray = Q - P;
    vec_PC = C - P;
    t = dot(vec_PC, vec_ray) / dot(vec_ray, vec_ray);
    if t < 0 || t > 1
        dist_PC_sq = sum(vec_PC.^2);
        dist_QC_sq = sum((C - Q).^2);
        does_intersect = (dist_PC_sq <= r^2) || (dist_QC_sq <= r^2);
    else
        dist_sq = sum(vec_PC.^2) - (t^2) * dot(vec_ray, vec_ray);
        does_intersect = (dist_sq <= r^2);
    end
end

% 向量化的导弹位置计算函数
function pos_matrix = getMissilePos_vec(t_vec, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos_matrix = M0 + t_vec' * (v_m * dir_vec);
end

% --- 新增的遗传算法辅助函数 ---
function selected_index = tournament_selection(fitness, tournament_size)
    population_size = length(fitness);
    indices = randi(population_size, 1, tournament_size);
    [~, best_idx_in_tournament] = max(fitness(indices));
    selected_index = indices(best_idx_in_tournament);
end

function individual = mutate(individual, mutation_rate, bounds)
    num_genes = length(individual);
    for i = 1:num_genes
        if rand < mutation_rate
            range = bounds.upper(i) - bounds.lower(i);
            mutation_value = (range * 0.1) * randn; % 高斯变异
            individual(i) = individual(i) + mutation_value;
            individual(i) = max(individual(i), bounds.lower(i));
            individual(i) = min(individual(i), bounds.upper(i));
        end
    end
end