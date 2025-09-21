%% 三维空间导弹-烟雾遮蔽模拟 - 遗传算法优化版 (方案二)
%
% ***********************************
% 描述:
% 本脚本使用遗传算法 (Genetic Algorithm, GA) 来优化无人机(FY1)的干扰方案。
% 目标是找到一组最优参数（无人机速度, 投放时间, 偏航角, 延迟起爆时间），
% 使得导弹视线被烟雾遮蔽真实目标的时间最长。
%
% 相比于原版的暴力搜索，遗传算法能更高效地在广阔的搜索空间中找到
% 质量较高的解，尤其适合处理复杂的非线性优化问题。
% ***********************************

% ***********************************
% 1. 参数定义和初始化
% ***********************************
clear; clc;
tic; % 开始计时

% 使用全局变量，以便在函数中访问
global M0 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt g A0 v_a;

% 初始坐标和速度
M0 = [20000, 0, 2000];   % 导弹初始坐标 (x, y, z)
v_m = 300;               % 导弹速度 (m/s)
A0 = [17800, 0, 1800];   % 无人机初始坐标 (x, y, z)
v_a = 120;               % 无人机投放前速度 (m/s), 假设值
O = [0, 0, 0];           % 原点坐标

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
t_end = 40;              % 模拟总时长 (s)
dt = 0.01;               % 时间步长 (s)
t_vec_full = t_start:dt:t_end;

% ***********************************
% 1.5. 预计算与优化 (与原版相同)
% ***********************************
fprintf('进行预计算以加速优化过程...\n');

% 1. 预计算导弹在每个时间步的位置
missile_traj = getMissilePos_vec(t_vec_full, M0, v_m, O);

% 2. 预计算目标的采样点 (增加采样点以提高精度)
num_samples = 100;
theta_vec = linspace(0, 2*pi, num_samples);
target_points_bottom = [T_c(1) + r_t * cos(theta_vec); ...
                        T_c(2) + r_t * sin(theta_vec); ...
                        repmat(T_c(3) - h_t/2, 1, num_samples)];
target_points_top = [T_c(1) + r_t * sin(theta_vec); ...
                     T_c(2) + r_t * cos(theta_vec); ...
                     repmat(T_c(3) + h_t/2, 1, num_samples)];
all_target_points = [target_points_bottom, target_points_top];


% ***********************************
% 2. 第二问：基于遗传算法的干扰优化
% ***********************************
fprintf('开始进行FY1无人机干扰方案优化 (遗传算法)...\n');

% --- 遗传算法参数设置 ---
population_size = 600;       % 种群大小 (个体数量)
num_generations = 250;      % 迭代代数
crossover_rate = 0.8;       % 交叉概率
mutation_rate = 0.16;        % 变异概率
elitism_count = 2;          % 精英个体数量 (直接复制到下一代)
tournament_size = 3;        % 锦标赛选择的规模

% 定义变量的搜索边界 (基因的取值范围)
% [v_fy1, t_deploy, theta_yaw, t_delay]
bounds.lower = [70, 0, 0, 0];
bounds.upper = [140, 20, 2*pi, 8];

% --- 初始化种群 ---
% 每一行代表一个体，每一列代表一个基因 (待优化参数)
population = zeros(population_size, 4);
for i = 1:4
    population(:, i) = bounds.lower(i) + (bounds.upper(i) - bounds.lower(i)) * rand(population_size, 1);
end

% 初始化最优解记录
best_fitness_history = zeros(num_generations, 1);
global_best_fitness = -1;
global_best_individual = zeros(1, 4);

% --- 遗传算法主循环 ---
for gen = 1:num_generations
    % 1. 适应度评估 (计算每个个体的遮蔽时间)
    fitness = zeros(population_size, 1);
    dir_vec_initial = (O - A0) / norm(O - A0); % 这个向量是固定的
    
    for i = 1:population_size
        individual = population(i, :);
        v_fy1 = individual(1);
        t_deploy = individual(2);
        theta_yaw = individual(3);
        t_delay = individual(4);
        
        A_pos_at_t_deploy = A0 + v_a * t_deploy * dir_vec_initial;
        deploy_v_vec = [v_fy1 * cos(theta_yaw), v_fy1 * sin(theta_yaw), 0];
        
        fitness(i) = calculateObscuredTime_optimized(t_deploy, A_pos_at_t_deploy, deploy_v_vec, t_delay, ...
                                                     missile_traj, all_target_points, t_vec_full);
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
    
    % 2.1 精英保留 (Elitism)
    [~, sorted_indices] = sort(fitness, 'descend');
    new_population(1:elitism_count, :) = population(sorted_indices(1:elitism_count), :);
    
    % 2.2 交叉与变异
    for i = (elitism_count + 1):2:population_size
        % a. 选择 (锦标赛选择法)
        parent1_idx = tournament_selection(fitness, tournament_size);
        parent2_idx = tournament_selection(fitness, tournament_size);
        parent1 = population(parent1_idx, :);
        parent2 = population(parent2_idx, :);
        
        % b. 交叉 (算术交叉)
        child1 = parent1;
        child2 = parent2;
        if rand < crossover_rate
            alpha = rand;
            child1 = alpha * parent1 + (1 - alpha) * parent2;
            child2 = alpha * parent2 + (1 - alpha) * parent1;
        end
        
        % c. 变异 (高斯变异)
        child1 = mutate(child1, mutation_rate, bounds);
        child2 = mutate(child2, mutation_rate, bounds);

        new_population(i, :) = child1;
        if i+1 <= population_size
           new_population(i+1, :) = child2;
        end
    end
    
    population = new_population;
end

% --- 输出最终优化结果 ---
fprintf('\n遗传算法优化完成！\n');

% 从最优个体中提取参数
best_v = global_best_individual(1);
best_t_deploy = global_best_individual(2);
best_theta = global_best_individual(3);
best_t_delay = global_best_individual(4);
max_obscured_time = global_best_fitness;

if max_obscured_time > 0
    best_dir_vec_initial = (O - A0) / norm(O - A0);
    best_deploy_point = A0 + v_a * best_t_deploy * best_dir_vec_initial;
    best_deploy_v_vec = [best_v * cos(best_theta), best_v * sin(best_theta), 0];
    
    % 修正起爆点计算物理公式
    best_exp_point = best_deploy_point + best_deploy_v_vec * best_t_delay + [0, 0, -0.5 * g * best_t_delay^2];

    fprintf('最优无人机速度: %.2f m/s\n', best_v);
    fprintf('最优投放时间: %.2f s\n', best_t_deploy);
    fprintf('最优延迟起爆时间: %.2f s\n', best_t_delay);
    fprintf('最优方向 (偏航角): %.2f rad (%.2f 度)\n', best_theta, rad2deg(best_theta));
    fprintf('最大遮蔽时长: %.2f s\n', max_obscured_time);
    fprintf('最优投放点: (%.2f, %.2f, %.2f)\n', best_deploy_point);
    fprintf('最优起爆点: (%.2f, %.2f, %.2f)\n', best_exp_point);
else
    fprintf('在搜索范围内未找到有效的遮蔽方案。\n');
end
toc; % 结束计时

% 绘制进化过程图
figure;
plot(1:num_generations, best_fitness_history, 'b-', 'LineWidth', 2);
title('遗传算法进化过程');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大遮蔽时长 s)');
grid on;

filename = 'a2.png'; % 定义图片文件名
saveas(gcf, filename); % gcf 获取当前图形句柄并保存

% ***********************************
% 3. 局部函数定义
% ***********************************
% 辅助函数：计算特定方案的遮蔽时长 (优化版) - (与原版相同)
function obscured_time = calculateObscuredTime_optimized(t_deploy, deploy_pos, deploy_v_vec, t_delay, missile_traj, all_target_points, t_vec_full)
    global v_down r_s t_smoke_effective dt g;

    t_exp = t_deploy + t_delay;
    t_sim_end = t_exp + t_smoke_effective;

    start_idx = floor(t_exp / dt) + 1;
    end_idx = min(floor(t_sim_end / dt) + 1, length(t_vec_full));
    
    if start_idx > end_idx
        obscured_time = 0;
        return;
    end
    
    obscured_steps = 0;
    smoke_exp_pos = deploy_pos + deploy_v_vec*t_delay + [0, 0, -0.5*g*t_delay^2];
    
    for i = start_idx:end_idx
        t = t_vec_full(i);
        missile_pos = missile_traj(i, :);
        delta_t_exp = t - t_exp;
        smoke_pos = smoke_exp_pos + [0, 0, -v_down * delta_t_exp];
        
        if checkRaySphereIntersection_vec(missile_pos, all_target_points, smoke_pos, r_s)
            obscured_steps = obscured_steps + 1;
        end
    end
    
    obscured_time = obscured_steps * dt;
end

% 向量化的导弹位置计算函数 - (与原版相同)
function pos_matrix = getMissilePos_vec(t_vec, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos_matrix = M0 + t_vec' * (v_m * dir_vec);
end

% 向量化的射线与球体相交判断函数 - (与原版相同)
function is_fully_obscured = checkRaySphereIntersection_vec(P, Q_matrix, C, r)
    vec_ray = Q_matrix' - P;
    vec_PC = C - P;
    t = dot(vec_ray, repmat(vec_PC, size(vec_ray, 1), 1), 2) ./ dot(vec_ray, vec_ray, 2);
    valid_t_indices = t > 0;
    if ~any(valid_t_indices)
        is_fully_obscured = false;
        return;
    end
    dist_sq = norm(vec_PC)^2 - (t.^2) .* dot(vec_ray, vec_ray, 2);
    intersect = (dist_sq <= r^2);
    is_fully_obscured = all(intersect(valid_t_indices));
end


% --- 新增的遗传算法辅助函数 ---

% 锦标赛选择函数
function selected_index = tournament_selection(fitness, tournament_size)
    population_size = length(fitness);
    % 随机选择 tournament_size 个体
    indices = randi(population_size, 1, tournament_size);
    tournament_fitness = fitness(indices);
    % 选择其中适应度最高的个体
    [~, best_idx_in_tournament] = max(tournament_fitness);
    selected_index = indices(best_idx_in_tournament);
end

% 变异函数
function individual = mutate(individual, mutation_rate, bounds)
    for i = 1:length(individual)
        if rand < mutation_rate
            % 采用高斯变异，扰动范围为该基因范围的10%
            range = bounds.upper(i) - bounds.lower(i);
            mutation_value = (range * 0.1) * randn; % randn是标准正态分布
            individual(i) = individual(i) + mutation_value;
            
            % 确保变异后的基因仍在边界内
            individual(i) = max(individual(i), bounds.lower(i));
            individual(i) = min(individual(i), bounds.upper(i));
        end
    end
end