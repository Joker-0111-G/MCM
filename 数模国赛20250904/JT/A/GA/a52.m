%% 三维空间多导弹-多无人机协同烟雾遮蔽模拟 (最终版遗传算法 - AND逻辑修正)
%
% ***********************************
% 描述:
% 最终综合问题：3枚来袭导弹 vs 5架无人机。
%
% 核心目标 (2025-09-07 修正):
% 最大化目标被【所有三枚导弹同时】遮蔽的总时长 (逻辑 AND)。
% 遗传算法将寻找一个能为目标提供最长“绝对安全”时间的协同策略。
% ***********************************

% ***********************************
% 1. 参数定义和初始化
% ***********************************
clear; clc;
tic; % 开始计时

% 使用全局变量
global M0_1 M0_2 M0_3 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt g;
global A0_1 A0_2 A0_3 A0_4 A0_5;

% --- 导弹初始信息 (3枚) ---
M0_1 = [20000, 0, 2000];     % M1 初始坐标
M0_2 = [19000, 600, 2100];    % M2 初始坐标
M0_3 = [18000, -600, 1900];   % M3 初始坐标
v_m = 300;                   % 导弹速度 (m/s)
O = [0, 0, 0];               % 目标原点

% --- 无人机初始信息 (5架) ---
A0_1 = [17800, 0, 1800];      % FY1 初始坐标
A0_2 = [12000, 1400, 1400];   % FY2 初始坐标
A0_3 = [6000, -3000, 700];    % FY3 初始坐标
A0_4 = [11000, 2000, 1800];   % FY4 初始坐标
A0_5 = [13000, -2000, 1300];  % FY5 初始坐标

% 真实目标参数 (圆柱体)
T_c = [0, 200, 0];           % 圆柱体中心
r_t = 7;                     % 圆柱体半径 (m)
h_t = 10;                    % 圆柱体高度 (m)

% 干扰弹和烟雾参数
v_down = 3;                  % 烟雾云团下沉速度 (m/s)
r_s = 10;                    % 烟雾有效遮蔽半径 (m)
t_smoke_effective = 20;      % 烟雾有效遮蔽时长 (s)
g = 9.8;                     % 重力加速度 (m/s^2)

% 模拟时间参数
t_start = 0;
t_end = 90;                  % 模拟总时长 (s)
dt = 0.01;                   % 时间步长 (s)
t_vec_full = t_start:dt:t_end;

% ***********************************
% 1.5. 预计算与优化
% ***********************************
fprintf('进行预计算以加速优化过程...\n');
% 预计算三枚导弹的轨迹
missile_traj_1 = getMissilePos_vec(t_vec_full, M0_1, v_m, O);
missile_traj_2 = getMissilePos_vec(t_vec_full, M0_2, v_m, O);
missile_traj_3 = getMissilePos_vec(t_vec_full, M0_3, v_m, O);
all_missile_trajs = {missile_traj_1, missile_traj_2, missile_traj_3};

% 预计算目标采样点
num_samples = 50;
theta_vec = linspace(0, 2*pi, num_samples);
target_points_bottom = [T_c(1) + r_t * cos(theta_vec); T_c(2) + r_t * sin(theta_vec); repmat(T_c(3) - h_t/2, 1, num_samples)];
target_points_top = [T_c(1) + r_t * sin(theta_vec); T_c(2) + r_t * cos(theta_vec); repmat(T_c(3) + h_t/2, 1, num_samples)];
all_target_points = [target_points_bottom, target_points_top];

% ***********************************
% 2. 基于遗传算法的多对多协同干扰优化
% ***********************************
fprintf('开始进行三导弹-五无人机协同干扰方案优化 (遗传算法)...\n');
fprintf('目标：最大化对三枚导弹的同时遮蔽时间 (AND逻辑)。\n');

% --- 遗传算法参数设置 ---
population_size = 1000;      % [关键] 增加种群以应对高维度
num_generations = 400;      % [关键] 增加代数以保证收敛
num_genes = 60;             % 基因数量 (5架无人机 x 12个参数)
crossover_rate = 0.85;       % 交叉概率
mutation_rate = 0.25;       % 变异概率
elitism_count = 2;          % 精英个体数量
tournament_size = 3;        % 锦标赛选择的规模

% --- 定义变量的搜索边界 (基因的取值范围) ---
single_drone_bounds.lower = [ 0, 70, 0, 0,  1, 70, 0, 0,  3, 70, 0, 0];
single_drone_bounds.upper = [15, 140, 2*pi, 8, 25, 140, 2*pi, 8, 35, 140, 2*pi, 8];
bounds.lower = repmat(single_drone_bounds.lower, 1, 5);
bounds.upper = repmat(single_drone_bounds.upper, 1, 5);

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
    fitness = zeros(population_size, 1);
    
    % 注意: 此处若要加速，可将 for 循环改为 parfor (并行计算)
    for i = 1:population_size
        fitness(i) = calculateFitness_MultiMulti(population(i, :), all_missile_trajs, all_target_points, t_vec_full, t_end);
    end
    
    [max_fitness_current_gen, idx] = max(fitness);
    if max_fitness_current_gen > global_best_fitness
        global_best_fitness = max_fitness_current_gen;
        global_best_individual = population(idx, :);
    end
    best_fitness_history(gen) = global_best_fitness;
    fprintf('第 %d/%d 代: 当前最优遮蔽时间 = %.4f s, 全局最优 = %.4f s\n', gen, num_generations, max_fitness_current_gen, global_best_fitness);

    % 生成下一代
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
if global_best_fitness > 0
    analyzeAndPrintBestSolution(global_best_individual, all_missile_trajs, all_target_points, t_vec_full, global_best_fitness);
else
    fprintf('在搜索范围内未找到有效的遮蔽方案。\n');
end
toc;

% 绘制进化过程图
figure;
plot(1:num_generations, best_fitness_history, 'b-', 'LineWidth', 2);
title('遗传算法进化过程 (三导弹-五无人机)');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大同时遮蔽时长 s)');
grid on;

filename = 'a5_corrected_AND.png'; % 定义图片文件名
saveas(gcf, filename); % gcf 获取当前图形句柄并保存


% *************************************************************************
%                          --- 辅助函数定义 ---
% *************************************************************************

% 适应度评估函数
function fitness = calculateFitness_MultiMulti(individual, all_missile_trajs, all_target_points, t_vec_full, t_end)
    global r_s dt;
    
    all_grenade_params = decodeIndividual(individual, t_end);
    if isempty(all_grenade_params)
        fitness = 0;
        return;
    end
    
    [active_times, smoke_exp_pos] = get_smoke_info(all_grenade_params);
    
    total_obscured_steps = 0;
    
    % 主循环，遍历时间
    for i = 1:length(t_vec_full)
        t = t_vec_full(i);
        
        % 获取当前所有有效烟雾中心
        active_smoke_centers = get_active_smoke_centers(t, active_times, smoke_exp_pos);
        if isempty(active_smoke_centers)
            continue;
        end
        
        % 检查对三枚导弹的遮蔽情况
        is_m1_obscured = isTargetFullyObscured(all_missile_trajs{1}(i,:), all_target_points, active_smoke_centers, r_s);
        is_m2_obscured = isTargetFullyObscured(all_missile_trajs{2}(i,:), all_target_points, active_smoke_centers, r_s);
        is_m3_obscured = isTargetFullyObscured(all_missile_trajs{3}(i,:), all_target_points, active_smoke_centers, r_s);
        
        % =================================================================
        % ======================= 核心逻辑修正 ==========================
        % =================================================================
        % 目标：最大化【同时】遮蔽所有导弹的时间
        % 因此，必须使用逻辑 "与" (AND, &&)
        if is_m1_obscured && is_m2_obscured && is_m3_obscured
            total_obscured_steps = total_obscured_steps + 1;
        end
    end
    
    fitness = total_obscured_steps * dt;
end

% (其他所有辅助函数与源文件 a5.m 相同，此处不再重复)
% ... a
% ... 
% ...
function analyzeAndPrintBestSolution(individual, all_missile_trajs, all_target_points, t_vec_full, total_fitness)
    global r_s dt;

    fprintf('正在进行最终方案的详细分析...\n');
    
    all_grenade_params = decodeIndividual(individual, t_vec_full(end));
    [active_times, smoke_exp_pos] = get_smoke_info(all_grenade_params);
    
    num_grenades = length(all_grenade_params);
    grenade_contribution = zeros(num_grenades, 3); % 每枚弹对M1, M2, M3的贡献

    % 分析循环
    for t_idx = 1:length(t_vec_full)
        t = t_vec_full(t_idx);
        missile_pos = [all_missile_trajs{1}(t_idx,:); all_missile_trajs{2}(t_idx,:); all_missile_trajs{3}(t_idx,:)];
        
        for g_idx = 1:num_grenades
            if t >= active_times(g_idx, 1) && t <= active_times(g_idx, 2)
                global v_down;
                delta_t = t - active_times(g_idx, 1);
                current_smoke_center = smoke_exp_pos(g_idx, :) + [0, 0, -v_down * delta_t];
                
                % 检查这枚弹对每个导弹的遮蔽情况
                if isTargetFullyObscured(missile_pos(1,:), all_target_points, current_smoke_center, r_s)
                    grenade_contribution(g_idx, 1) = grenade_contribution(g_idx, 1) + dt;
                end
                if isTargetFullyObscured(missile_pos(2,:), all_target_points, current_smoke_center, r_s)
                    grenade_contribution(g_idx, 2) = grenade_contribution(g_idx, 2) + dt;
                end
                if isTargetFullyObscured(missile_pos(3,:), all_target_points, current_smoke_center, r_s)
                    grenade_contribution(g_idx, 3) = grenade_contribution(g_idx, 3) + dt;
                end
            end
        end
    end
    
    fprintf('分析完成。\n\n');
    fprintf('======================================================\n');
    fprintf('            最优协同干扰方案详细结果\n');
    fprintf('======================================================\n');
    fprintf('最大总遮蔽时长 (对所有导弹同时有效): %.2f s\n\n', total_fitness);
    
    for i = 1:num_grenades
        p = all_grenade_params{i};
        drone_idx = floor((i-1)/3) + 1;
        grenade_in_drone_idx = mod(i-1, 3) + 1;
        
        fprintf('--- 无人机 FY%d - 烟雾弹 %d ---\n', drone_idx, grenade_in_drone_idx);
        fprintf('投放时无人机速度: %.2f m/s\n', p.v_fy);
        fprintf('投放时无人机方向: %.2f rad (%.2f 度)\n', p.theta_yaw, rad2deg(p.theta_yaw));
        fprintf('无人机飞行投放时间点: %.2f s\n', p.t_deploy);
        fprintf('烟雾弹延迟引爆时间: %.2f s\n', p.t_delay);
        fprintf('总引爆时间点: %.2f s\n', p.t_exp);
        fprintf('投放坐标: (%.2f, %.2f, %.2f)\n', p.deploy_pos);
        fprintf('引爆坐标: (%.2f, %.2f, %.2f)\n', p.exp_pos);
        fprintf('** 单独贡献分析 **:\n');
        fprintf('   - 对 M1 遮蔽时长: %.2f s\n', grenade_contribution(i, 1));
        fprintf('   - 对 M2 遮蔽时长: %.2f s\n', grenade_contribution(i, 2));
        fprintf('   - 对 M3 遮蔽时长: %.2f s\n\n', grenade_contribution(i, 3));
    end
end

function all_grenade_params = decodeIndividual(individual, t_end)
    global A0_1 A0_2 A0_3 A0_4 A0_5 g;
    all_A0 = {A0_1, A0_2, A0_3, A0_4, A0_5};
    all_grenade_params = {};
    
    for i = 1:5 % 遍历5架无人机
        drone_params = individual((i-1)*12 + 1 : i*12);
        A0 = all_A0{i};
        
        t_deploy1 = drone_params(1); v_fy1 = drone_params(2); theta_yaw1 = drone_params(3); t_delay1 = drone_params(4);
        delta_t2  = drone_params(5); v_fy2 = drone_params(6); theta_yaw2 = drone_params(7); t_delay2 = drone_params(8);
        delta_t3  = drone_params(9); v_fy3 = drone_params(10);theta_yaw3 = drone_params(11);t_delay3 = drone_params(12);
        
        t_deploy2 = t_deploy1 + 1 + delta_t2;
        t_deploy3 = t_deploy2 + 1 + delta_t3;
        
        if t_deploy3 > t_end
            all_grenade_params = {}; % 无效个体
            return;
        end
        
        deploy_pos1 = A0 + [v_fy1 * cos(theta_yaw1) * t_deploy1, v_fy1 * sin(theta_yaw1) * t_deploy1, 0];
        exp_pos1 = deploy_pos1 + [v_fy1 * cos(theta_yaw1) * t_delay1, v_fy1 * sin(theta_yaw1) * t_delay1, -0.5*g*t_delay1^2];
        all_grenade_params{end+1} = struct('v_fy', v_fy1, 'theta_yaw', theta_yaw1, 't_deploy', t_deploy1, 't_delay', t_delay1, 't_exp', t_deploy1+t_delay1, 'deploy_pos', deploy_pos1, 'exp_pos', exp_pos1);

        deploy_pos2 = deploy_pos1 + [v_fy2 * cos(theta_yaw2) * (t_deploy2-t_deploy1), v_fy2 * sin(theta_yaw2) * (t_deploy2-t_deploy1), 0];
        exp_pos2 = deploy_pos2 + [v_fy2 * cos(theta_yaw2) * t_delay2, v_fy2 * sin(theta_yaw2) * t_delay2, -0.5*g*t_delay2^2];
        all_grenade_params{end+1} = struct('v_fy', v_fy2, 'theta_yaw', theta_yaw2, 't_deploy', t_deploy2, 't_delay', t_delay2, 't_exp', t_deploy2+t_delay2, 'deploy_pos', deploy_pos2, 'exp_pos', exp_pos2);

        deploy_pos3 = deploy_pos2 + [v_fy3 * cos(theta_yaw3) * (t_deploy3-t_deploy2), v_fy3 * sin(theta_yaw3) * (t_deploy3-t_deploy2), 0];
        exp_pos3 = deploy_pos3 + [v_fy3 * cos(theta_yaw3) * t_delay3, v_fy3 * sin(theta_yaw3) * t_delay3, -0.5*g*t_delay3^2];
        all_grenade_params{end+1} = struct('v_fy', v_fy3, 'theta_yaw', theta_yaw3, 't_deploy', t_deploy3, 't_delay', t_delay3, 't_exp', t_deploy3+t_delay3, 'deploy_pos', deploy_pos3, 'exp_pos', exp_pos3);
    end
end

function [active_times, smoke_exp_pos] = get_smoke_info(all_grenade_params)
    global t_smoke_effective;
    num_grenades = length(all_grenade_params);
    active_times = zeros(num_grenades, 2);
    smoke_exp_pos = zeros(num_grenades, 3);
    for i = 1:num_grenades
        p = all_grenade_params{i};
        active_times(i, 1) = p.t_exp;
        active_times(i, 2) = p.t_exp + t_smoke_effective;
        smoke_exp_pos(i, :) = p.exp_pos;
    end
end

function active_smoke_centers = get_active_smoke_centers(t, active_times, smoke_exp_pos)
    global v_down;
    active_indices = find(t >= active_times(:,1) & t <= active_times(:,2));
    if isempty(active_indices)
        active_smoke_centers = [];
        return;
    end
    
    delta_t = t - active_times(active_indices, 1);
    active_smoke_centers = smoke_exp_pos(active_indices, :) + [zeros(length(active_indices), 2), -v_down * delta_t];
end

function is_fully_obscured = isTargetFullyObscured(missile_pos, target_points, smoke_centers, r)
    num_target_points = size(target_points, 2);
    for i = 1:num_target_points
        target_point = target_points(:, i)';
        is_this_point_obscured = false;
        for j = 1:size(smoke_centers, 1)
            if checkSingleRaySphereIntersection(missile_pos, target_point, smoke_centers(j, :), r)
                is_this_point_obscured = true;
                break;
            end
        end
        if ~is_this_point_obscured
            is_fully_obscured = false;
            return;
        end
    end
    is_fully_obscured = true;
end

function does_intersect = checkSingleRaySphereIntersection(P, Q, C, r)
    vec_ray = Q - P; vec_PC = C - P;
    t = dot(vec_PC, vec_ray) / dot(vec_ray, vec_ray);
    if t < 0 || t > 1
        dist_PC_sq = sum(vec_PC.^2); dist_QC_sq = sum((C - Q).^2);
        does_intersect = (dist_PC_sq <= r^2) || (dist_QC_sq <= r^2);
    else
        dist_sq = sum(vec_PC.^2) - (t^2) * dot(vec_ray, vec_ray);
        does_intersect = (dist_sq <= r^2);
    end
end

function pos_matrix = getMissilePos_vec(t_vec, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos_matrix = M0 + t_vec' * (v_m * dir_vec);
end

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