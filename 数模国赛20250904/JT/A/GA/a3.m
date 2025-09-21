%% 三维空间导弹-烟雾遮蔽模拟 - 三枚干扰弹遗传算法优化版 (v2 - 增加单弹贡献分析)
%
% ***********************************
% 描述:
% 本脚本使用遗传算法 (Genetic Algorithm, GA) 来优化三枚烟幕弹的组合干扰方案。
% 目标是找到一组包含12个参数的最优组合，使得导弹视线被烟雾遮蔽的
% 总时长最长。
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
t_end = 60;              % 模拟总时长 (s)
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
% 2. 基于遗传算法的三枚烟幕弹干扰优化
% ***********************************
fprintf('开始进行三枚烟幕弹干扰方案优化 (遗传算法)...\n');

% --- 遗传算法参数设置 ---
population_size = 600;      % 种群大小 (个体数量)
num_generations = 350;      % 迭代代数
num_genes = 12;             % 基因数量 (12个待优化参数)
crossover_rate = 0.86;       % 交叉概率
mutation_rate = 0.17;       % 变异概率
elitism_count = 2;          % 精英个体数量
tournament_size = 3;        % 锦标赛选择的规模

% --- 定义变量的搜索边界 (基因的取值范围) ---
%                投放时间  速度  角度  引爆时间   投放时间  速度  角度  引爆时间  投放时间  速度  角度  引爆时间
% 染色体编码: [t_deploy1, v1, theta1, t_delay1, delta_t2, v2, theta2, t_delay2, delta_t3, v3, theta3, t_delay3]
bounds.lower = [ 0, 70,    0,   0,    1,  70,    0,  0,    2, 70,    0,  0];
bounds.upper = [10, 140, 2*pi,  8,   25, 140, 2*pi,  8,   35, 140, 2*pi, 8];

% --- 初始化种群 ---
population = zeros(population_size, num_genes);
for i = 1:num_genes
    population(:, i) = bounds.lower(i) + (bounds.upper(i) - bounds.lower(i)) * rand(population_size, 1);
end

% 初始化最优解记录
best_fitness_history = zeros(num_generations, 1);
global_best_fitness = -1;
global_best_individual = zeros(1, num_genes);
dir_vec_initial = (O - A0) / norm(O - A0); % 初始方向向量，固定值



% --- 遗传算法主循环 ---
for gen = 1:num_generations
    % 1. 适应度评估
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        % 解码染色体
        ind = population(i, :);
        t_deploy1 = ind(1);  v_fy1 = ind(2); theta_yaw1 = ind(3); t_delay1 = ind(4);
        delta_t2  = ind(5);  v_fy2 = ind(6); theta_yaw2 = ind(7); t_delay2 = ind(8);
        delta_t3  = ind(9);  v_fy3 = ind(10);theta_yaw3 = ind(11);t_delay3 = ind(12);

        % 根据解码后的基因计算物理参数
        t_deploy2 = t_deploy1 + 1 + delta_t2;
        t_deploy3 = t_deploy2 + 1 + delta_t3;
        
        if t_deploy3 > t_end % 如果最终投放时间超出模拟范围，适应度为0
            fitness(i) = 0;
            continue;
        end
        
        A_pos_at_t_deploy1 = A0 + v_a * t_deploy1 * dir_vec_initial;
        deploy_v_vec1 = [v_fy1 * cos(theta_yaw1), v_fy1 * sin(theta_yaw1), 0];
        A_pos_at_t_deploy2 = A_pos_at_t_deploy1 + deploy_v_vec1 * (t_deploy2 - t_deploy1);
        deploy_v_vec2 = [v_fy2 * cos(theta_yaw2), v_fy2 * sin(theta_yaw2), 0];
        A_pos_at_t_deploy3 = A_pos_at_t_deploy2 + deploy_v_vec2 * (t_deploy3 - t_deploy2);
        deploy_v_vec3 = [v_fy3 * cos(theta_yaw3), v_fy3 * sin(theta_yaw3), 0];

        params.grenade1 = struct('t_deploy', t_deploy1, 'deploy_pos', A_pos_at_t_deploy1, 'deploy_v_vec', deploy_v_vec1, 't_delay', t_delay1, 'v_fy', v_fy1, 'theta_yaw', theta_yaw1);
        params.grenade2 = struct('t_deploy', t_deploy2, 'deploy_pos', A_pos_at_t_deploy2, 'deploy_v_vec', deploy_v_vec2, 't_delay', t_delay2, 'v_fy', v_fy2, 'theta_yaw', theta_yaw2);
        params.grenade3 = struct('t_deploy', t_deploy3, 'deploy_pos', A_pos_at_t_deploy3, 'deploy_v_vec', deploy_v_vec3, 't_delay', t_delay3, 'v_fy', v_fy3, 'theta_yaw', theta_yaw3);
        
        fitness(i) = calculateObscuredTime_three_grenades(params, missile_traj, all_target_points, t_vec_full);
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
params_final.grenade1.t_deploy = best_ind(1);  params_final.grenade1.v_fy = best_ind(2); params_final.grenade1.theta_yaw = best_ind(3); params_final.grenade1.t_delay = best_ind(4);
delta_t2                      = best_ind(5);  params_final.grenade2.v_fy = best_ind(6); params_final.grenade2.theta_yaw = best_ind(7); params_final.grenade2.t_delay = best_ind(8);
delta_t3                      = best_ind(9);  params_final.grenade3.v_fy = best_ind(10);params_final.grenade3.theta_yaw = best_ind(11);params_final.grenade3.t_delay = best_ind(12);

params_final.grenade2.t_deploy = params_final.grenade1.t_deploy + 1 + delta_t2;
params_final.grenade3.t_deploy = params_final.grenade2.t_deploy + 1 + delta_t3;

% 重新计算最优方案的投放点和爆炸点
g1 = params_final.grenade1;
g2 = params_final.grenade2;
g3 = params_final.grenade3;

g1.deploy_pos = A0 + v_a * g1.t_deploy * dir_vec_initial;
g1.deploy_v_vec = [g1.v_fy * cos(g1.theta_yaw), g1.v_fy * sin(g1.theta_yaw), 0];
g2.deploy_pos = g1.deploy_pos + g1.deploy_v_vec * (g2.t_deploy - g1.t_deploy);
g2.deploy_v_vec = [g2.v_fy * cos(g2.theta_yaw), g2.v_fy * sin(g2.theta_yaw), 0];
g3.deploy_pos = g2.deploy_pos + g2.deploy_v_vec * (g3.t_deploy - g2.t_deploy);
g3.deploy_v_vec = [g3.v_fy * cos(g3.theta_yaw), g3.v_fy * sin(g3.theta_yaw), 0];

exp_point1 = g1.deploy_pos + g1.deploy_v_vec * g1.t_delay + [0, 0, -0.5 * g * g1.t_delay^2];
exp_point2 = g2.deploy_pos + g2.deploy_v_vec * g2.t_delay + [0, 0, -0.5 * g * g2.t_delay^2];
exp_point3 = g3.deploy_pos + g3.deploy_v_vec * g3.t_delay + [0, 0, -0.5 * g * g3.t_delay^2];

% *** 新增：计算每枚烟雾弹的单独贡献 ***
fprintf('正在计算各烟雾弹的单独贡献...\n');
obscured_time_1 = calculateIndividualObscuredTime(g1, missile_traj, all_target_points, t_vec_full);
obscured_time_2 = calculateIndividualObscuredTime(g2, missile_traj, all_target_points, t_vec_full);
obscured_time_3 = calculateIndividualObscuredTime(g3, missile_traj, all_target_points, t_vec_full);
fprintf('计算完成。\n\n');

fprintf('最大总遮蔽时长: %.2f s\n\n', global_best_fitness);
fprintf('--- 烟雾弹 1 ---\n');
fprintf('投放时间: %.2f s\n', g1.t_deploy);
fprintf('投放后无人机速度: %.2f m/s\n', g1.v_fy);
fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', g1.theta_yaw, rad2deg(g1.theta_yaw));
fprintf('延迟起爆时间: %.2f s\n', g1.t_delay);
fprintf('投放点: (%.2f, %.2f, %.2f)\n', g1.deploy_pos);
fprintf('起爆点: (%.2f, %.2f, %.2f)\n', exp_point1);
fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', obscured_time_1);

fprintf('--- 烟雾弹 2 ---\n');
fprintf('投放时间: %.2f s\n', g2.t_deploy);
fprintf('投放后无人机速度: %.2f m/s\n', g2.v_fy);
fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', g2.theta_yaw, rad2deg(g2.theta_yaw));
fprintf('延迟起爆时间: %.2f s\n', g2.t_delay);
fprintf('投放点: (%.2f, %.2f, %.2f)\n', g2.deploy_pos);
fprintf('起爆点: (%.2f, %.2f, %.2f)\n', exp_point2);
fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', obscured_time_2);

fprintf('--- 烟雾弹 3 ---\n');
fprintf('投放时间: %.2f s\n', g3.t_deploy);
fprintf('投放后无人机速度: %.2f m/s\n', g3.v_fy);
fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', g3.theta_yaw, rad2deg(g3.theta_yaw));
fprintf('延迟起爆时间: %.2f s\n', g3.t_delay);
fprintf('投放点: (%.2f, %.2f, %.2f)\n', g3.deploy_pos);
fprintf('起爆点: (%.2f, %.2f, %.2f)\n', exp_point3);
fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', obscured_time_3);

toc;

% 绘制进化过程图
figure;
plot(1:num_generations, best_fitness_history, 'b-', 'LineWidth', 2);
title('遗传算法进化过程 (三枚烟幕弹)');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大遮蔽时长 s)');
grid on;

filename = 'a3.png'; % 定义图片文件名
saveas(gcf, filename); % gcf 获取当前图形句柄并保存

% ***********************************
% 3. 局部函数定义
% ***********************************

% 计算三枚烟幕弹协同作用下的总遮蔽时间
function obscured_time = calculateObscuredTime_three_grenades(params, missile_traj, all_target_points, t_vec_full)
    global v_down r_s t_smoke_effective dt g;
    g1 = params.grenade1; g2 = params.grenade2; g3 = params.grenade3;
    t_exp1 = g1.t_deploy + g1.t_delay; t_exp2 = g2.t_deploy + g2.t_delay; t_exp3 = g3.t_deploy + g3.t_delay;
    smoke_exp_pos1 = g1.deploy_pos + g1.deploy_v_vec*g1.t_delay + [0, 0, -0.5*g*g1.t_delay^2];
    smoke_exp_pos2 = g2.deploy_pos + g2.deploy_v_vec*g2.t_delay + [0, 0, -0.5*g*g2.t_delay^2];
    smoke_exp_pos3 = g3.deploy_pos + g3.deploy_v_vec*g3.t_delay + [0, 0, -0.5*g*g3.t_delay^2];
    t_sim_start1 = t_exp1; t_sim_end1 = t_exp1 + t_smoke_effective;
    t_sim_start2 = t_exp2; t_sim_end2 = t_exp2 + t_smoke_effective;
    t_sim_start3 = t_exp3; t_sim_end3 = t_exp3 + t_smoke_effective;
    min_start_time = min([t_sim_start1, t_sim_start2, t_sim_start3]);
    max_end_time = max([t_sim_end1, t_sim_end2, t_sim_end3]);
    start_idx = floor(min_start_time / dt) + 1;
    end_idx = min(floor(max_end_time / dt) + 1, length(t_vec_full));
    if start_idx > end_idx; obscured_time = 0; return; end
    obscured_steps = 0;
    for i = start_idx:end_idx
        t = t_vec_full(i); missile_pos = missile_traj(i, :);
        active_smoke_centers = [];
        if t >= t_sim_start1 && t <= t_sim_end1
            active_smoke_centers = [active_smoke_centers; smoke_exp_pos1 + [0, 0, -v_down * (t - t_exp1)]];
        end
        if t >= t_sim_start2 && t <= t_sim_end2
            active_smoke_centers = [active_smoke_centers; smoke_exp_pos2 + [0, 0, -v_down * (t - t_exp2)]];
        end
        if t >= t_sim_start3 && t <= t_sim_end3
            active_smoke_centers = [active_smoke_centers; smoke_exp_pos3 + [0, 0, -v_down * (t - t_exp3)]];
        end
        if isempty(active_smoke_centers); continue; end
        if isTargetFullyObscured(missile_pos, all_target_points, active_smoke_centers, r_s)
            obscured_steps = obscured_steps + 1;
        end
    end
    obscured_time = obscured_steps * dt;
end

% *** 新增函数：计算单枚烟幕弹的独立遮蔽时间 ***
function obscured_time = calculateIndividualObscuredTime(grenade_params, missile_traj, all_target_points, t_vec_full)
    global v_down r_s t_smoke_effective dt g;
    
    g1 = grenade_params;
    
    % 计算爆炸时间和位置
    t_exp1 = g1.t_deploy + g1.t_delay;
    smoke_exp_pos1 = g1.deploy_pos + g1.deploy_v_vec*g1.t_delay + [0, 0, -0.5*g*g1.t_delay^2];
    
    % 确定此烟雾弹的有效时间窗口
    t_sim_start1 = t_exp1;
    t_sim_end1 = t_exp1 + t_smoke_effective;
    
    start_idx = floor(t_sim_start1 / dt) + 1;
    end_idx = min(floor(t_sim_end1 / dt) + 1, length(t_vec_full));
    
    if start_idx > end_idx; obscured_time = 0; return; end
    
    obscured_steps = 0;
    for i = start_idx:end_idx
        t = t_vec_full(i);
        missile_pos = missile_traj(i, :);
        
        % 计算当前时刻的烟雾中心
        current_smoke_center = smoke_exp_pos1 + [0, 0, -v_down * (t - t_exp1)];
        
        % 检查这一个烟雾弹是否完全遮蔽目标
        if isTargetFullyObscured(missile_pos, all_target_points, current_smoke_center, r_s)
            obscured_steps = obscured_steps + 1;
        end
    end
    obscured_time = obscured_steps * dt;
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