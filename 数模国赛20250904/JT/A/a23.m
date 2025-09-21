%% 三维空间导弹-烟雾遮蔽模拟 - 修正与优化版 (方案一)

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
% 1.5. 预计算与优化
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
% 2. 第二问：无人机FY1干扰优化
% ***********************************
fprintf('开始进行FY1无人机干扰方案优化 (优化版暴力搜索)...\n');

% 定义变量搜索范围 (为快速演示，可适当减小范围)
v_fy1_range = 70:5:140;        % 无人机速度范围
t_deploy_range = 5:0.05:15;        % 投放时间范围
theta_yaw_range = 0:pi/16:2*pi;  % 偏航角范围
t_delay_range = 2:0.05:8;          % 延迟起爆时间范围

% 初始化最优解存储
max_obscured_time = -1;
best_v = 0;
best_t_deploy = 0;
best_theta = 0;
best_t_delay = 0;

% 遍历搜索空间
total_iterations = length(v_fy1_range)*length(t_deploy_range)*length(theta_yaw_range)*length(t_delay_range);
iter_count = 0;

for v_fy1 = v_fy1_range
    dir_vec_initial = (O - A0) / norm(O - A0);
    for t_deploy = t_deploy_range
        A_pos_at_t_deploy = A0 + v_a * t_deploy * dir_vec_initial;
        for theta_yaw = theta_yaw_range
            deploy_v_vec = [v_fy1 * cos(theta_yaw), v_fy1 * sin(theta_yaw), 0];
            for t_delay = t_delay_range
                iter_count = iter_count + 1;
                if mod(iter_count, 1000) == 0
                    fprintf('进度: %.2f%%\n', 100 * iter_count / total_iterations);
                end
                
                % 使用优化后的函数计算遮蔽时间
                current_obscured_time = calculateObscuredTime_optimized(t_deploy, A_pos_at_t_deploy, deploy_v_vec, t_delay, ...
                                                                        missile_traj, all_target_points, t_vec_full);
                
                % 更新最优解
                if current_obscured_time > max_obscured_time
                    max_obscured_time = current_obscured_time;
                    best_v = v_fy1;
                    best_t_deploy = t_deploy;
                    best_theta = theta_yaw;
                    best_t_delay = t_delay;
                end
            end
        end
    end
end

% 在循环结束后，根据最优参数计算最终结果
fprintf('\n优化完成！\n');
if max_obscured_time >= 0
    best_dir_vec_initial = (O - A0) / norm(O - A0);
    best_deploy_point = A0 + v_a * best_t_deploy * best_dir_vec_initial;
    best_deploy_v_vec = [best_v * cos(best_theta), best_v * sin(best_theta), 0];
    
    % *** 修正起爆点计算物理公式 ***
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

% ***********************************
% 3. 局部函数定义
% ***********************************
% 辅助函数：计算特定方案的遮蔽时长 (优化版)
function obscured_time = calculateObscuredTime_optimized(t_deploy, deploy_pos, deploy_v_vec, t_delay, missile_traj, all_target_points, t_vec_full)
    global v_down r_s t_smoke_effective dt g;

    t_exp = t_deploy + t_delay;
    t_sim_end = t_exp + t_smoke_effective;

    % 找到需要在循环内处理的时间步的索引
    start_idx = floor(t_exp / dt) + 1;
    end_idx = min(floor(t_sim_end / dt) + 1, length(t_vec_full));
    
    if start_idx > end_idx
        obscured_time = 0;
        return;
    end
    
    obscured_steps = 0;
    
    % 计算一次烟雾弹爆炸点
    smoke_exp_pos = deploy_pos + deploy_v_vec*t_delay + [0, 0, -0.5*g*t_delay^2];
    
    for i = start_idx:end_idx
        t = t_vec_full(i);
        
        missile_pos = missile_traj(i, :);
        
        % 烟雾中心位置
        delta_t_exp = t - t_exp;
        smoke_pos = smoke_exp_pos + [0, 0, -v_down * delta_t_exp];
        
        % *** 向量化遮蔽判断 ***
        if checkRaySphereIntersection_vec(missile_pos, all_target_points, smoke_pos, r_s)
            obscured_steps = obscured_steps + 1;
        end
    end
    
    % *** 修正时长计算 ***
    obscured_time = obscured_steps * dt;
end

% 向量化的导弹位置计算函数
function pos_matrix = getMissilePos_vec(t_vec, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos_matrix = M0 + t_vec' * (v_m * dir_vec); % t_vec' 使得结果为 N x 3 矩阵
end

% 向量化的射线与球体相交判断函数
function is_fully_obscured = checkRaySphereIntersection_vec(P, Q_matrix, C, r)
    % P: 1x3 射线起点 (导弹)
    % Q_matrix: 3xN 射线终点矩阵 (目标采样点)
    % C: 1x3 球心 (烟雾中心)
    % r: 标量 半径
    
    vec_ray = Q_matrix' - P;  % N x 3
    vec_PC = C - P;           % 1 x 3
    
    % 向量化点乘
    t = dot(vec_ray, repmat(vec_PC, size(vec_ray, 1), 1), 2) ./ dot(vec_ray, vec_ray, 2);
    
    % 检查 t 是否在 0 和 1 之间 (对于线段)
    % 此处我们关心的是射线，所以 t > 0
    valid_t_indices = t > 0;
    if ~any(valid_t_indices)
        is_fully_obscured = false;
        return;
    end
    
    % 计算射线到球心的最近距离的平方
    dist_sq = norm(vec_PC)^2 - (t.^2) .* dot(vec_ray, vec_ray, 2);
    
    intersect = (dist_sq <= r^2);
    
    % 如果所有有效射线都与球体相交，则判定为完全遮蔽
    is_fully_obscured = all(intersect(valid_t_indices));
end