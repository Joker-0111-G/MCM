%% 三维空间导弹-烟雾遮蔽模拟 - 三枚干扰弹最终优化版
% 修正内容:
% 1. 移除不准确的总迭代预算和百分比进度。
% 2. 采用更灵活的循环，精确实现 t_deploy(i+1) >= t_deploy(i) + 1s 的约束。
% 3. 为所有变量（速度、时间）提供更精细的搜索步长，提高优化精度。
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
% 2. 最终优化版：三枚烟幕弹干扰优化
% ***********************************
fprintf('开始进行三枚烟幕弹干扰方案优化 (最终优化版)...\n');

% --- 定义搜索范围和步长 (用户可根据需求调整) ---
% 时间步长
dt_search_deploy = 0.5;      % 投放时间的搜索步长 (s)
dt_search_delay = 0.5;     % 延迟引爆的搜索步长 (s)
% 速度步长
dv_search = 10;            % 速度的搜索步长 (m/s)
% 角度步长
d_theta_search = pi/16;     % 角度的搜索步长 (rad)

% 定义各变量的搜索范围
% =========================================================================
% --- 烟雾弹 1 的参数范围 ---
t_deploy1_range = 5:dt_search_deploy:10;        % 弹1的投放时间范围 (秒)。无人机在此时间点投放第一枚弹。
v_fy1_range = 70:dv_search:140;                % 弹1投放后，无人机的飞行速度范围 (米/秒)。
theta_yaw1_range = 0:d_theta_search:pi;         % 弹1投放后，无人机的飞行方向(偏航角)范围 (弧度)。pi代表180度。
t_delay1_range = 2:dt_search_delay:5;           % 弹1自身的延迟引爆时间范围 (秒)。

% --- 烟雾弹 2 的参数范围 ---
% 注意：弹2的投放时间 t_deploy2 是在循环中动态确定的 (t_deploy2 >= t_deploy1 + 1)。
v_fy2_range = 70:dv_search:140;                % 弹2投放后，无人机的飞行速度范围 (米/秒)。
theta_yaw2_range = 0:d_theta_search:2*pi;       % 弹2投放后，无人机的飞行方向范围 (弧度)。2*pi代表360度全向。
t_delay2_range = 2:dt_search_delay:5;           % 弹2自身的延迟引爆时间范围 (秒)。

% --- 烟雾弹 3 的参数范围 ---
% 注意：弹3的投放时间 t_deploy3 是在循环中动态确定的 (t_deploy3 >= t_deploy2 + 1)。
v_fy3_range = 70:dv_search:140;                % 弹3投放后，无人机的飞行速度范围 (米/秒)。
theta_yaw3_range = 0:d_theta_search:2*pi;       % 弹3投放后，无人机的飞行方向范围 (弧度)。
t_delay3_range = 2:dt_search_delay:5;           % 弹3自身的延迟引爆时间范围 (秒)。
% =========================================================================

% 初始化最优解存储
max_obscured_time = -1;
best_params = struct();
iter_count = 0;
dir_vec_initial = (O - A0) / norm(O - A0);

% === 遍历搜索空间 (采用精确时间约束) ===
% --- 烟雾弹 1 ---
for t_deploy1 = t_deploy1_range
    for v_fy1 = v_fy1_range
        for theta_yaw1 = theta_yaw1_range
            for t_delay1 = t_delay1_range
                
                % --- 烟雾弹 2 (精确实现 t_deploy2 >= t_deploy1 + 1) ---
                t_deploy2_start = t_deploy1 + 1;
                for t_deploy2 = t_deploy2_start:dt_search_deploy:t_end
                    for v_fy2 = v_fy2_range
                        for theta_yaw2 = theta_yaw2_range
                            for t_delay2 = t_delay2_range
                                
                                % --- 烟雾弹 3 (精确实现 t_deploy3 >= t_deploy2 + 1) ---
                                t_deploy3_start = t_deploy2 + 1;
                                for t_deploy3 = t_deploy3_start:dt_search_deploy:t_end
                                    for v_fy3 = v_fy3_range
                                        for theta_yaw3 = theta_yaw3_range
                                            for t_delay3 = t_delay3_range
                                                iter_count = iter_count + 1;
                                                if mod(iter_count, 2000) == 0
                                                    fprintf('已完成 %d 次有效组合计算... 当前最优时长: %.2f s\n', iter_count, max_obscured_time);
                                                end

                                                % 如果投放时间超出模拟总长，则跳过
                                                if t_deploy1 > t_end || t_deploy2 > t_end || t_deploy3 > t_end
                                                    continue;
                                                end

                                                % 物理模型计算
                                                A_pos_at_t_deploy1 = A0 + v_a * t_deploy1 * dir_vec_initial;
                                                deploy_v_vec1 = [v_fy1 * cos(theta_yaw1), v_fy1 * sin(theta_yaw1), 0];
                                                A_pos_at_t_deploy2 = A_pos_at_t_deploy1 + deploy_v_vec1 * (t_deploy2 - t_deploy1);
                                                deploy_v_vec2 = [v_fy2 * cos(theta_yaw2), v_fy2 * sin(theta_yaw2), 0];
                                                A_pos_at_t_deploy3 = A_pos_at_t_deploy2 + deploy_v_vec2 * (t_deploy3 - t_deploy2);
                                                deploy_v_vec3 = [v_fy3 * cos(theta_yaw3), v_fy3 * sin(theta_yaw3), 0];
                                                
                                                params.grenade1 = struct('t_deploy', t_deploy1, 'deploy_pos', A_pos_at_t_deploy1, 'deploy_v_vec', deploy_v_vec1, 't_delay', t_delay1, 'v_fy', v_fy1, 'theta_yaw', theta_yaw1);
                                                params.grenade2 = struct('t_deploy', t_deploy2, 'deploy_pos', A_pos_at_t_deploy2, 'deploy_v_vec', deploy_v_vec2, 't_delay', t_delay2, 'v_fy', v_fy2, 'theta_yaw', theta_yaw2);
                                                params.grenade3 = struct('t_deploy', t_deploy3, 'deploy_pos', A_pos_at_t_deploy3, 'deploy_v_vec', deploy_v_vec3, 't_delay', t_delay3, 'v_fy', v_fy3, 'theta_yaw', theta_yaw3);

                                                current_obscured_time = calculateObscuredTime_three_grenades(params, missile_traj, all_target_points, t_vec_full);

                                                if current_obscured_time > max_obscured_time
                                                    max_obscured_time = current_obscured_time;
                                                    best_params = params;
                                                    fprintf('!!! 找到新的最优解: %.2f s (在第 %d 次计算后)\n', max_obscured_time, iter_count);
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

% 循环结束后，报告最终结果
fprintf('\n优化完成！总共计算了 %d 种有效组合。\n', iter_count);
if max_obscured_time > 0
    % 计算最优起爆点
    exp_point1 = best_params.grenade1.deploy_pos + best_params.grenade1.deploy_v_vec * best_params.grenade1.t_delay + [0, 0, -0.5 * g * best_params.grenade1.t_delay^2];
    exp_point2 = best_params.grenade2.deploy_pos + best_params.grenade2.deploy_v_vec * best_params.grenade2.t_delay + [0, 0, -0.5 * g * best_params.grenade2.t_delay^2];
    exp_point3 = best_params.grenade3.deploy_pos + best_params.grenade3.deploy_v_vec * best_params.grenade3.t_delay + [0, 0, -0.5 * g * best_params.grenade3.t_delay^2];

    fprintf('最大总遮蔽时长: %.2f s\n\n', max_obscured_time);
    
    fprintf('--- 烟雾弹 1 ---\n');
    fprintf('投放时间: %.2f s\n', best_params.grenade1.t_deploy);
    fprintf('投放后无人机速度: %.2f m/s\n', best_params.grenade1.v_fy);
    fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', best_params.grenade1.theta_yaw, rad2deg(best_params.grenade1.theta_yaw));
    fprintf('延迟起爆时间: %.2f s\n', best_params.grenade1.t_delay);
    fprintf('投放点: (%.2f, %.2f, %.2f)\n', best_params.grenade1.deploy_pos);
    fprintf('起爆点: (%.2f, %.2f, %.2f)\n\n', exp_point1);

    fprintf('--- 烟雾弹 2 ---\n');
    fprintf('投放时间: %.2f s\n', best_params.grenade2.t_deploy);
    fprintf('投放后无人机速度: %.2f m/s\n', best_params.grenade2.v_fy);
    fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', best_params.grenade2.theta_yaw, rad2deg(best_params.grenade2.theta_yaw));
    fprintf('延迟起爆时间: %.2f s\n', best_params.grenade2.t_delay);
    fprintf('投放点: (%.2f, %.2f, %.2f)\n', best_params.grenade2.deploy_pos);
    fprintf('起爆点: (%.2f, %.2f, %.2f)\n\n', exp_point2);

    fprintf('--- 烟雾弹 3 ---\n');
    fprintf('投放时间: %.2f s\n', best_params.grenade3.t_deploy);
    fprintf('投放后无人机速度: %.2f m/s\n', best_params.grenade3.v_fy);
    fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', best_params.grenade3.theta_yaw, rad2deg(best_params.grenade3.theta_yaw));
    fprintf('延迟起爆时间: %.2f s\n', best_params.grenade3.t_delay);
    fprintf('投放点: (%.2f, %.2f, %.2f)\n', best_params.grenade3.deploy_pos);
    fprintf('起爆点: (%.2f, %.2f, %.2f)\n\n', exp_point3);

else
    fprintf('在搜索范围内未找到有效的遮蔽方案。\n');
end
toc;

% ***********************************
% 3. 局部函数定义 (此部分函数无需修改)
% ***********************************
% (此处省略 calculateObscuredTime_three_grenades, isTargetFullyObscured, 
% checkSingleRaySphereIntersection, getMissilePos_vec 等函数的代码，
% 因为它们与上一版相同，无需改动)
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