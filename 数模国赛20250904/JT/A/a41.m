%% 三维空间导弹-烟雾遮蔽模拟 - 三架无人机协同干扰
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
t_end = 60;              % 适当延长模拟总时长
dt = 0.01;               % 时间步长 (s)
t_vec_full = t_start:dt:t_end;

% ***********************************
% 1.5. 预计算与优化
% ***********************************
fprintf('进行预计算以加速优化过程...\n');
missile_traj = getMissilePos_vec(t_vec_full, M0, v_m, O);
num_samples = 50; % 为加速遍历，减少采样点
theta_vec = linspace(0, 2*pi, num_samples);
target_points_bottom = [T_c(1) + r_t * cos(theta_vec); T_c(2) + r_t * sin(theta_vec); repmat(T_c(3) - h_t/2, 1, num_samples)];
target_points_top = [T_c(1) + r_t * sin(theta_vec); T_c(2) + r_t * cos(theta_vec); repmat(T_c(3) + h_t/2, 1, num_samples)];
all_target_points = [target_points_bottom, target_points_top];


% ***********************************
% 2. 三无人机协同干扰优化 (暴力搜索)
% ***********************************
fprintf('开始进行三无人机协同干扰方案优化 (暴力搜索)...\n');
fprintf('警告: 这是一个12维搜索空间，为演示仅使用极小范围！\n');

% --- 定义搜索范围 (根据用户要求更新为精细范围) ---
% FY1
v_fy1_range = 70:5:140;        % 无人机速度范围 (m/s)
t_deploy1_range = 5:0.05:15;       % 投放时间范围 (s)
theta_yaw1_range = 0:pi/16:2*pi; % 偏航角范围 (rad)
t_delay1_range = 2:0.05:8;         % 延迟起爆时间范围 (s)
% FY2
v_fy2_range = 70:5:140;
t_deploy2_range = 5:0.05:15;
theta_yaw2_range = 0:pi/16:2*pi;
t_delay2_range = 2:0.05:8;
% FY3
v_fy3_range = 70:5:140;
t_deploy3_range = 5:0.05:15;
theta_yaw3_range = 0:pi/16:2*pi;
t_delay3_range = 2:0.05:8;

% 初始化最优解存储
max_obscured_time = -1;
best_params = struct();
iter_count = 0;

% === 12层嵌套循环遍历搜索空间 ===
for v_fy1 = v_fy1_range
 for t_deploy1 = t_deploy1_range
  for theta_yaw1 = theta_yaw1_range
   for t_delay1 = t_delay1_range
    for v_fy2 = v_fy2_range
     for t_deploy2 = t_deploy2_range
      for theta_yaw2 = theta_yaw2_range
       for t_delay2 = t_delay2_range
        for v_fy3 = v_fy3_range
         for t_deploy3 = t_deploy3_range
          for theta_yaw3 = theta_yaw3_range
           for t_delay3 = t_delay3_range
                iter_count = iter_count + 1;
                if mod(iter_count, 100) == 0
                    fprintf('已完成 %d 次组合计算... 当前最优时长: %.2f s\n', iter_count, max_obscured_time);
                end

                % 将所有参数打包
                params.p1 = struct('v_fy', v_fy1, 't_deploy', t_deploy1, 'theta_yaw', theta_yaw1, 't_delay', t_delay1, 'A0', A0_1);
                params.p2 = struct('v_fy', v_fy2, 't_deploy', t_deploy2, 'theta_yaw', theta_yaw2, 't_delay', t_delay2, 'A0', A0_2);
                params.p3 = struct('v_fy', v_fy3, 't_deploy', t_deploy3, 'theta_yaw', theta_yaw3, 't_delay', t_delay3, 'A0', A0_3);

                % 计算总遮蔽时间
                current_obscured_time = calculateObscuredTime_3UAVs(params, missile_traj, all_target_points, t_vec_full);

                % 更新最优解
                if current_obscured_time > max_obscured_time
                    max_obscured_time = current_obscured_time;
                    best_params = params;
                    fprintf('!!! 找到新的最优解: %.2f s\n', max_obscured_time);
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
    fprintf('最大总遮蔽时长: %.2f s\n\n', max_obscured_time);
    
    uavs = {'FY1', 'FY2', 'FY3'};
    ps = {best_params.p1, best_params.p2, best_params.p3};
    
    for i = 1:3
        p = ps{i};
        deploy_v_vec = [p.v_fy * cos(p.theta_yaw), p.v_fy * sin(p.theta_yaw), 0];
        deploy_point = p.A0 + deploy_v_vec * p.t_deploy;
        exp_point = deploy_point + deploy_v_vec * p.t_delay + [0, 0, -0.5 * g * p.t_delay^2];
        
        fprintf('--- %s 无人机最优参数 ---\n', uavs{i});
        fprintf('投放速度: %.2f m/s\n', p.v_fy);
        fprintf('投放时间: %.2f s\n', p.t_deploy);
        fprintf('投放方向 (偏航角): %.2f rad (%.2f 度)\n', p.theta_yaw, rad2deg(p.theta_yaw));
        fprintf('延迟引爆时间: %.2f s\n', p.t_delay);
        fprintf('投放点: (%.2f, %.2f, %.2f)\n', deploy_point);
        fprintf('起爆点: (%.2f, %.2f, %.2f)\n\n', exp_point);
    end
else
    fprintf('在搜索范围内未找到有效的遮蔽方案。\n');
end
toc;

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

% 检查单条射线（线段）与单个球体是否相交 (从a23.m修改而来)
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

% 向量化的导弹位置计算函数 (来自a23.m)
function pos_matrix = getMissilePos_vec(t_vec, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos_matrix = M0 + t_vec' * (v_m * dir_vec);
end