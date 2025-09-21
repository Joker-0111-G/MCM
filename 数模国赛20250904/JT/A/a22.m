%% 三维空间导弹-烟雾遮蔽模拟 - 完整脚本 (第二问优化版)

% ***********************************
% 1. 参数定义和初始化
% ***********************************

% 在使用变量前将需要全局化的变量声明为全局变量
global M0 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt t_end g;

% 初始坐标和速度
M0 = [20000, 0, 2000];   % 导弹初始坐标 (x, y, z)
v_m = 300;               % 导弹速度 (m/s)
A0 = [17800, 0, 1800];   % 飞机初始坐标 (x, y, z)
v_a = 120;               % 飞机速度 (m/s)
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

% ***********************************
% 2. 第二问：无人机FY1干扰优化
% ***********************************

fprintf('开始进行FY1无人机干扰方案优化...\n');

% 定义变量搜索范围
v_fy1_range = 70:3:150;        % 无人机速度范围
t_deploy_range = 1:0.05:15;      % 投放时间范围
theta_yaw_range = 0:pi/16:2*pi;  % 偏航角范围 (0到360度)
t_delay_range = 1:0.05:8;         % 延迟起爆时间范围

% 初始化最优解存储
max_obscured_time = 0;
best_v = 0;
best_t_deploy = 0;
best_theta = 0;
best_t_delay = 0;
best_deploy_point = [0,0,0];
best_exp_point = [0,0,0];

% 遍历搜索空间
for v_fy1 = v_fy1_range
    % 无人机FY1初始位置A0，假设其方向一直指向原点，直到投放前
    dir_vec_initial = (O - A0) / norm(O - A0);
    
    for t_deploy = t_deploy_range
        % 无人机在投放点的位置
        A_pos_at_t_deploy = A0 + v_a * t_deploy * dir_vec_initial;
        
        for theta_yaw = theta_yaw_range
            % 定义投放后无人机的速度向量（与XOY平面平行）
            deploy_v_vec = [v_fy1 * cos(theta_yaw), v_fy1 * sin(theta_yaw), 0];
            
            for t_delay = t_delay_range
                % 计算当前组合的遮蔽时间
                current_obscured_time = calculateObscuredTime(v_fy1, t_deploy, A_pos_at_t_deploy, deploy_v_vec, t_delay);
                
                % 更新最优解
                if current_obscured_time > max_obscured_time
                    max_obscured_time = current_obscured_time;
                    best_v = v_fy1;
                    best_t_deploy = t_deploy;
                    best_theta = theta_yaw;
                    best_t_delay = t_delay;
                    
                    % 记录最优投放点和起爆点
                    t_exp = t_deploy + t_delay;
                    best_deploy_point = A_pos_at_t_deploy;
                    best_exp_point = best_deploy_point + v_fy1 * t_delay * (deploy_v_vec / norm(deploy_v_vec));
                end
            end
        end
    end
end

fprintf('\n优化完成！\n');
fprintf('最优无人机速度: %.2f m/s\n', best_v);
fprintf('最优投放时间: %.2f s\n', best_t_deploy);
fprintf('最优延迟起爆时间: %.2f s\n', best_t_delay);
fprintf('最优方向 (偏航角): %.2f rad (%.2f 度)\n', best_theta, rad2deg(best_theta));
fprintf('最大遮蔽时长: %.2f s\n', max_obscured_time);
fprintf('最优投放点: (%.2f, %.2f, %.2f)\n', best_deploy_point);
fprintf('最优起爆点: (%.2f, %.2f, %.2f)\n', best_exp_point);

% ***********************************
% 3. 局部函数定义
% ***********************************
% 所有局部函数必须放在文件的末尾

% 辅助函数：计算特定方案的遮蔽时长
function obscured_time = calculateObscuredTime(v_fy1, t_deploy, deploy_pos, deploy_v_vec, t_delay)
    % 继承主脚本中的常量
    global M0 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt t_end g;
    
    t_exp = t_deploy + t_delay;
    t_vec_sim = t_deploy:dt:t_end;
    is_obscured_sim = false(size(t_vec_sim));

    for i = 1:length(t_vec_sim)
        t = t_vec_sim(i);
        if t >= t_exp && t <= t_exp + t_smoke_effective
            missile_pos = getMissilePos(t, M0, v_m, O);
            
            % 烟雾弹的平抛运动
            delta_t_deploy = t - t_deploy;
            smoke_bomb_pos_at_t = [deploy_pos(1) + deploy_v_vec(1)*delta_t_deploy, ...
                                   deploy_pos(2) + deploy_v_vec(2)*delta_t_deploy, ...
                                   deploy_pos(3) + deploy_v_vec(3)*delta_t_deploy - 0.5*g*delta_t_deploy^2];
            
            % 烟雾中心计算
            delta_t_exp = t - t_exp;
            smoke_exp_pos = [deploy_pos(1) + deploy_v_vec(1)*t_delay, ...
                             deploy_pos(2) + deploy_v_vec(2)*t_delay, ...
                             deploy_pos(3) + deploy_v_vec(3)*t_delay - 0.5*g*t_delay^2];
            smoke_pos = smoke_exp_pos + [0, 0, -v_down * delta_t_exp];
            
            is_fully_obscured_now = true;
            num_samples = 10; % 简化采样点，以加速计算
            theta_vec = linspace(0, 2*pi, num_samples+1);
            theta_vec(end) = [];
            
            % 检查下底圆周上的点
            for j = 1:num_samples
                x_t = T_c(1) + r_t * cos(theta_vec(j));
                y_t = T_c(2) + r_t * sin(theta_vec(j));
                z_t = T_c(3) - h_t/2;
                target_point = [x_t, y_t, z_t];
                if ~checkRaySphereIntersection(missile_pos, target_point, smoke_pos, r_s)
                    is_fully_obscured_now = false;
                    break;
                end
            end
            
            if ~is_fully_obscured_now
                is_obscured_sim(i) = false;
                continue;
            end
            
            % 检查上底圆周上的点
            for j = 1:num_samples
                x_t = T_c(1) + r_t * cos(theta_vec(j));
                y_t = T_c(2) + r_t * sin(theta_vec(j));
                z_t = T_c(3) + h_t/2;
                target_point = [x_t, y_t, z_t];
                if ~checkRaySphereIntersection(missile_pos, target_point, smoke_pos, r_s)
                    is_fully_obscured_now = false;
                    break;
                end
            end
            is_obscured_sim(i) = is_fully_obscured_now;
        end
    end
    
    obscured_indices = find(is_obscured_sim);
    if ~isempty(obscured_indices)
        start_index = obscured_indices(1);
        end_index = obscured_indices(end);
        obscured_time = t_vec_sim(end_index) - t_vec_sim(start_index);
    else
        obscured_time = 0;
    end
end

function pos = getMissilePos(t, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos = M0 + v_m * t * dir_vec;
end

function intersect = checkRaySphereIntersection(P, Q, C, r)
    vec_ray = Q - P;
    vec_PC = C - P;
    t = dot(vec_PC, vec_ray) / dot(vec_ray, vec_ray);
    if t < 0
        intersect = false;
        return;
    end
    dist_sq = norm(vec_PC)^2 - t^2 * norm(vec_ray)^2;
    if dist_sq <= r^2
        intersect = true;
    else
        intersect = false;
    end
end