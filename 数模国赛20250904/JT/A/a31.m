%% 三维空间导弹-烟雾遮蔽模拟 - 三枚干扰弹优化版
% 脚本作者: Gemini

% ***********************************
% 1. 参数定义和初始化
% ***********************************

% 将需要全局化的变量声明为全局变量
global M0 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt t_end g t_start;

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
% 2. 三枚干扰弹方案优化
% ***********************************

fprintf('开始进行三枚烟幕弹干扰方案优化...\n');

% 定义变量搜索范围 (为简化计算，此处使用较宽泛的范围，实际使用时可调整)
v_fy1_range = 70:20:150;
t_deploy_1_range = 1:5:15;
theta_yaw_range = 0:pi/4:2*pi;
t_delay_range = 1:3:8;

% 初始化最优解存储
max_total_obscured_time = 0;
best_params = struct(...
    'v_1', 0, 't_deploy_1', 0, 'theta_1', 0, 't_delay_1', 0, ...
    'v_2', 0, 't_deploy_2', 0, 'theta_2', 0, 't_delay_2', 0, ...
    'v_3', 0, 't_deploy_3', 0, 'theta_3', 0, 't_delay_3', 0 ...
);

% 导弹方向向量
dir_vec_initial_missile = (O - M0) / norm(O - M0);

% 飞机初始方向向量
dir_vec_initial_aircraft = (O - A0) / norm(O - A0);

% 嵌套循环进行穷举搜索
for v_fy1_1 = v_fy1_range
    for t_deploy_1 = t_deploy_1_range
        A_pos_at_t_deploy_1 = A0 + v_a * t_deploy_1 * dir_vec_initial_aircraft;
        for theta_yaw_1 = theta_yaw_range
            deploy_v_vec_1 = [v_fy1_1 * cos(theta_yaw_1), v_fy1_1 * sin(theta_yaw_1), 0];
            for t_delay_1 = t_delay_range

                % 烟幕弹2的投放时间必须晚于烟幕弹1至少1s
                t_deploy_2_range = (t_deploy_1 + 1):5:15;
                for v_fy1_2 = v_fy1_range
                    for t_deploy_2 = t_deploy_2_range
                        A_pos_at_t_deploy_2 = A0 + v_a * t_deploy_2 * dir_vec_initial_aircraft;
                        for theta_yaw_2 = theta_yaw_range
                            deploy_v_vec_2 = [v_fy1_2 * cos(theta_yaw_2), v_fy1_2 * sin(theta_yaw_2), 0];
                            for t_delay_2 = t_delay_range

                                % 烟幕弹3的投放时间必须晚于烟幕弹2至少1s
                                t_deploy_3_range = (t_deploy_2 + 1):5:15;
                                for v_fy1_3 = v_fy1_range
                                    for t_deploy_3 = t_deploy_3_range
                                        A_pos_at_t_deploy_3 = A0 + v_a * t_deploy_3 * dir_vec_initial_aircraft;
                                        for theta_yaw_3 = theta_yaw_range
                                            deploy_v_vec_3 = [v_fy1_3 * cos(theta_yaw_3), v_fy1_3 * sin(theta_yaw_3), 0];
                                            for t_delay_3 = t_delay_range
                                                
                                                % 调用新函数计算三枚烟幕弹的总遮蔽时间
                                                current_total_obscured_time = calculateTotalObscuredTime(...
                                                    t_deploy_1, A_pos_at_t_deploy_1, deploy_v_vec_1, t_delay_1, ...
                                                    t_deploy_2, A_pos_at_t_deploy_2, deploy_v_vec_2, t_delay_2, ...
                                                    t_deploy_3, A_pos_at_t_deploy_3, deploy_v_vec_3, t_delay_3);
                                                
                                                % 更新最优解
                                                if current_total_obscured_time > max_total_obscured_time
                                                    max_total_obscured_time = current_total_obscured_time;
                                                    best_params.v_1 = v_fy1_1;
                                                    best_params.t_deploy_1 = t_deploy_1;
                                                    best_params.theta_1 = theta_yaw_1;
                                                    best_params.t_delay_1 = t_delay_1;
                                                    
                                                    best_params.v_2 = v_fy1_2;
                                                    best_params.t_deploy_2 = t_deploy_2;
                                                    best_params.theta_2 = theta_yaw_2;
                                                    best_params.t_delay_2 = t_delay_2;
                                                    
                                                    best_params.v_3 = v_fy1_3;
                                                    best_params.t_deploy_3 = t_deploy_3;
                                                    best_params.theta_3 = theta_yaw_3;
                                                    best_params.t_delay_3 = t_delay_3;
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

fprintf('\n优化完成！\n');
fprintf('最大总遮蔽时长: %.2f s\n', max_total_obscured_time);
fprintf('最优烟幕弹1参数:\n');
fprintf('  速度: %.2f m/s, 投放时间: %.2f s, 偏航角: %.2f rad (%.2f 度), 延迟起爆: %.2f s\n', ...
    best_params.v_1, best_params.t_deploy_1, best_params.theta_1, rad2deg(best_params.theta_1), best_params.t_delay_1);
fprintf('最优烟幕弹2参数:\n');
fprintf('  速度: %.2f m/s, 投放时间: %.2f s, 偏航角: %.2f rad (%.2f 度), 延迟起爆: %.2f s\n', ...
    best_params.v_2, best_params.t_deploy_2, best_params.theta_2, rad2deg(best_params.theta_2), best_params.t_delay_2);
fprintf('最优烟幕弹3参数:\n');
fprintf('  速度: %.2f m/s, 投放时间: %.2f s, 偏航角: %.2f rad (%.2f 度), 延迟起爆: %.2f s\n', ...
    best_params.v_3, best_params.t_deploy_3, best_params.theta_3, rad2deg(best_params.theta_3), best_params.t_delay_3);


% ***********************************
% 3. 局部函数定义
% ***********************************

% 新增核心函数：计算三枚烟幕弹的总遮蔽时长
function obscured_time = calculateTotalObscuredTime(...
    t_deploy_1, deploy_pos_1, deploy_v_vec_1, t_delay_1, ...
    t_deploy_2, deploy_pos_2, deploy_v_vec_2, t_delay_2, ...
    t_deploy_3, deploy_pos_3, deploy_v_vec_3, t_delay_3)
    
    % 继承主脚本中的常量
    global M0 v_m O T_c r_t h_t v_down r_s t_smoke_effective dt t_end g t_start;

    % 计算每枚烟幕弹的起爆时间
    t_exp_1 = t_deploy_1 + t_delay_1;
    t_exp_2 = t_deploy_2 + t_delay_2;
    t_exp_3 = t_deploy_3 + t_delay_3;

    t_vec_sim = t_start:dt:t_end;
    is_fully_obscured_sim = false(size(t_vec_sim));

    for i = 1:length(t_vec_sim)
        t = t_vec_sim(i);
        missile_pos = getMissilePos(t, M0, v_m, O);
        
        % 检查当前时刻目标是否被完全遮蔽
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
            
            % 判断该点是否被任一烟幕弹遮蔽
            is_obscured_by_any = false;
            
            % 检查烟幕弹1
            if t >= t_exp_1 && t <= t_exp_1 + t_smoke_effective
                smoke_pos_1 = calculateSmokePos(t, t_exp_1, deploy_pos_1, deploy_v_vec_1);
                if checkRaySphereIntersection(missile_pos, target_point, smoke_pos_1, r_s)
                    is_obscured_by_any = true;
                end
            end
            
            % 检查烟幕弹2
            if ~is_obscured_by_any && t >= t_exp_2 && t <= t_exp_2 + t_smoke_effective
                smoke_pos_2 = calculateSmokePos(t, t_exp_2, deploy_pos_2, deploy_v_vec_2);
                if checkRaySphereIntersection(missile_pos, target_point, smoke_pos_2, r_s)
                    is_obscured_by_any = true;
                end
            end
            
            % 检查烟幕弹3
            if ~is_obscured_by_any && t >= t_exp_3 && t <= t_exp_3 + t_smoke_effective
                smoke_pos_3 = calculateSmokePos(t, t_exp_3, deploy_pos_3, deploy_v_vec_3);
                if checkRaySphereIntersection(missile_pos, target_point, smoke_pos_3, r_s)
                    is_obscured_by_any = true;
                end
            end

            if ~is_obscured_by_any
                is_fully_obscured_now = false;
                break;
            end
        end

        % 如果下底被遮蔽，继续检查上底
        if is_fully_obscured_now
            for j = 1:num_samples
                x_t = T_c(1) + r_t * cos(theta_vec(j));
                y_t = T_c(2) + r_t * sin(theta_vec(j));
                z_t = T_c(3) + h_t/2;
                target_point = [x_t, y_t, z_t];
                
                is_obscured_by_any = false;
                if t >= t_exp_1 && t <= t_exp_1 + t_smoke_effective
                    smoke_pos_1 = calculateSmokePos(t, t_exp_1, deploy_pos_1, deploy_v_vec_1);
                    if checkRaySphereIntersection(missile_pos, target_point, smoke_pos_1, r_s)
                        is_obscured_by_any = true;
                    end
                end
                
                if ~is_obscured_by_any && t >= t_exp_2 && t <= t_exp_2 + t_smoke_effective
                    smoke_pos_2 = calculateSmokePos(t, t_exp_2, deploy_pos_2, deploy_v_vec_2);
                    if checkRaySphereIntersection(missile_pos, target_point, smoke_pos_2, r_s)
                        is_obscured_by_any = true;
                    end
                end
                
                if ~is_obscured_by_any && t >= t_exp_3 && t <= t_exp_3 + t_smoke_effective
                    smoke_pos_3 = calculateSmokePos(t, t_exp_3, deploy_pos_3, deploy_v_vec_3);
                    if checkRaySphereIntersection(missile_pos, target_point, smoke_pos_3, r_s)
                        is_obscured_by_any = true;
                    end
                end

                if ~is_obscured_by_any
                    is_fully_obscured_now = false;
                    break;
                end
            end
        end
        
        is_fully_obscured_sim(i) = is_fully_obscured_now;
    end
    
    obscured_time = sum(is_fully_obscured_sim) * dt;
end

% 辅助函数：计算烟幕中心位置
function pos = calculateSmokePos(t, t_exp, deploy_pos, deploy_v_vec)
    global g v_down;
    
    if t < t_exp
        % 烟幕弹在起爆前是平抛运动
        delta_t = t - (t_exp - t_delay);
        pos = [deploy_pos(1) + deploy_v_vec(1)*delta_t, ...
               deploy_pos(2) + deploy_v_vec(2)*delta_t, ...
               deploy_pos(3) + deploy_v_vec(3)*delta_t - 0.5*g*delta_t^2];
    else
        % 烟雾中心在起爆后匀速下沉
        delta_t_exp = t - t_exp;
        
        % 烟雾起爆点位置
        t_delay = t_exp - t_deploy;
        smoke_exp_pos = [deploy_pos(1) + deploy_v_vec(1)*t_delay, ...
                         deploy_pos(2) + deploy_v_vec(2)*t_delay, ...
                         deploy_pos(3) + deploy_v_vec(3)*t_delay - 0.5*g*t_delay^2];
                         
        pos = smoke_exp_pos + [0, 0, -v_down * delta_t_exp];
    end
end


% 辅助函数：获取导弹位置
function pos = getMissilePos(t, M0, v_m, O)
    dir_vec = (O - M0) / norm(O - M0);
    pos = M0 + v_m * t * dir_vec;
end

% 辅助函数：检查射线与球体是否相交（即目标点是否被遮蔽）
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