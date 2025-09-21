%% 三维空间导弹-烟雾遮蔽模拟 - 单文件脚本

% ***********************************
% 1. 参数定义和初始化
% ***********************************

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
t1 = 1.5;                % 投放时间 (s)
t_delay = 3.6;           % 延时起爆时间 (s)
t2 = t1 + t_delay;       % 烟雾起爆时间 (s)
v_down = 3;              % 烟雾云团下沉速度 (m/s)
r_s = 10;                % 烟雾有效遮蔽半径 (m)
t_smoke_effective = 20;  % 烟雾有效遮蔽时长 (s)

% 模拟时间参数
t_start = 0;
t_end = 40;              % 模拟总时长 (s)
dt = 0.01;               % 时间步长 (s)
t_vec = t_start:dt:t_end;% 时间向量

% 结果存储
is_obscured = false(size(t_vec)); % 存储每个时间步的遮蔽状态

% ***********************************
% 2. 主模拟循环
% ***********************************

for i = 1:length(t_vec)
    t = t_vec(i);
    
    % 判断是否在烟雾有效遮蔽时间内
    if t >= t2 && t <= t2 + t_smoke_effective
        missile_pos = getMissilePos(t, M0, v_m, O);
        smoke_pos = getSmokePos(t, t1, t2, A0, v_a, O, v_down);
        
        % 判断真实目标是否完全被遮蔽
        is_fully_obscured_now = true;
        
        % 采样真实目标圆柱体
        num_samples = 10000;
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
            is_obscured(i) = false;
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
        
        is_obscured(i) = is_fully_obscured_now;
    end
end

% ***********************************
% 3. 结果分析和可视化
% ***********************************

% 找到第一个和最后一个连续遮蔽的时间点
obscured_indices = find(is_obscured);
if ~isempty(obscured_indices)
    start_index = obscured_indices(1);
    end_index = obscured_indices(end);
    
    start_time = t_vec(start_index);
    end_time = t_vec(end_index);
    
    obscured_duration = end_time - start_time;
    
    fprintf('真实目标完全被遮蔽的时间段为：%.2f s 到 %.2f s。\n', start_time, end_time);
    fprintf('总遮蔽时长为：%.2f s。\n', obscured_duration);
else
    fprintf('在模拟时间内，真实目标未被完全遮蔽。\n');
end

% 绘制三维空间轨迹和物体
figure;
hold on;
axis equal;
grid on;
title('三维空间模拟');
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');

% 绘制初始位置
plot3(M0(1), M0(2), M0(3), 'r.', 'MarkerSize', 20, 'DisplayName', '导弹初始位置');
plot3(A0(1), A0(2), A0(3), 'b.', 'MarkerSize', 20, 'DisplayName', '飞机初始位置');

% 绘制轨迹
t_plot = t_start:0.1:t_end;
missile_traj = zeros(length(t_plot), 3);
plane_traj = zeros(length(t_plot), 3);
smoke_pos_traj = zeros(length(t_plot), 3);

for k = 1:length(t_plot)
    missile_traj(k, :) = getMissilePos(t_plot(k), M0, v_m, O);
    plane_traj(k, :) = getPlanePos(t_plot(k), A0, v_a, O);
    smoke_pos_traj(k, :) = getSmokePos(t_plot(k), t1, t2, A0, v_a, O, v_down);
end

plot3(missile_traj(:,1), missile_traj(:,2), missile_traj(:,3), 'r-', 'LineWidth', 2, 'DisplayName', '导弹轨迹');
plot3(plane_traj(:,1), plane_traj(:,2), plane_traj(:,3), 'b--', 'LineWidth', 1, 'DisplayName', '飞机轨迹');
plot3(smoke_pos_traj(:,1), smoke_pos_traj(:,2), smoke_pos_traj(:,3), 'g.', 'MarkerSize', 10, 'DisplayName', '烟雾中心轨迹');

% 绘制真实目标 (圆柱体)
[X, Y, Z] = cylinder(r_t);
Z = Z * h_t - h_t/2;
X = X + T_c(1);
Y = Y + T_c(2);
Z = Z + T_c(3);
surf(X, Y, Z, 'FaceColor', 'y', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', '真实目标');

legend;
view(3);
hold off;

% ***********************************
% 4. 局部函数定义
% ***********************************
% 所有局部函数必须放在文件的末尾

function pos = getMissilePos(t, M0, v_m, O)
    % 计算导弹位置
    dir_vec = (O - M0) / norm(O - M0);
    pos = M0 + v_m * t * dir_vec;
end

function pos = getPlanePos(t, A0, v_a, O)
    % 计算飞机位置
    dir_vec = (O - A0) / norm(O - A0);
    pos = A0 + v_a * t * dir_vec;
end

function smoke_pos = getSmokePos(t, t1, t2, A0, v_a, O, v_down)
    % 计算烟雾中心位置
    if t < t2
        smoke_pos = [NaN, NaN, NaN]; % 烟雾尚未形成
    else
        dir_vec = (O - A0) / norm(O - A0);
        plane_pos_t1 = A0 + v_a * t1 * dir_vec;
        smoke_exp_pos = plane_pos_t1 + v_a * (t2 - t1) * dir_vec;
        smoke_pos = smoke_exp_pos + [0, 0, -v_down] * (t - t2);
    end
end

function intersect = checkRaySphereIntersection(P, Q, C, r)
    % 检查从点 P 到点 Q 的射线是否与球心为 C、半径为 r 的球体相交
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