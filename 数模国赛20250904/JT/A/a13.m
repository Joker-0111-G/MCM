%% 三维空间导弹-烟雾遮蔽模拟 - 单文件脚本 (平抛运动最终版)

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
v_down = 3;              % 烟雾云团下沉速度 (m/s)
r_s = 10;                % 烟雾有效遮蔽半径 (m)
t_smoke_effective = 20;  % 烟雾有效遮蔽时长 (s)
g = 9.8;                 % 重力加速度 (m/s^2)

% 模拟时间参数
t_start = 0;
t_end = 40;              % 模拟总时长 (s)
dt = 0.0001;               % 时间步长 (s)
t_vec = t_start:dt:t_end;% 时间向量

% 结果存储
is_obscured = false(size(t_vec)); % 存储每个时间步的遮蔽状态

% ***********************************
% 2. 主模拟循环
% ***********************************

% 在循环前先计算好干扰弹起爆点
t_deploy = t1;
t_exp = t1 + t_delay;

plane_pos_t_deploy = getPlanePos(t_deploy, A0, v_a);
% 烟雾弹起爆点的平抛运动
smoke_exp_pos = [plane_pos_t_deploy(1) - v_a * t_delay, ...
                 plane_pos_t_deploy(2), ...
                 plane_pos_t_deploy(3) - 0.5 * g * t_delay^2];

for i = 1:length(t_vec)
    t = t_vec(i);
    
    % 判断是否在烟雾有效遮蔽时间内
    if t >= t_exp && t <= t_exp + t_smoke_effective
        missile_pos = getMissilePos(t, M0, v_m, O);
        
        % 烟雾云团中心位置（在起爆点基础上匀速下沉）
        smoke_pos = smoke_exp_pos + [0, 0, -v_down * (t - t_exp)];
        
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
% 3. 结果分析和可视化 (*** 已修改 ***)
% ***********************************

% 找到所有被遮蔽的时间点
if any(is_obscured)
    % 计算总遮蔽时长 (累加所有为 true 的时间步)
    total_obscured_duration = sum(is_obscured) * dt;
    fprintf('总遮蔽时长为：%.4f s。\n\n', total_obscured_duration);
    
    % 查找并打印每个独立的遮蔽时间段
    fprintf('真实目标被完全遮蔽的时间段如下：\n');
    
    % 使用 diff 寻找连续块的开始和结束
    % is_obscured(:) 确保其为列向量
    % 在首尾加0, 使得第一个为1和最后一个为1的情况也能被diff检测到
    diff_obscured = diff([0; is_obscured(:); 0]);
    start_indices = find(diff_obscured == 1);
    end_indices = find(diff_obscured == -1) - 1;
    
    for k = 1:length(start_indices)
        start_time = t_vec(start_indices(k));
        end_time = t_vec(end_indices(k));
        fprintf('  - 从 %.4f s 到 %.4f s\n', start_time, end_time);
    end
    
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
smoke_pos_traj(1:find(t_plot>=t_exp,1)-1,:) = NaN; % 在起爆前不绘制

% 计算烟雾弹的平抛轨迹
smoke_bomb_pos_traj = zeros(length(t_plot), 3);
for k = 1:length(t_plot)
    t = t_plot(k);
    if t < t_exp
        plane_pos_at_t = getPlanePos(t, A0, v_a);
        missile_traj(k, :) = getMissilePos(t, M0, v_m, O);
        plane_traj(k, :) = plane_pos_at_t;
    else
        % 烟雾轨迹
        missile_traj(k, :) = getMissilePos(t, M0, v_m, O);
        plane_traj(k, :) = getPlanePos(t, A0, v_a);
        smoke_pos_traj(k, :) = smoke_exp_pos + [0, 0, -v_down * (t - t_exp)];
    end
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

filename = 'a1.png'; % 定义图片文件名
saveas(gcf, filename); % gcf 获取当前图形句柄并保存

% ***********************************
% 4. 局部函数定义
% ***********************************
% 所有局部函数必须放在文件的末尾

function pos = getMissilePos(t, M0, v_m, O)
    % 计算导弹位置
    dir_vec = (O - M0) / norm(O - M0);
    pos = M0 + v_m * t * dir_vec;
end

function pos = getPlanePos(t, A0, v_a)
    % 计算飞机位置，垂直于Z轴，向X轴负方向前进
    pos = A0 + [-v_a * t, 0, 0];
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