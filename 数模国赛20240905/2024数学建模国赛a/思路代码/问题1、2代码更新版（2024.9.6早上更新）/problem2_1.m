% 清空工作区和关闭所有图形窗口
clear all;
close all;

% 定义常量
L_head = 3.41; % 龙头长度（米）
L_body = 2.20; % 龙身长度（米）
L_eff = L_body - 2 * 0.275; % 有效长度（米）
L_extend = 0.275; % 外延长度（米）
spiral_pitch = 0.55; % 螺距（米）
a = spiral_pitch / (2 * pi); % 螺线参数
v_head = 1; % 龙头速度（米/秒）
total_time = 1000; % 模拟总时间（秒）
dt = 0.5; % 时间步长（秒）
n_segments = 223; % 板凳总数
collision_tolerance = 0.1; % 碰撞检测容差

% 初始化数组
t = 0:dt:total_time;
theta = zeros(n_segments, length(t));
x = zeros(n_segments, length(t));
y = zeros(n_segments, length(t));
vx = zeros(n_segments, length(t));
vy = zeros(n_segments, length(t));


% 初始化位置（所有板凳都在螺线上）
theta_start = 16 * 2 * pi; % 起始角度（第16圈）
for i = 1:n_segments
    if i == 1
        theta(i, 1) = theta_start;
    else
        % 计算下一个板凳的theta
        if i==2
        delta_s = L_head ;
        else 
            delta_s =  L_eff;
        end
        delta_theta = delta_s / (a * sqrt(1 + theta(i-1, 1)^2));
        theta(i, 1) = theta(i-1, 1) + delta_theta; 
    end
    
    % 计算初始位置
    x(i, 1) = a * theta(i, 1) * cos(theta(i, 1));
    y(i, 1) = a * theta(i, 1) * sin(theta(i, 1));
end
figure,

plot(x(:,1),y(:,1), 'r-', 'LineWidth', 2);

title('板凳龙初始位置');
xlabel('X 坐标 (米)');
ylabel('Y 坐标 (米)');
grid on;
axis equal;

%%

% 主循环
collision_time = -1;
for j = 2:length(t)
    % 更新位置和速度
    delta_s = v_head * dt;
    delta_theta = delta_s ./ (a * theta(:, j-1));
    theta(:, j) = theta(:, j-1) - delta_theta;
    
    x(:, j) = a * theta(:, j) .* cos(theta(:, j));
    y(:, j) = a * theta(:, j) .* sin(theta(:, j));
    
    vx(:, j) = (x(:, j) - x(:, j-1)) / dt;
    vy(:, j) = (y(:, j) - y(:, j-1)) / dt;
    
    % 碰撞检测
    seg_start = [x(:, j) - L_extend * cos(theta(:, j)), y(:, j) - L_extend * sin(theta(:, j))];
    seg_end = [x(:, j) + (L_body + L_extend) * cos(theta(:, j)), y(:, j) + (L_body + L_extend) * sin(theta(:, j))];
    
    % 向量化碰撞检测
    [intersect, ~] = lineSegmentIntersect(reshape([seg_start, seg_end]', 4, [])');
    intersect = reshape(intersect, n_segments, n_segments);
    intersect = triu(intersect, 1); % 只考虑上三角矩阵，避免自交和重复检测
    
    if any(intersect(:))
        collision_time = t(j);
        break;
    end
end

% 保存result2.xlsx
result2 = zeros(n_segments, 3);
result2(:,1) = x(:, j);
result2(:,2) = y(:, j);
result2(:,3) = sqrt(vx(end, j)^2 + vy(end, j)^2);
writematrix(round(result2, 6), '问题2_result2.xlsx');

% 保存论文中需要的表格数据
selected_segments = [1, 2, 52, 102, 152, 202, 223];

position_table = zeros(14, 1);
velocity_table = zeros(7, 1);

for i = 1:length(selected_segments)
    seg_index = selected_segments(i);
    position_table(2*i-1, 1) = x(seg_index, j);
    position_table(2*i, 1) = y(seg_index, j);
    velocity_table(i, 1) = sqrt(vx(seg_index, j)^2 + vy(seg_index, j)^2);
end

% 保存位置表格
position_table_rounded = round(position_table, 6);
writematrix(position_table_rounded, '问题2_位置表格.xlsx');

% 保存速度表格
velocity_table_rounded = round(velocity_table, 6);
writematrix(velocity_table_rounded, '问题2_速度表格.xlsx');

% 打印表格（用于论文）
disp('位置表格：');
disp(position_table_rounded);
disp('速度表格：');
disp(velocity_table_rounded);
disp(['碰撞时刻：', num2str(collision_time), ' 秒']);

% 绘制碰撞时刻的板凳龙形状
figure('Position', [100, 100, 800, 600]);
plot(x(:, j), y(:, j), 'b-', 'LineWidth', 2);
hold on;
plot(x(1, j), y(1, j), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(x(end, j), y(end, j), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
for i = 1:n_segments
    plot([x(i, j), x(i, j) + L_body * cos(theta(i, j))], [y(i, j), y(i, j) + L_body * sin(theta(i, j))], 'r-', 'LineWidth', 1);
end
title('碰撞时刻的板凳龙形状');
xlabel('X 坐标 (米)');
ylabel('Y 坐标 (米)');
axis equal;
grid on;
saveas(gcf, '问题2_碰撞时刻形状.png');
saveas(gcf, '问题2_碰撞时刻形状.fig');

% 绘制运动轨迹
figure('Position', [100, 100, 800, 600]);
plot(x(:, 1:j), y(:, 1:j), 'b-', 'LineWidth', 1);
hold on;
plot(x(1, j), y(1, j), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
plot(x(end, j), y(end, j), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
title('板凳龙运动轨迹');
xlabel('X 坐标 (米)');
ylabel('Y 坐标 (米)');
axis equal;
grid on;
saveas(gcf, '问题2_运动轨迹.png');
saveas(gcf, '问题2_运动轨迹.fig');

% 设置图像分辨率为300dpi并重新保存
print('问题2_碰撞时刻形状', '-dpng', '-r300');
print('问题2_运动轨迹', '-dpng', '-r300');


% 线段相交检测函数（向量化版本）
function [I,P] = lineSegmentIntersect(XY)
    X1 = XY(:,1); Y1 = XY(:,2);
    X2 = XY(:,3); Y2 = XY(:,4);
    
    % 创建所有可能的线段对
    [X1, X3] = meshgrid(X1, X1);
    [Y1, Y3] = meshgrid(Y1, Y1);
    [X2, X4] = meshgrid(X2, X2);
    [Y2, Y4] = meshgrid(Y2, Y2);
    
    % 计算分母
    D = (X1-X2).*(Y3-Y4) - (Y1-Y2).*(X3-X4);
    
    % 计算 ua 和 ub
    ua = ((X1-X3).*(Y3-Y4) - (Y1-Y3).*(X3-X4)) ./ D;
    ub = ((X1-X3).*(Y1-Y2) - (Y1-Y3).*(X1-X2)) ./ D;
    
    % 判断是否相交
    I = (ua >= 0 & ua <= 1 & ub >= 0 & ub <= 1);
    
    % 计算交点坐标（如果需要）
    P = [X1 + ua.*(X2-X1), Y1 + ua.*(Y2-Y1)];
end