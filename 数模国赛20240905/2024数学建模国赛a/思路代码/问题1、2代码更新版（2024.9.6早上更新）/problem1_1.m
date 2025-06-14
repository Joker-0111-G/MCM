
clear 

% 定义常量
L_head = 3.41; % 龙头长度（米）
L_body = 2.20; % 龙身长度（米）
L_eff = L_body - 2 * 0.275; % 有效长度（米）
spiral_pitch = 0.55; % 螺距（米）
a = spiral_pitch / (2 * pi); % 螺线参数
v_head = 1; % 龙头速度（米/秒）
total_time = 300; % 总时间（秒）
dt = 1; % 时间步长（秒）
n_segments = 223; % 板凳总数

% 初始化数组
t = 0:dt:total_time;
theta = zeros(n_segments, length(t));
x = zeros(n_segments, length(t));
y = zeros(n_segments, length(t));
vx = zeros(n_segments, length(t));
vy = zeros(n_segments, length(t));

% 初始化位置
theta(1, 1) = 16 * 2 * pi; % 龙头初始位置在第16圈
for i = 2:n_segments
    if i == 2
        delta_s = L_head;
    else
        delta_s = L_eff;
    end
    delta_theta = delta_s / (a * theta(i-1, 1));
    theta(i, 1) = theta(i-1, 1) + delta_theta;
end

% 计算初始位置的x和y坐标
for i = 1:n_segments
    x(i, 1) = a * theta(i, 1) * cos(theta(i, 1));
    y(i, 1) = a * theta(i, 1) * sin(theta(i, 1));
end

% 主循环
for j = 2:length(t)
    % 计算龙头新的theta（顺时针盘入，theta减小）
    delta_s = v_head * dt;
    delta_theta = delta_s / (a * theta(1, j-1));
    theta(1, j) = theta(1, j-1) - delta_theta;
    
    % 计算龙头位置
    x(1, j) = a * theta(1, j) * cos(theta(1, j));
    y(1, j) = a * theta(1, j) * sin(theta(1, j));
    
    % 计算龙头速度
    vx(1, j) = -v_head * sin(theta(1, j));
    vy(1, j) = v_head * cos(theta(1, j));
    
    % 计算龙身和龙尾位置和速度
    for i = 2:n_segments
        if i == 2
            delta_s = L_head;
        else
            delta_s = L_eff;
        end
        delta_theta = delta_s / (a * theta(i-1, j));
        theta(i, j) = theta(i-1, j) + delta_theta;
        
        x(i, j) = a * theta(i, j) * cos(theta(i, j));
        y(i, j) = a * theta(i, j) * sin(theta(i, j));
        
        % 计算速度
        vx(i, j) = (x(i, j) - x(i, j-1)) / dt;
        vy(i, j) = (y(i, j) - y(i, j-1)) / dt;
    end
end

% 保存result1.xlsx
result1 = zeros(447, length(t));
for i = 1:n_segments
    result1(2*i-1, :) = x(i, :);
    result1(2*i, :) = y(i, :);
end
result1(447, :) = sqrt(vx(end, :).^2 + vy(end, :).^2);
writematrix(round(result1', 6), '问题1_result1.xlsx');

% 保存论文中需要的表格数据
selected_times = [0, 60, 120, 180, 240, 300];
selected_segments = [1,  52, 102, 152, 202, 223];

position_table = zeros(14, 6);
velocity_table = zeros(7, 6);

for i = 1:length(selected_times)
    t_index = selected_times(i) + 1;
    for j = 1:length(selected_segments)
        seg_index = selected_segments(j);
        position_table(2*j-1, i) = x(seg_index, t_index);
        position_table(2*j, i) = y(seg_index, t_index);
        velocity_table(j, i) = sqrt(vx(seg_index, t_index)^2 + vy(seg_index, t_index)^2);
    end
end

% 保存位置表格
position_table_rounded = round(position_table, 6);
%位置表格
writematrix(position_table_rounded, 'Q1_Position.xlsx');

% 保存速度表格
velocity_table_rounded = round(velocity_table, 6);
%速度表格
writematrix(velocity_table_rounded, 'Q1_Speed.xlsx');

% 论文展现的位置以及速度
%  位置
disp('Position:');
disp(position_table_rounded);
%速度
disp('Speed:');
disp(velocity_table_rounded);

% 绘制运动轨迹图
figure('Position', [100, 100, 800, 600]);
plot(x(1,:), y(1,:), 'r-', 'LineWidth', 2);
hold on;
plot(x(end,:), y(end,:), 'b-', 'LineWidth', 2);
plot(x(:,end), y(:,end), 'g-', 'LineWidth', 2);
%运动轨迹
title('trajectory');
xlabel('X  (m)');
ylabel('Y  (m)');
% 龙头轨迹  龙尾轨迹  最终位置
legend('trajectory_h', 'trajectory_t', 'final position');
grid on;
axis equal;
saveas(gcf, 'Q1_trajectory.png');
saveas(gcf, 'Q1_trajectory.fig');


% 绘制特定时间点的速度向量图  
selected_times = [0, 60, 120, 180, 240, 300];  
selected_segments = [1, 2, 52, 102, 152, 202, 223]; % 龙头、龙头后第1、51、101、151、201节和龙尾  
  
for k = 1:length(selected_times)  
    time_index = selected_times(k) + 1; % MATLAB索引从1开始  
      
    % 提取当前时间点的位置和速度  
    x_current = x(:, time_index);  
    y_current = y(:, time_index);  
    vx_current = vx(:, time_index);  
    vy_current = vy(:, time_index);  
      
    % 筛选特定把手的位置和速度  
    x_selected = x_current(selected_segments);  
    y_selected = y_current(selected_segments);  
    vx_selected = vx_current(selected_segments);  
    vy_selected = vy_current(selected_segments);  
      
    % 绘制速度向量图  
    figure('Position', [100 + 200*(k-1), 100, 800, 600]); % 偏移窗口位置以便查看所有图形  
    quiver(x_selected, y_selected, vx_selected, vy_selected, 0.5, 'b');  
    title(sprintf('时间 %d 秒时的速度向量图', selected_times(k)));  
    xlabel('X 坐标 (米)');  
    ylabel('Y 坐标 (米)');  
    axis equal;  
    grid on;  
      
    % 保存图形  
    saveas(gcf, sprintf('时间%d秒速度向量.png', selected_times(k)));  
end  

