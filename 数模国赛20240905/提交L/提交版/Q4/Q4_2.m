clc,clear,close all;

% 定义基本参数
spiral_pitch = 1.7; % 螺距
spiral_coefficient = spiral_pitch / (2*pi); % 系数
head_length = 341e-2; % 龙头长度
head_handle_distance = head_length - 27.5e-2*2; % 龙头把手孔距
body_length = 220e-2; % 龙身龙尾长度
body_handle_distance = body_length - 27.5e-2*2; % 龙身把手孔距
head_speed = 1; % 龙头速度

% 绘制螺线
theta_range = 5*2*pi:-0.01:0*pi;
radius = spiral_coefficient * theta_range;
x_spiral_in = radius .* cos(theta_range);
y_spiral_in = radius .* sin(theta_range);

% 创建图形窗口并设置大小
figure('Name', '问题4：螺线和调头路径', 'Position', [200 200 600 600]);
spiral_in_color = rand(1,3);
plot(x_spiral_in, y_spiral_in, '-', 'Color', spiral_in_color, 'LineWidth', 1.3)
axis equal
grid on
xlabel('x (m)')
ylabel('y (m)')
set(gca, 'FontSize', 18)
hold on

% 计算并绘制盘出螺线
theta_out = theta_range - pi;
radius_out = spiral_coefficient * (theta_out + pi);
x_spiral_out = radius_out .* cos(theta_out);
y_spiral_out = radius_out .* sin(theta_out);
spiral_out_color = rand(1,3);
plot(x_spiral_out, y_spiral_out, '-', 'Color', spiral_out_color, 'LineWidth', 1.3)

% 绘制调头区域
turning_radius = 4.5; % 调头区域半径（米）
theta_turning = linspace(0, 2*pi, 100);
x_turning = turning_radius * cos(theta_turning);
y_turning = turning_radius * sin(theta_turning);
turning_area_color = sort(rand(1,3));
plot(x_turning, y_turning, 'Color', turning_area_color, 'LineWidth', 2)

% 添加图例
legend('盘入螺线', '盘出螺线', '调头边界',  '调头曲线1',  '圆心1','调头曲线2', '圆心2','Location', 'best')

% 计算螺线与调头区域的交点
theta_intersection_in = turning_radius / spiral_coefficient;
theta_intersection_out = turning_radius / spiral_coefficient - pi;

% 计算交点处的切线斜率
slope_intersection = (spiral_coefficient * sin(theta_intersection_in) + turning_radius * cos(theta_intersection_in)) / ...
                     (spiral_coefficient * cos(theta_intersection_in) - turning_radius * sin(theta_intersection_in));

% 计算调头曲线的几何参数
theta_max_1 = atan(-1/slope_intersection) + pi;
theta_equal_angle = atan(tan(theta_intersection_in)) + pi - theta_max_1;
r_centers = turning_radius / cos(theta_equal_angle);
radius_arc2 = r_centers / 3;
radius_arc1 = radius_arc2 * 2;
phi = 2 * theta_equal_angle;
arc_length1 = radius_arc1 * (pi - phi);
arc_length2 = radius_arc2 * (pi - phi);
theta_min_1 = theta_max_1 - arc_length1 / radius_arc1;
theta_min_2 = theta_min_1 - pi;
theta_max_2 = theta_min_2 + arc_length2 / radius_arc2;

% 计算圆弧中心坐标
x_center1 = turning_radius * cos(theta_intersection_in) + radius_arc1 * cos(theta_max_1 - pi);
y_center1 = turning_radius * sin(theta_intersection_in) + radius_arc1 * sin(theta_max_1 - pi);
x_center2 = turning_radius * cos(theta_intersection_out) - radius_arc2 * cos(theta_max_2);
y_center2 = turning_radius * sin(theta_intersection_out) - radius_arc2 * sin(theta_max_2);

% 绘制调头曲线
theta_arc1 = linspace(theta_min_1, theta_max_1, 50);
theta_arc2 = linspace(theta_min_2, theta_max_2, 50);
plot(x_center1 + radius_arc1 * cos(theta_arc1), y_center1 + radius_arc1 * sin(theta_arc1), 'r', 'LineWidth', 2)
plot(x_center1, y_center1, 'r*')
plot(x_center2 + radius_arc2 * cos(theta_arc2), y_center2 + radius_arc2 * sin(theta_arc2), 'b', 'LineWidth', 2)
plot(x_center2, y_center2, 'b*')

% 计算盘入螺线上头节点的位置
time_step = 1; % 时间步长（秒）
time_span = 0:time_step:100;
angle_derivative = @(t, theta) 1 ./ (spiral_coefficient * sqrt(1 + theta.^2));
[time, theta] = ode45(angle_derivative, time_span, theta_intersection_in);

% 初始化位置和角度数组
total_segments = 224;
total_time_steps = 200 / time_step + 1;
x_positions = zeros(total_segments, total_time_steps);
y_positions = zeros(total_segments, total_time_steps);
angle_data = zeros(total_segments, total_time_steps);

% 计算并填充头节点在盘入螺线上的位置
x_head_spiral_in = spiral_coefficient * theta .* cos(theta);
y_head_spiral_in = spiral_coefficient * theta .* sin(theta);
x_positions(1, 1:length(x_head_spiral_in)) = flip(x_head_spiral_in);
y_positions(1, 1:length(y_head_spiral_in)) = flip(y_head_spiral_in);
angle_data(1, 1:length(theta)) = flip(theta);

% 计算并填充头节点在圆弧C1上的位置
time_arc1 = time_step:time_step:arc_length1;
theta_arc1 = -time_arc1 / radius_arc1 + theta_max_1;
x_head_arc1 = radius_arc1 * cos(theta_arc1) + x_center1;
y_head_arc1 = radius_arc1 * sin(theta_arc1) + y_center1;
start_index_arc1 = length(theta) + 1;
end_index_arc1 = start_index_arc1 + length(time_arc1) - 1;
x_positions(1, start_index_arc1:end_index_arc1) = x_head_arc1;
y_positions(1, start_index_arc1:end_index_arc1) = y_head_arc1;
angle_data(1, start_index_arc1:end_index_arc1) = theta_arc1;

% 计算并填充头节点在圆弧C2的位置
time_arc2 = (time_arc1(end) + time_step):time_step:(arc_length1 + arc_length2);
theta_arc2 = (time_arc2 - arc_length1) / radius_arc2 + theta_min_1 - pi;
x_head_arc2 = radius_arc2 * cos(theta_arc2) + x_center2;
y_head_arc2 = radius_arc2 * sin(theta_arc2) + y_center2;
start_index_arc2 = end_index_arc1 + 1;
end_index_arc2 = start_index_arc2 + length(time_arc2) - 1;
x_positions(1, start_index_arc2:end_index_arc2) = x_head_arc2;
y_positions(1, start_index_arc2:end_index_arc2) = y_head_arc2;
angle_data(1, start_index_arc2:end_index_arc2) = theta_arc2;

% 计算头节点在盘出螺线上的位置
time_span_out = time_arc2(end) + time_step:time_step:100;
angle_derivative_out = @(t, theta) 1 ./ (spiral_coefficient * sqrt(1 + (theta + pi)^2));
[time_out, theta_out] = ode45(angle_derivative_out, time_span_out, theta_intersection_out);

% 计算头节点在盘出螺线上的位置
x_head_spiral_out = spiral_coefficient * (theta_out + pi) .* cos(theta_out);
y_head_spiral_out = spiral_coefficient * (theta_out + pi) .* sin(theta_out);

% 填充头节点在盘出螺线上的位置数据
start_index_spiral_out = end_index_arc2 + 1;
x_positions(1, start_index_spiral_out:end) = x_head_spiral_out;
y_positions(1, start_index_spiral_out:end) = y_head_spiral_out;
angle_data(1, start_index_spiral_out:end) = theta_out;

% 计算所有时间点的位置
total_time = -100:time_step:100;
for time_index = 1:length(total_time)
    current_time = total_time(time_index);
    
    if current_time <= 0 % 盘入螺线阶段
        for segment = 2:total_segments
            handle_distance = (segment <= 2) * head_handle_distance + (segment > 2) * body_handle_distance;
            next_angle = solve_angle(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance);
            angle_data(segment, time_index) = next_angle;
            x_positions(segment, time_index) = spiral_coefficient * next_angle * cos(next_angle);
            y_positions(segment, time_index) = spiral_coefficient * next_angle * sin(next_angle);
        end
    elseif current_time > 0 && current_time <= arc_length1 % 圆弧C1阶段
        arc_flag = 2;
        for segment = 2:total_segments
            handle_distance = (segment <= 2) * head_handle_distance + (segment > 2) * body_handle_distance;
            if arc_flag == 2 % 仍在圆弧C1上
                [x_next, y_next, angle_next, arc_flag] = solve_point_arc1(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance, radius_arc1, x_center1, y_center1, theta_max_1);
                angle_data(segment, time_index) = angle_next;
                x_positions(segment, time_index) = x_next;
                y_positions(segment, time_index) = y_next;
            else % 已经过渡到盘入螺线
                next_angle = solve_angle(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance);
                angle_data(segment, time_index) = next_angle;
                x_positions(segment, time_index) = spiral_coefficient * next_angle * cos(next_angle);
                y_positions(segment, time_index) = spiral_coefficient * next_angle * sin(next_angle);
            end
        end
    elseif current_time > arc_length1 && current_time <= (arc_length1 + arc_length2) % 圆弧C2阶段
        arc_flag = 3;
        for segment = 2:total_segments
            handle_distance = (segment <= 2) * head_handle_distance + (segment > 2) * body_handle_distance;
            if arc_flag == 3 % 
                [x_next, y_next, angle_next, arc_flag] = solve_point_arc2(x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance, radius_arc1, x_center1, y_center1, radius_arc2, x_center2, y_center2, theta_min_2);
                angle_data(segment, time_index) = angle_next;
                x_positions(segment, time_index) = x_next;
                y_positions(segment, time_index) = y_next;
            elseif arc_flag == 2 
                [x_next, y_next, angle_next, arc_flag] = solve_point_arc1(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance, radius_arc1, x_center1, y_center1, theta_max_1);
                angle_data(segment, time_index) = angle_next;
                x_positions(segment, time_index) = x_next;
                y_positions(segment, time_index) = y_next;
            else % 过渡到盘入螺线
                next_angle = solve_angle(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance);
                angle_data(segment, time_index) = next_angle;
                x_positions(segment, time_index) = spiral_coefficient * next_angle * cos(next_angle);
                y_positions(segment, time_index) = spiral_coefficient * next_angle * sin(next_angle);
            end
        end
    else % 盘出阶段
        spiral_out_flag = 4;
        for segment = 2:total_segments
            handle_distance = (segment <= 2) * head_handle_distance + (segment > 2) * body_handle_distance;
            if spiral_out_flag == 4 % 盘出螺线
                [x_next, y_next, angle_next, spiral_out_flag] = solve_point_spiral_out(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance, radius_arc2, x_center2, y_center2, theta_max_2);
                angle_data(segment, time_index) = angle_next;
                x_positions(segment, time_index) = x_next;
                y_positions(segment, time_index) = y_next;
            elseif spiral_out_flag == 3 
                [x_next, y_next, angle_next, spiral_out_flag] = solve_point_arc2(x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance, radius_arc1, x_center1, y_center1, radius_arc2, x_center2, y_center2, theta_min_2);
                angle_data(segment, time_index) = angle_next;
                x_positions(segment, time_index) = x_next;
                y_positions(segment, time_index) = y_next;
            elseif spiral_out_flag == 2 
                [x_next, y_next, angle_next, spiral_out_flag] = solve_point_arc1(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance, radius_arc1, x_center1, y_center1, theta_max_1);
                angle_data(segment, time_index) = angle_next;
                x_positions(segment, time_index) = x_next;
                y_positions(segment, time_index) = y_next;
            else % 过渡到盘入螺线
                next_angle = solve_angle(spiral_pitch, x_positions(segment-1, time_index), y_positions(segment-1, time_index), angle_data(segment-1, time_index), handle_distance);
                angle_data(segment, time_index) = next_angle;
                x_positions(segment, time_index) = spiral_coefficient * next_angle * cos(next_angle);
                y_positions(segment, time_index) = spiral_coefficient * next_angle * sin(next_angle);
            end
        end
    end
end

% 计算并保存速度数据
velocity_x = diff(x_positions, 1, 2) / time_step;
velocity_y = diff(y_positions, 1, 2) / time_step;
velocity = sqrt(velocity_x.^2 + velocity_y.^2);

% 保存结果
result_time = total_time';
result_x = x_positions';
result_y = y_positions';
result_velocity = [velocity, zeros(size(velocity, 1), 1)]';

% 指定时间点的速度
specified_times = [-100, -50, 0, 50, 100];
specified_segments = [1, 2, 52, 102, 152, 202, 224];  % 龙头前把手、第1、51、101、151、201节龙身前把手和龙尾后把手

figure('Name', '指定时间点的速度');
for t = 1:length(specified_times)
    nexttile
    hold on;
    
    time_index = find(abs(total_time - specified_times(t)) < 1e-6);
    
    % 绘制板凳龙
    plot(x_positions(:, time_index), y_positions(:, time_index), 'g-', 'LineWidth', 1);
    
    % 指定节点的速度
    for seg = specified_segments
        if time_index < size(velocity_x, 2)
            quiver(x_positions(seg, time_index), y_positions(seg, time_index), ...
                   velocity_x(seg, time_index), velocity_y(seg, time_index), ...
                   0.5, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
        end
    end
    
    title(sprintf('t = %d s', specified_times(t)));
    xlabel('x (m)');
    ylabel('y (m)');
    axis equal;
    axis([-20 20 -20 20]);
    grid on;
end

% 创建表格
table_data = cell(length(specified_segments), 5);
for i = 1:length(specified_times)
    time_index = find(abs(total_time - specified_times(i)) < 1e-6);
    for j = 1:length(specified_segments)
        seg = specified_segments(j);
        x = x_positions(seg, time_index);
        y = y_positions(seg, time_index);
        if time_index <= size(velocity, 2)
            v = velocity(seg, time_index);
        else
            v = NaN;
        end
        table_data{j, i} = sprintf('(%.4f, %.4f, %.4f)', x, y, v);
    end
end

% 创建表格
T = array2table(table_data, 'VariableNames', {'t_neg100', 't_neg50', 't_0', 't_50', 't_100'}, ...
                'RowNames', {'龙头前把手', '第1节龙身', '第51节龙身', '第101节龙身', '第151节龙身', '第201节龙身', '龙尾后把手'});
writetable(T, '问题4_指定时间点数据.xlsx', 'WriteRowNames', true);
disp(T);


function next_angle = solve_angle(spiral_pitch, x1, y1, theta1, d)
    % 求解下一个角度的函数
    k = spiral_pitch / (2*pi);
    distance_func = @(theta) (k*theta.*cos(theta)-x1).^2 + (k*theta.*sin(theta)-y1).^2 - d^2;
    options = optimset('Display', 'off');
    next_angle = fsolve(distance_func, theta1 + 0.1, options);
    
    while next_angle <= theta1 || abs(k*next_angle - k*theta1) > spiral_pitch/2
        next_angle = fsolve(distance_func, next_angle + 0.1, options);
    end
end

function [x, y, theta, flag] = solve_point_arc1(spiral_pitch, x1, y1, theta1, d, r_c1, x_c, y_c, theta_max)
    % 求解圆弧C1上的下一个点
    k = spiral_pitch / (2*pi);
    delta_theta = 2 * asin(d / (2 * r_c1));
    if delta_theta <= theta_max - theta1
        flag = 2;
        theta = theta1 + delta_theta;
        x = x_c + r_c1 * cos(theta);
        y = y_c + r_c1 * sin(theta);
    else
        theta = solve_angle(spiral_pitch, x1, y1, 4.5/k, d);
        flag = 1;
        x = k * theta * cos(theta);
        y = k * theta * sin(theta);
    end
end

function [x, y, theta, flag] = solve_point_arc2(x1, y1, theta1, d, r_c1, x_c1, y_c1, r_c2, x_c2, y_c2, theta_min)
    % 求解圆弧C2上的下一个点
    delta_theta = 2 * asin(d / (2 * r_c2));
    if delta_theta <= theta1 - theta_min
        flag = 3;
        theta = theta1 - delta_theta;
        x = x_c2 + r_c2 * cos(theta);
        y = y_c2 + r_c2 * sin(theta);
    else
        di = sqrt((x1-x_c1)^2 + (y1-y_c1)^2);
        delta_theta = acos((di^2 + r_c1^2 - d^2) / (2 * di * r_c1));
        theta_c1_di = atan2(y1-y_c1, x1-x_c1);
        theta = theta_c1_di + delta_theta;
        flag = 2;
        x = x_c1 + r_c1 * cos(theta);
        y = y_c1 + r_c1 * sin(theta);
    end
end

function [x, y, theta, flag] = solve_point_spiral_out(spiral_pitch, x1, y1, theta1, d, r_c2, x_c, y_c, theta_max)
    % 求解盘出螺线上的下一个点
    k = spiral_pitch / (2*pi);
    theta = solve_angle_out(spiral_pitch, x1, y1, theta1, d);
    if theta >= 4.5/k - pi
        flag = 4;
        x = k * (theta + pi) * cos(theta);
        y = k * (theta + pi) * sin(theta);
    else
        fun = @(t) (x_c + r_c2*cos(theta_max-t) - x1)^2 + (y_c + r_c2*sin(theta_max-t) - y1)^2 - d^2;
        options = optimset('Display', 'off');
        delta_theta = fsolve(fun, 0.1, options);
        theta = theta_max - delta_theta;
        flag = 3;
        x = x_c + r_c2 * cos(theta);
        y = y_c + r_c2 * sin(theta);
    end
end

function theta = solve_angle_out(spiral_pitch, x1, y1, theta1, d)
    % 求解盘出螺线上的下一个角度
    k = spiral_pitch / (2*pi);
    fun = @(theta) (k*(theta+pi).*cos(theta)-x1).^2 + (k*(theta+pi).*sin(theta)-y1).^2 - d^2;
    q = -0.1;
    options = optimset('Display', 'off');
    theta = fsolve(fun, theta1+q, options);
    while theta >= theta1 || abs(k*theta - k*theta1) > spiral_pitch/2
        q = q - 0.1;
        theta = fsolve(fun, theta+q, options);
    end
end