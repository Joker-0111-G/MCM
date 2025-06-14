%% 初始化参数
clear; close all; clc;
warning off
time_step = 1 ; % 时间步长（秒）(这里修改，越大算的越快，但是精度也较低
spiral_pitch_range = (50:-0.5:40) * 1e-2; % 螺距取值范围（单位：米）
min_angle = zeros(size(spiral_pitch_range)); % 每个螺距对应的最小角度

% 主循环
for pitch_index = 1:length(spiral_pitch_range)
    current_pitch = spiral_pitch_range(pitch_index);
    spiral_coefficient = current_pitch / (2*pi); % 螺线方程系数 r = k*theta
    head_length = 341e-2; % 龙头长度（米）
    head_handle_distance = head_length - 27.5e-2*2; % 龙头把手两个孔之间的距离
    body_length = 220e-2; % 龙身和龙尾长度（米）
    body_handle_distance = body_length - 27.5e-2*2; % 其他凳子把手两个孔之间的距离
    
    % 确定初始位置
    initial_turns = ceil(4.5/current_pitch) + 3; % 从4.5米圆的外面的第三条螺线开始出发
    initial_angle = 2*pi*initial_turns; % 初始角度
    
    collision_flag = 0;
    iteration_count = 0;
    
    total_segments = 223; % 龙头+龙身+龙尾总的个数
    x_positions = nan(total_segments+1, 3); % 记录每个把手点在一个时间区间内的x坐标
    y_positions = nan(total_segments+1, 3); % 记录每个把手点在一个时间区间内的y坐标
    angle_data = nan(total_segments+1, 3); % 记录每个孔在时间区间的位置对应的角度theta
    angle_data(1,3) = initial_angle; % 头把手初始角度
    
    % 盘入模拟
    while collision_flag == 0
        iteration_count = iteration_count + 1;
        x_positions(:,1) = x_positions(:,3);
        y_positions(:,1) = y_positions(:,3);
        angle_data(:,1) = angle_data(:,3);
        
        % 求解龙头运动
        time_span = [0, time_step/2, time_step];
        [~, new_angles] = ode45(@(t,theta) -1/(spiral_coefficient*sqrt(1+theta^2)), time_span, angle_data(1,1));
        new_x = spiral_coefficient * new_angles .* cos(new_angles);
        new_y = spiral_coefficient * new_angles .* sin(new_angles);
        
        % 更新位置和角度
        x_positions(1,:) = new_x;
        y_positions(1,:) = new_y;
        angle_data(1,:) = new_angles;
        
        % 计算龙身和龙尾位置
        for time_index = 2:length(time_span)
            for segment_index = 2:total_segments+1
                handle_distance = head_handle_distance*(segment_index<=2) + body_handle_distance*(segment_index>2);
                new_angle = solve_angle(current_pitch, x_positions(segment_index-1,time_index), y_positions(segment_index-1,time_index), angle_data(segment_index-1,time_index), handle_distance);
                angle_data(segment_index,time_index) = new_angle;
                x_positions(segment_index,time_index) = spiral_coefficient * new_angle * cos(new_angle);
                y_positions(segment_index,time_index) = spiral_coefficient * new_angle * sin(new_angle);
            end
        end
        
        % 碰撞检测
        for segment_index = 1:round(total_segments/2)
            x1 = x_positions(segment_index,2); x2 = x_positions(segment_index+1,2);
            y1 = y_positions(segment_index,2); y2 = y_positions(segment_index+1,2);
            angle1 = angle_data(segment_index,2);
            angle2 = angle_data(segment_index+1,2);
            
            outer_indices = find((angle1+2*pi-angle_data(:,2))>0);
            outer_indices = outer_indices(max(1,end-2):end);
            inner_indices = find(angle_data(:,2)-(angle2+2*pi)>0);
            if isempty(inner_indices)
                break;
            else
                inner_indices = inner_indices(1:min(3,length(inner_indices)));
            end
            check_indices = outer_indices(1):inner_indices(end);
            
            for check_index = 1:length(check_indices)-1
                point1 = [x_positions(check_indices(check_index),2); y_positions(check_indices(check_index),2)];
                point2 = [x_positions(check_indices(check_index+1),2); y_positions(check_indices(check_index+1),2)];
                collision = check_intersection(head_length*(segment_index<=1)+body_length*(segment_index>1), [x1;y1], [x2;y2], body_length, point1, point2, 10, 20);
                if ~isempty(collision)
                    collision_flag = 1;
                    break;
                end
            end
            if collision_flag == 1
                break;
            end
        end
        
        % 检查是否到达调头空间边界
        if sqrt(x_positions(1,end)^2 + y_positions(1,end)^2) <= 4.5
            collision_flag = 1;
        end
        
        fprintf('当前螺距: %.2f m, 迭代步数: %d\n', current_pitch, iteration_count);
    end
    
    min_angle(pitch_index) = angle_data(1,end);
end
save problem3_tmp_data


%% 找到最优螺距
[~, optimal_index] = min(abs(min_angle - theoretical_min_angle));
optimal_pitch = spiral_pitch_range(optimal_index);
fprintf('最优螺距: %.4f m\n', optimal_pitch);


%% 辅助函数
function new_angle = solve_angle(pitch, x1, y1, angle1, distance)
    spiral_coeff = pitch / (2*pi);
    angle_func = @(angle) (spiral_coeff*angle.*cos(angle)-x1).^2 + (spiral_coeff*angle.*sin(angle)-y1).^2 - distance^2;
    options = optimoptions('fsolve', 'Display', 'off');
    new_angle = fsolve(angle_func, angle1+0.1, options);
    while new_angle <= angle1 || abs(spiral_coeff*new_angle-spiral_coeff*angle1) > pitch/2
        new_angle = fsolve(angle_func, new_angle+0.1, options);
    end
end

