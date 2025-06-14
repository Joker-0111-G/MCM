
clear; close all; clc;  
warning off  
ts = 1;             % 时间步长（秒）  
pitch_range = (50:-0.5:40) * 1e-2; % 螺距取值范围（米）  
min_ang = zeros(size(pitch_range)); % 每个螺距对应的最小角度  
  
 
for p_idx = 1:length(pitch_range)  
    p = pitch_range(p_idx);  
    k = p / (2*pi);      % 螺线方程系数  
    hl = 341e-2;         % 龙头长度（米）  
    hh_d = hl - 27.5e-2*2;% 龙头把手距离  
    bl = 220e-2;         % 龙身和龙尾长度（米）  
    bh_d = bl - 27.5e-2*2;% 其他凳子把手距离  
      
    % 确定初始位置  
    init_turns = ceil(4.5/p) + 3;  
    init_ang = 2*pi*init_turns;  
      
    coll_flag = 0;  
    iter_count = 0;  
      
    n_segs = 223; % 龙头+龙身+龙尾总数  
    x_pos = nan(n_segs+1, 3);  
    y_pos = nan(n_segs+1, 3);  
    ang_data = nan(n_segs+1, 3);  
    ang_data(1,3) = init_ang;  
      
    % 盘入模拟  
    while coll_flag == 0  
        iter_count = iter_count + 1;  
        x_pos(:,1) = x_pos(:,3);  
        y_pos(:,1) = y_pos(:,3);  
        ang_data(:,1) = ang_data(:,3);  
          
        % 求解龙头运动  
        tspan = [0, ts/2, ts];  
        [~, new_angs] = ode45(@(t,theta) -1/(k*sqrt(1+theta^2)), tspan, ang_data(1,1));  
        new_x = k * new_angs .* cos(new_angs);  
        new_y = k * new_angs .* sin(new_angs);  
          
        % 更新位置和角度  
        x_pos(1,:) = new_x;  
        y_pos(1,:) = new_y;  
        ang_data(1,:) = new_angs;  
          
        % 计算龙身和龙尾位置  
        for t_idx = 2:length(tspan)  
            for seg_idx = 2:n_segs+1  
                h_d = hh_d*(seg_idx<=2) + bh_d*(seg_idx>2);  
                new_ang = solve_angle(p, x_pos(seg_idx-1,t_idx), y_pos(seg_idx-1,t_idx), ang_data(seg_idx-1,t_idx), h_d);  
                ang_data(seg_idx,t_idx) = new_ang;  
                x_pos(seg_idx,t_idx) = k * new_ang * cos(new_ang);  
                y_pos(seg_idx,t_idx) = k * new_ang * sin(new_ang);  
            end  
        end  
          
        % 碰 撞  
        for seg_idx = 1:round(n_segs/2)  
            x1 = x_pos(seg_idx,2); x2 = x_pos(seg_idx+1,2);  
            y1 = y_pos(seg_idx,2); y2 = y_pos(seg_idx+1,2);  
            ang1 = ang_data(seg_idx,2);  
            ang2 = ang_data(seg_idx+1,2);  
              
            outer_idxs = find((ang1+2*pi-ang_data(:,2))>0);  
            outer_idxs = outer_idxs(max(1,end-2):end);  
            inner_idxs = find(ang_data(:,2)-(ang2+2*pi)>0);  
            if isempty(inner_idxs)  
                break;  
            else  
                inner_idxs = inner_idxs(1:min(3,length(inner_idxs)));  
            end  
            check_idxs = outer_idxs(1):inner_idxs(end);  
              
            for c_idx = 1:length(check_idxs)-1  
                pt1 = [x_pos(check_idxs(c_idx),2); y_pos(check_idxs(c_idx),2)];  
                pt2 = [x_pos(check_idxs(c_idx+1),2); y_pos(check_idxs(c_idx+1),2)];  
                coll = check_intersection(head_length*(seg_idx<=1)+body_length*(seg_idx>1), [x1;y1], [x2;y2], body_length, pt1, pt2, 10, 20);  
                if ~isempty(coll)  
                    coll_flag = 1;  
                    break;  
                end  
            end  
            if coll_flag == 1  
                break;  
            end  
        end  
          
        % 检查到达调头空间边界  
        if sqrt(x_pos(1,end)^2 + y_pos(1,end)^2) <= 4.5  
            coll_flag = 1;  
        end  
          
        fprintf('当前螺距: %.2f m, 迭代步数: %d\n', p, iter_count);  
    end  
      
    min_ang(p_idx) = ang_data(1,end);  
end  
save('problem3_tmp_data');  
  
% 找到最优螺距（注意：需要定义theoretical_min_angle）  
[~, optimal_idx] = min(abs(min_ang - theoretical_min_angle));  
optimal_pitch = pitch_range(optimal_idx);  
fprintf('最优螺距: %.4f m\n', optimal_pitch);  
  
% 辅助函数...（保持不变）

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

