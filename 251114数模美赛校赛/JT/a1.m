function evacuation_advanced_solver_delayed()
    % ================= 1. 参数初始化 =================
    clc; clear; close all;
    
    global ROOM_W ROOM_H SPEED_BASE R_MAX R_MIN LAMBDA EFFICIENCY EXITS ROOMS;
    
    % --- 【在此处修改起始时间】 ---
    % 设定搜救人员进入建筑的延迟时间 (秒)
    % 0 = 立即进入; 120 = 火灾发生2分钟后进入 (此时烟雾已较浓)
    START_DELAY = 120; 
    
    fprintf('当前设定起始延迟时间: %.0f 秒\n', START_DELAY);

    % --- 几何尺寸 ---
    ROOM_W = 6.0;
    ROOM_H = 5.0;
    
    % --- 运动与烟雾参数 ---
    SPEED_BASE = 1.0; % m/s
    R_MAX = sqrt(ROOM_W^2 + ROOM_H^2); % 约 7.81m
    R_MIN = 0.5;
    T_CRIT = 6 * 60; % 360s
    LAMBDA = 3 / T_CRIT;
    
    % 搜索效率
    EFFICIENCY = 0.3; 
    
    % --- 定义出口坐标 ---
    EXITS.Left = [-1, 6.5];
    EXITS.Right = [19, 6.5];
    
    % --- 定义房间信息 ---
    ROOMS = struct();
    ROOMS(1).rect = [0, 8, 6, 5]; ROOMS(1).door = [5.5, 8]; ROOMS(1).label = 'R1';
    ROOMS(2).rect = [6, 8, 6, 5]; ROOMS(2).door = [9.0, 8]; ROOMS(2).label = 'R2';
    ROOMS(3).rect = [12, 8, 6, 5];ROOMS(3).door = [12.5, 8];ROOMS(3).label = 'R3';
    ROOMS(4).rect = [0, 0, 6, 5]; ROOMS(4).door = [5.5, 5]; ROOMS(4).label = 'R4';
    ROOMS(5).rect = [6, 0, 6, 5]; ROOMS(5).door = [9.0, 5]; ROOMS(5).label = 'R5';
    ROOMS(6).rect = [12, 0, 6, 5];ROOMS(6).door = [12.5, 5];ROOMS(6).label = 'R6';

    % ================= 2. 全局优化求解 =================
    fprintf('正在计算延迟 %.0fs 后的最优策略...\n', START_DELAY);
    
    best_time = inf;
    best_solution = [];
    
    all_rooms = 1:6;
    all_perms = perms(all_rooms); 
    [n_perms, ~] = size(all_perms);
    
    % 遍历所有排列
    for i = 1:n_perms
        current_perm = all_perms(i, :);
        
        % 遍历所有切分点 k
        for k = 1:5
            group1 = current_perm(1:k);
            group2 = current_perm(k+1:end);
            
            exit_opts = {'Left', 'Right'};
            
            for s1 = 1:2 
            for s2 = 1:2 
                % 【修改点】：传入 START_DELAY 作为起始时间
                [t1, trace1] = simulate_route_strict(group1, exit_opts{s1}, START_DELAY);
                [t2, trace2] = simulate_route_strict(group2, exit_opts{s2}, START_DELAY);
                
                % 系统完成时刻 (注意：t1, t2 已经是包含 delay 的绝对时刻)
                total_system_time = max(t1, t2);
                
                if total_system_time < best_time
                    best_time = total_system_time;
                    best_solution.t_total = total_system_time;
                    best_solution.start_delay = START_DELAY; % 记录延迟
                    best_solution.r1 = struct('path', group1, 'start', exit_opts{s1}, 'time', t1, 'trace', trace1);
                    best_solution.r2 = struct('path', group2, 'start', exit_opts{s2}, 'time', t2, 'trace', trace2);
                end
            end
            end
        end
    end
    
    % ================= 3. 结果输出与绘图 =================
    if isempty(best_solution)
        fprintf('错误：未能找到可行解。\n');
        return;
    end

    fprintf('------------------------------------------------\n');
    fprintf('起始延迟: %.0f 秒\n', best_solution.start_delay);
    fprintf('最终完成时刻 (从火灾开始算起): %.2f 秒\n', best_solution.t_total);
    fprintf('实际搜救耗时: %.2f 秒\n', best_solution.t_total - best_solution.start_delay);
    fprintf('Responder 1: Rooms %s -> Finish At %.2f s\n', mat2str(best_solution.r1.path), best_solution.r1.time);
    fprintf('Responder 2: Rooms %s -> Finish At %.2f s\n', mat2str(best_solution.r2.path), best_solution.r2.time);
    fprintf('------------------------------------------------\n');
    
    plot_strict_paths(best_solution);
end

% ================= 辅助函数 =================

% 【修改点】：增加 start_time_offset 参数
function [finish_time, path_coords] = simulate_route_strict(room_order, start_exit_name, start_time_offset)
    global EXITS ROOMS SPEED_BASE
    
    current_pos = EXITS.(start_exit_name);
    
    % 【关键修改】：初始时间设为延迟时间
    current_time = start_time_offset;
    
    path_coords = [current_pos];
    
    for i = 1:length(room_order)
        rid = room_order(i);
        target_door = ROOMS(rid).door;
        
        % 1. 走廊移动
        dist_hall = norm(current_pos - target_door);
        current_time = current_time + dist_hall / SPEED_BASE;
        current_pos = target_door;
        path_coords = [path_coords; target_door];
        
        % 2. 房间内搜寻 (此时 current_time 较大，烟雾较浓)
        [sweep_duration, room_pts] = calc_sweep_strict(current_time, rid);
        
        current_time = current_time + sweep_duration;
        path_coords = [path_coords; room_pts];
        
        % 回到门口
        path_coords = [path_coords; target_door];
    end
    
    % 3. 撤离
    d_left = norm(current_pos - EXITS.Left);
    d_right = norm(current_pos - EXITS.Right);
    
    if d_left < d_right
        exit_pos = EXITS.Left;
        evac_time = d_left / SPEED_BASE;
    else
        exit_pos = EXITS.Right;
        evac_time = d_right / SPEED_BASE;
    end
    
    current_time = current_time + evac_time;
    path_coords = [path_coords; exit_pos];
    
    finish_time = current_time;
end


function [duration, path_points] = calc_sweep_strict(arrival_time, room_id)
    global ROOM_W ROOM_H SPEED_BASE R_MAX R_MIN LAMBDA EFFICIENCY ROOMS
    
    % 计算到达时刻的 R(t)
    R_curr = (R_MAX - R_MIN) * exp(-LAMBDA * arrival_time) + R_MIN;
    
    % 计算速度惩罚
    if R_curr >= 2.0
        current_speed = SPEED_BASE;
    else
        current_speed = 0.2 + 0.8 * (R_curr - 0.5) / 1.5;
    end
    
    rx = ROOMS(room_id).rect(1);
    ry = ROOMS(room_id).rect(2);
    door = ROOMS(room_id).door;
    
    c1 = [rx, ry]; c2 = [rx+ROOM_W, ry]; c3 = [rx+ROOM_W, ry+ROOM_H]; c4 = [rx, ry+ROOM_H];
    corners = [c1; c2; c3; c4];
    
    path_points = [door]; 
    dist_accum = 0;
    
    % 路径生成逻辑 (根据 R_curr 决定密度)
    if R_curr > 2.5
        dists = sum((corners - door).^2, 2);
        [~, start_idx] = min(dists);
        corner_order = circshift(1:4, -(start_idx-1)); 
        
        prev_pt = door;
        for idx = corner_order
            curr_pt = corners(idx, :);
            center = [rx + ROOM_W/2, ry + ROOM_H/2];
            pt_adjusted = center + (curr_pt - center) * 0.9; 
            path_points = [path_points; pt_adjusted];
            dist_accum = dist_accum + norm(pt_adjusted - prev_pt);
            prev_pt = pt_adjusted;
        end
        dist_accum = dist_accum + norm(prev_pt - door);
        path_points = [path_points; door];
        duration = dist_accum / current_speed;
    else
        step_size = max(0.6, R_curr * 1.4); 
        y_levels = (ry + 0.5) : step_size : (ry + ROOM_H - 0.5);
        if isempty(y_levels), y_levels = [ry + ROOM_H/2]; end
        if y_levels(end) < (ry + ROOM_H - 1.0), y_levels = [y_levels, ry + ROOM_H - 0.5]; end
        
        prev_pt = door;
        for k = 1:length(y_levels)
            y = y_levels(k);
            x_left = rx + 0.5; x_right = rx + ROOM_W - 0.5;
            if mod(k, 2) == 1
                pts = [x_left, y; x_right, y];
            else
                pts = [x_right, y; x_left, y];
            end
            path_points = [path_points; pts];
            dist_accum = dist_accum + norm(pts(1,:) - prev_pt);
            dist_accum = dist_accum + norm(pts(2,:) - pts(1,:));
            prev_pt = pts(2,:);
        end
        dist_accum = dist_accum + norm(prev_pt - door);
        path_points = [path_points; door];
        duration = (dist_accum / current_speed) * (1/EFFICIENCY);
    end
end

function plot_strict_paths(sol)
    global ROOMS EXITS
    figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
    hold on; axis equal;
    
    rectangle('Position', [0, 5, 18, 3], 'FaceColor', [0.96, 0.96, 0.96], 'EdgeColor', 'k');
    text(9, 6.5, 'HALLWAY', 'HorizontalAlignment', 'center', 'Color', [0.6,0.6,0.6], 'FontSize', 14);
    
    for i = 1:6
        r = ROOMS(i).rect;
        rectangle('Position', r, 'FaceColor', [0.9, 0.9, 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
        text(r(1)+r(3)/2, r(2)+r(4)/2, ROOMS(i).label, 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        d = ROOMS(i).door;
        plot(d(1), d(2), 's', 'MarkerSize', 8, 'MarkerFaceColor', [0.8, 0.4, 0.4], 'MarkerEdgeColor', 'none');
    end
    
    plot(EXITS.Left(1), EXITS.Left(2), 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
    plot(EXITS.Right(1), EXITS.Right(2), 'p', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
    
    p1 = sol.r1.trace;
    plot(p1(:,1), p1(:,2)+0.05, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 6);
    
    p2 = sol.r2.trace;
    plot(p2(:,1), p2(:,2)-0.05, 'r.-', 'LineWidth', 1.5, 'MarkerSize', 6);
    
    title(sprintf('Start Delay: %ds | Finish Time: %.1fs', sol.start_delay, sol.t_total), 'FontSize', 14);
    legend({'Res 1', 'Res 2'}, 'Location', 'best');
    xlabel('X (m)'); ylabel('Y (m)');
    xlim([-2, 20]); ylim([-1, 14]);
    box on; hold off;
end