function evacuation_two_stairs_solver_labeled()
    % ================= 1. 参数配置 =================
    clc; clear; close all;
    
    global ROOMS STAIRS EXITS PARAMS;
    
    % --- 自动寻优设定 ---
    MIN_N = 1;      % 最少人数
    MAX_N = 4;      % 最多人数
    
    % --- 物理参数 ---
    PARAMS.SPEED_H = 1.0;          % 水平速度 (m/s)
    PARAMS.SPEED_V = 1.0;          % 垂直上下楼速度 (m/s)
    PARAMS.FLOOR_H = 2.5;          % 层高 (m)
    PARAMS.R_MAX = sqrt(6^2 + 5^2);% 最大视野 (~7.81m)
    PARAMS.R_MIN = 0.5;            % 最小视野
    T_CRIT = 6 * 60;               % 6分钟临界时间
    PARAMS.LAMBDA = 3 / T_CRIT;
    PARAMS.EFFICIENCY = 0.3;       % 搜寻效率
    
    % --- 【用户设定：起始延迟】 ---
    PARAMS.START_DELAY = 100; 
    
    % --- 建筑节点 (双楼梯) ---
    % 楼梯位于走廊两端 (X=0 和 X=18, Y=6.5)
    STAIRS.LOCS = [
        0.0,  6.5;  % 左楼梯
        18.0, 6.5   % 右楼梯
    ];
    
    % 初始位置：一层走廊两端
    EXITS.Starts = [
        0.0,  6.5, 0;  % 左入口 (F1)
        18.0, 6.5, 0   % 右入口 (F1)
    ];

    % --- 构建房间数据 ---
    ROOMS = build_two_floor_building();
    
    fprintf('=== 双层双楼梯搜救优化 (带数值标注版) ===\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('| 人数 (N) |  最短耗时 (s)  |  效率提升 (vs N-1) |\n');
    fprintf('------------------------------------------------------------\n');

    % ================= 2. 循环寻优 =================
    
    history = struct('n', {}, 'time', {}, 'sol', {});
    global_best_sol = struct('time', inf, 'n', 0);
    
    for n = MIN_N : MAX_N
        % 求解
        [best_time_n, best_sol_n] = solve_optimal_strategy(n);
        
        % 记录与输出
        improvement = 0;
        if n > MIN_N, improvement = history(n-MIN_N).time - best_time_n; end
        if n == MIN_N, imp_str = '-'; else, imp_str = sprintf('%.1f s', improvement); end
        
        fprintf('|    %d     |    %7.2f     |    %12s    |\n', n, best_time_n, imp_str);
        
        history(n).n = n; history(n).time = best_time_n; history(n).sol = best_sol_n;
        
        if best_time_n < global_best_sol.time
            global_best_sol = best_sol_n; global_best_sol.n = n;
        end
    end
    fprintf('------------------------------------------------------------\n');

    % ================= 3. 绘图 (主要修改部分) =================
    
    % 3.1 曲线图 (带数值标注)
    figure('Color','w', 'Position', [50, 50, 600, 400]);
    n_vals = [history.n];
    t_vals = [history.time];
    
    plot(n_vals, t_vals, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    hold on;
    
    % 添加数值标签
    y_range = max(t_vals) - min(t_vals);
    if y_range == 0, y_range = 10; end % 防止除零
    offset = y_range * 0.08; % 标签向上偏移量
    
    for i = 1:length(n_vals)
        text(n_vals(i), t_vals(i) + offset, sprintf('%.1f s', t_vals(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 10, 'FontWeight', 'bold', 'Color', 'k');
    end
    
    % 调整Y轴范围以防文字被遮挡
    ylim([min(t_vals) - offset*2, max(t_vals) + offset*3]);
    
    xlabel('人数 (N)'); ylabel('时间 (s)'); 
    title('搜救时间 vs 人数 (双楼梯)'); 
    grid on;
    
    % 3.2 详情
    fprintf('\n>>> 最优方案 (N=%d) <<<\n', global_best_sol.n);
    for r = 1:global_best_sol.n
        res = global_best_sol.responders(r);
        fprintf('  队员 %d: 路线 %s -> %.1fs\n', r, mat2str(res.path), res.finish_time);
    end
    
    % 3.3 3D图 (保持原样)
    plot_3d_building(global_best_sol);
    
    % 3.4 2D图 (保持原样)
    plot_2d_plans(global_best_sol);
end

% ================= 核心求解算法 (暴力全排列) =================

function [best_time, best_sol] = solve_optimal_strategy(num_responders)
    global ROOMS EXITS PARAMS
    PARAMS.NUM_RESPONDERS = num_responders;
    
    best_time = inf; best_sol = [];
    all_rooms = 1:length(ROOMS);
    perms_list = perms(all_rooms); % 6! = 720
    [n_perms, ~] = size(perms_list);
    
    for k = 1:n_perms
        room_seq = perms_list(k, :);
        
        % 多次尝试随机初始位置
        for try_idx = 1:30
            responders = repmat(struct('time', PARAMS.START_DELAY, 'pos', [], 'path', [], 'trace', [], 'finish_time', 0), num_responders, 1);
            for r = 1:num_responders
                start_idx = randi(2); % 1=左, 2=右
                responders(r).pos = EXITS.Starts(start_idx, :);
                responders(r).trace = [responders(r).pos];
            end
            
            temp_res = responders;
            for i = 1:length(room_seq)
                rid = room_seq(i);
                costs = zeros(1, num_responders);
                % 预估谁去最快
                for r = 1:num_responders
                    [c, ~] = simulate_step(temp_res(r).pos, temp_res(r).time, rid, false);
                    costs(r) = temp_res(r).time + c;
                end
                [~, best_r] = min(costs);
                
                % 执行移动
                [cost, new_pos, trace] = simulate_step(temp_res(best_r).pos, temp_res(best_r).time, rid, true);
                temp_res(best_r).time = temp_res(best_r).time + cost;
                temp_res(best_r).pos = new_pos;
                temp_res(best_r).path = [temp_res(best_r).path, rid];
                temp_res(best_r).trace = [temp_res(best_r).trace; trace];
            end
            
            % 撤离
            final_times = zeros(1, num_responders);
            for r = 1:num_responders
                [t_evac, trace_evac] = calculate_evacuation(temp_res(r).pos);
                final_times(r) = temp_res(r).time + t_evac;
                temp_res(r).trace = [temp_res(r).trace; trace_evac];
                temp_res(r).finish_time = final_times(r);
            end
            
            makespan = max(final_times);
            if makespan < best_time
                best_time = makespan;
                best_sol.responders = temp_res;
                best_sol.time = best_time;
            end
        end
    end
end

% ================= 物理模拟 =================

function [cost_time, end_pos, trace] = simulate_step(start_pos, start_time, rid, gen_trace)
    global ROOMS STAIRS PARAMS
    curr_pos = start_pos; curr_time = start_time; trace = [];
    target_room = ROOMS(rid); target_door = target_room.door;
    
    % --- 1. 移动逻辑 ---
    if abs(curr_pos(3) - target_door(3)) > 0.1
        % 跨层：比较左右楼梯
        stair_L_curr = [STAIRS.LOCS(1,:), curr_pos(3)];
        stair_L_next = [STAIRS.LOCS(1,:), target_door(3)];
        dist_A = norm(curr_pos(1:2) - stair_L_curr(1:2)) + norm(stair_L_next(1:2) - target_door(1:2));
        
        stair_R_curr = [STAIRS.LOCS(2,:), curr_pos(3)];
        stair_R_next = [STAIRS.LOCS(2,:), target_door(3)];
        dist_B = norm(curr_pos(1:2) - stair_R_curr(1:2)) + norm(stair_R_next(1:2) - target_door(1:2));
        
        if dist_A < dist_B
            chosen_stair_curr = stair_L_curr;
            chosen_stair_next = stair_L_next;
        else
            chosen_stair_curr = stair_R_curr;
            chosen_stair_next = stair_R_next;
        end
        
        curr_time = curr_time + norm(curr_pos(1:2)-chosen_stair_curr(1:2))/PARAMS.SPEED_H;
        if gen_trace, trace = [trace; chosen_stair_curr]; end
        
        curr_time = curr_time + PARAMS.FLOOR_H / PARAMS.SPEED_V; 
        if gen_trace, trace = [trace; chosen_stair_next]; end
        
        curr_time = curr_time + norm(chosen_stair_next(1:2)-target_door(1:2))/PARAMS.SPEED_H;
        if gen_trace, trace = [trace; target_door]; end
        
        curr_pos = target_door;
    else
        % 同层
        dist = norm(curr_pos(1:2) - target_door(1:2));
        curr_time = curr_time + dist / PARAMS.SPEED_H;
        curr_pos = target_door;
        if gen_trace, trace = [trace; curr_pos]; end
    end
    
    % --- 2. 搜寻 ---
    [sweep_dur, sweep_pts] = calc_sweep_time(curr_time, target_room);
    curr_time = curr_time + sweep_dur;
    if gen_trace, trace = [trace; sweep_pts; target_door]; end
    
    cost_time = curr_time - start_time; end_pos = target_door;
end

function [t_evac, trace] = calculate_evacuation(curr_pos)
    global STAIRS PARAMS EXITS
    t_evac = 0; trace = [];
    
    % 若在二楼，先下楼
    if curr_pos(3) > 0.1
        d_L = norm(curr_pos(1:2) - STAIRS.LOCS(1,:));
        d_R = norm(curr_pos(1:2) - STAIRS.LOCS(2,:));
        
        if d_L < d_R
            stair_curr = [STAIRS.LOCS(1,:), curr_pos(3)];
            stair_ground = [STAIRS.LOCS(1,:), 0];
        else
            stair_curr = [STAIRS.LOCS(2,:), curr_pos(3)];
            stair_ground = [STAIRS.LOCS(2,:), 0];
        end
        
        t_evac = t_evac + norm(curr_pos(1:2)-stair_curr(1:2))/PARAMS.SPEED_H;
        trace = [trace; stair_curr];
        
        t_evac = t_evac + PARAMS.FLOOR_H/PARAMS.SPEED_V;
        trace = [trace; stair_ground];
        curr_pos = stair_ground;
    end
    
    % 撤离到最近出口
    d_exit1 = norm(curr_pos(1:2) - EXITS.Starts(1,1:2));
    d_exit2 = norm(curr_pos(1:2) - EXITS.Starts(2,1:2));
    
    if d_exit1 < d_exit2
        target = EXITS.Starts(1,:); t = d_exit1 / PARAMS.SPEED_H;
    else
        target = EXITS.Starts(2,:); t = d_exit2 / PARAMS.SPEED_H;
    end
    
    t_evac = t_evac + t;
    trace = [trace; target];
end

function [duration, pts] = calc_sweep_time(arrival_time, room)
    global PARAMS
    R_curr = (PARAMS.R_MAX - PARAMS.R_MIN) * exp(-PARAMS.LAMBDA * arrival_time) + PARAMS.R_MIN;
    if R_curr >= 2.0, spd = PARAMS.SPEED_H; else, spd = 0.2 + 0.8*(R_curr - 0.5)/1.5; end
    
    rx=room.rect(1); ry=room.rect(2); rw=room.rect(3); rh=room.rect(4); z=room.floor_z; door=room.door(1:2);
    pts=[];
    
    if R_curr > 2.5
        corners = [rx,ry; rx+rw,ry; rx+rw,ry+rh; rx,ry+rh];
        dists = sum((corners-door).^2,2); [~,idx]=min(dists); order=circshift(1:4, -(idx-1));
        prev=door; dist=0;
        for i=order
            p=corners(i,:); center=[rx+rw/2, ry+rh/2]; p_adj=center+(p-center)*0.9;
            pts=[pts; p_adj, z]; dist=dist+norm(p_adj-prev); prev=p_adj;
        end
        dist=dist+norm(prev-door); duration=dist/spd;
    else
        step=max(0.6, R_curr*1.4); y_vals=ry+0.5:step:ry+rh-0.5;
        prev=door; dist=0;
        for k=1:length(y_vals)
            y=y_vals(k); if mod(k,2)==1, xs=[rx+0.5, rx+rw-0.5]; else, xs=[rx+rw-0.5, rx+0.5]; end
            for x=xs, p=[x,y]; pts=[pts; p,z]; dist=dist+norm(p-prev); prev=p; end
        end
        dist=dist+norm(prev-door); duration=(dist/spd)/PARAMS.EFFICIENCY;
    end
end

% ================= 建筑几何构建 =================

function R = build_two_floor_building()
    R = []; cnt = 0;
    z = 2.5;
    % F2
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R1'; r.rect=[0, 0, 6, 5]; r.door=[5.5, 5, z]; R=[R, r];
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R2'; r.rect=[6, 0, 6, 5]; r.door=[9.0, 5, z]; R=[R, r];
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R3'; r.rect=[12, 0, 6, 5]; r.door=[12.5, 5, z]; R=[R, r];
    z = 0;
    % F1
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-R1'; r.rect=[0, 0, 6, 5]; r.door=[5.5, 5, z]; R=[R, r];
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-R2'; r.rect=[6, 0, 6, 5]; r.door=[9.0, 5, z]; R=[R, r];
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-R3'; r.rect=[12, 0, 6, 5]; r.door=[12.5, 5, z]; R=[R, r];
end

% ================= 绘图 =================

function plot_3d_building(sol)
    global ROOMS
    figure('Color','w'); hold on; axis equal; view(3); grid on;
    for i = 1:length(ROOMS)
        r = ROOMS(i);
        patch([r.rect(1), r.rect(1)+r.rect(3), r.rect(1)+r.rect(3), r.rect(1)], ...
              [r.rect(2), r.rect(2), r.rect(2)+r.rect(4), r.rect(2)+r.rect(4)], ...
              [r.floor_z, r.floor_z, r.floor_z, r.floor_z], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5);
        plot3(r.door(1), r.door(2), r.door(3), 'ro', 'MarkerSize', 5, 'LineWidth', 2);
    end
    colors = lines(length(sol.responders));
    for k = 1:length(sol.responders)
        trace = sol.responders(k).trace;
        if ~isempty(trace)
            plot3(trace(:,1), trace(:,2), trace(:,3), '.-', 'LineWidth', 1.5, 'Color', colors(k,:));
        end
    end
    title(sprintf('3D Path (Time: %.1fs)', sol.time)); xlabel('X'); ylabel('Y'); zlabel('Floor');
end

function plot_2d_plans(sol)
    global ROOMS STAIRS
    figure('Color','w','Position',[100,100,500,600]);
    
    % F2
    subplot(2,1,1); hold on; axis equal; grid on; title('Floor 2 (Up)');
    rectangle('Position',[-2,-2,22,10], 'EdgeColor','none', 'FaceColor',[0.98 0.98 0.98]);
    for i = 1:3
        r = ROOMS(i); rectangle('Position', r.rect, 'FaceColor', [0.9 0.9 0.9]);
        plot(r.door(1), r.door(2), 'ro', 'MarkerFaceColor','w');
    end
    plot(STAIRS.LOCS(1,1), STAIRS.LOCS(1,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','y');
    plot(STAIRS.LOCS(2,1), STAIRS.LOCS(2,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','y');
    
    colors = lines(length(sol.responders));
    for k = 1:length(sol.responders)
        trace = sol.responders(k).trace;
        if ~isempty(trace)
            idx = abs(trace(:,3) - 2.5) < 0.1;
            x=trace(:,1); y=trace(:,2); x(~idx)=NaN; y(~idx)=NaN;
            plot(x, y, '.-', 'LineWidth', 1.5, 'Color', colors(k,:));
        end
    end
    xlim([-1, 19]); ylim([-1, 9]);
    
    % F1
    subplot(2,1,2); hold on; axis equal; grid on; title('Floor 1 (Down)');
    rectangle('Position',[-2,-2,22,10], 'EdgeColor','none', 'FaceColor',[0.98 0.98 0.98]);
    for i = 4:6
        r = ROOMS(i); rectangle('Position', r.rect, 'FaceColor', [0.9 0.9 0.9]);
        plot(r.door(1), r.door(2), 'ro', 'MarkerFaceColor','w');
    end
    plot(STAIRS.LOCS(1,1), STAIRS.LOCS(1,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','y');
    plot(STAIRS.LOCS(2,1), STAIRS.LOCS(2,2), 'kp', 'MarkerSize',12, 'MarkerFaceColor','y');
    
    for k = 1:length(sol.responders)
        trace = sol.responders(k).trace;
        if ~isempty(trace)
            idx = abs(trace(:,3) - 0) < 0.1;
            x=trace(:,1); y=trace(:,2); x(~idx)=NaN; y(~idx)=NaN;
            plot(x, y, '.-', 'LineWidth', 1.5, 'Color', colors(k,:));
        end
    end
    xlim([-1, 19]); ylim([-1, 9]);
end