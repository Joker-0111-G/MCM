function evacuation_optimizer_multi_lambda()
    % ================= 1. 全局参数配置 =================
    clc; clear; close all;
    
    global ROOMS STAIRS EXITS PARAMS;
    
    % --- 迭代设定 ---
    MIN_N = 1;      
    MAX_N = 6;      % 建议范围
    ITER_PER_N = 3000; 
    
    % --- 物理与环境参数 ---
    PARAMS.SPEED_H = 1.0;          % 水平速度
    PARAMS.SPEED_V = 1.0;          % 垂直速度
    PARAMS.FLOOR_H = 2.5;          % 层高
    PARAMS.R_MAX = sqrt(6^2 + 9^2);% 最大可能视野 (10.8m)
    PARAMS.R_MIN = 0.5;            % 最小视野
    PARAMS.EFFICIENCY = 0.3;       % 烟雾中搜索效率

    % --- 【用户设定区：时间延迟】 ---
    PARAMS.START_DELAY = 100; % 0=立即进入, 300=5分钟后进入
    
    % --- 建筑节点 ---
    STAIRS.LOC = [10.5, 7.5];      % 楼梯中心
    EXITS.Starts = [
        9.0,  6.0, 0; 12.0, 6.0, 0;
        9.0,  9.0, 0; 12.0, 9.0, 0
    ];

    % --- 构建房间 (在此处定义不同房间的 Lambda) ---
    ROOMS = build_complex_building();
    
    fprintf('=== 自动寻优：多级烟雾扩散模型 ===\n');
    fprintf('大房间临界时间: 10分钟 | 小房间临界时间: 6分钟\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('| 人数 (N) |  最短耗时 (s)  |  效率提升 (vs N-1) | 计算耗时 (s) |\n');
    fprintf('------------------------------------------------------------\n');

    % ================= 2. 循环寻优 =================
    
    history = struct('n', {}, 'time', {}, 'sol', {});
    global_best_sol = struct('time', inf, 'n', 0);
    
    for n = MIN_N : MAX_N
        t_start = tic;
        [best_time_n, best_sol_n] = solve_for_n(n, ITER_PER_N);
        t_calc = toc(t_start);
        
        improvement = 0;
        if n > MIN_N, improvement = history(n-MIN_N).time - best_time_n; end
        
        if n == MIN_N, imp_str = '-'; else, imp_str = sprintf('%.1f s', improvement); end
        fprintf('|    %d     |    %7.2f     |    %12s    |    %6.2f    |\n', ...
            n, best_time_n, imp_str, t_calc);
        
        history(n).n = n;
        history(n).time = best_time_n;
        history(n).sol = best_sol_n;
        
        if best_time_n < global_best_sol.time
            global_best_sol = best_sol_n;
            global_best_sol.n = n;
        end
    end
    fprintf('------------------------------------------------------------\n');

    % ================= 3. 绘图与输出 =================
    
    % 3.1 曲线图
    figure('Color','w', 'Position', [50, 50, 600, 400]);
    n_vals = [history.n]; t_vals = [history.time];
    plot(n_vals, t_vals, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b'); hold on;
    
    y_range = max(t_vals) - min(t_vals); offset = y_range * 0.05;
    for i = 1:length(n_vals)
        text(n_vals(i), t_vals(i) + offset, sprintf('%.1f s', t_vals(i)), ...
            'Horiz', 'center', 'Vert', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
    end
    ylim([min(t_vals) - offset*2, max(t_vals) + offset*3]);
    xlabel('响应人员数量 (N)'); ylabel('时间 (s)');
    title(sprintf('Sensitivity Analysis (Delay=%ds)', PARAMS.START_DELAY)); grid on;
    
    % 3.2 详情
    fprintf('\n>>> 全局最优方案详情 (N=%d) <<<\n', global_best_sol.n);
    fprintf('总耗时: %.2f 秒\n', global_best_sol.time);
    for r = 1:global_best_sol.n
        res = global_best_sol.responders(r);
        fprintf('  队员 %d: %s -> %.1fs\n', r, mat2str(res.path), res.finish_time);
    end
    
    plot_3d_building(global_best_sol, sprintf('Optimal Path (N=%d)', global_best_sol.n));
    plot_2d_floor_plans(global_best_sol);
end

% ================= 核心求解器 =================

function [best_time, best_sol] = solve_for_n(num_responders, max_iter)
    global ROOMS EXITS PARAMS
    PARAMS.NUM_RESPONDERS = num_responders;
    best_time = inf; best_sol = [];
    all_rooms = 1:length(ROOMS);
    
    for iter = 1:max_iter
        perm_seq = all_rooms(randperm(length(all_rooms)));
        responders = repmat(struct('time', PARAMS.START_DELAY, 'pos', [], 'path', [], 'trace', [], 'finish_time', 0), num_responders, 1);
        for r = 1:num_responders
            start_node = EXITS.Starts(randi(4), :);
            responders(r).pos = start_node; responders(r).trace = [start_node];
        end
        
        temp_res = responders;
        for i = 1:length(perm_seq)
            rid = perm_seq(i);
            costs = zeros(1, num_responders);
            for r = 1:num_responders
                [c, ~] = simulate_step(temp_res(r).pos, temp_res(r).time, rid, false);
                costs(r) = temp_res(r).time + c;
            end
            [~, best_r] = min(costs);
            [cost, new_pos, trace_step] = simulate_step(temp_res(best_r).pos, temp_res(best_r).time, rid, true);
            temp_res(best_r).time = temp_res(best_r).time + cost;
            temp_res(best_r).pos = new_pos;
            temp_res(best_r).path = [temp_res(best_r).path, rid];
            temp_res(best_r).trace = [temp_res(best_r).trace; trace_step];
        end
        
        final_times = zeros(1, num_responders);
        for r = 1:num_responders
            [t_evac, trace_evac] = calculate_evacuation(temp_res(r).pos);
            final_times(r) = temp_res(r).time + t_evac;
            temp_res(r).trace = [temp_res(r).trace; trace_evac];
            temp_res(r).finish_time = final_times(r);
        end
        
        makespan = max(final_times);
        if makespan < best_time
            best_time = makespan; best_sol.responders = temp_res; best_sol.time = best_time;
        end
    end
end

function [cost_time, end_pos, trace] = simulate_step(start_pos, start_time, rid, gen_trace)
    global ROOMS STAIRS PARAMS
    curr_pos = start_pos; curr_time = start_time; trace = [];
    target_room = ROOMS(rid); target_door = target_room.door;
    
    if abs(curr_pos(3) - target_door(3)) > 0.1
        stair_curr = [STAIRS.LOC, curr_pos(3)];
        d1 = norm(curr_pos(1:2) - stair_curr(1:2));
        curr_time = curr_time + d1 / PARAMS.SPEED_H;
        if gen_trace, trace = [trace; stair_curr]; end
        stair_next = [STAIRS.LOC, target_door(3)];
        curr_time = curr_time + PARAMS.FLOOR_H / PARAMS.SPEED_V;
        if gen_trace, trace = [trace; stair_next]; end
        curr_pos = stair_next;
    end
    
    d2 = norm(curr_pos(1:2) - target_door(1:2));
    curr_time = curr_time + d2 / PARAMS.SPEED_H;
    curr_pos = target_door;
    if gen_trace, trace = [trace; curr_pos]; end
    
    [sweep_dur, sweep_pts] = calc_sweep_time(curr_time, target_room);
    curr_time = curr_time + sweep_dur;
    if gen_trace, trace = [trace; sweep_pts; target_door]; end
    
    cost_time = curr_time - start_time; end_pos = target_door;
end

function [t_evac, trace] = calculate_evacuation(curr_pos)
    global STAIRS PARAMS EXITS
    t_evac = 0; trace = [];
    if curr_pos(3) > 0.1
        stair_curr = [STAIRS.LOC, curr_pos(3)];
        t_evac = t_evac + norm(curr_pos(1:2)-stair_curr(1:2))/PARAMS.SPEED_H;
        trace = [trace; stair_curr];
        stair_ground = [STAIRS.LOC, 0];
        t_evac = t_evac + PARAMS.FLOOR_H/PARAMS.SPEED_V;
        trace = [trace; stair_ground];
        curr_pos = stair_ground;
    end
    dists = sum((EXITS.Starts - curr_pos).^2, 2); [~, idx] = min(dists); target = EXITS.Starts(idx, :);
    t_evac = t_evac + norm(curr_pos(1:2)-target(1:2))/PARAMS.SPEED_H;
    trace = [trace; target];
end

% 【修改点】计算搜寻时间时使用房间专属的 Lambda
function [duration, pts] = calc_sweep_time(arrival_time, room)
    global PARAMS
    
    % 使用 room.lambda 而不是 PARAMS.LAMBDA
    R_curr = (PARAMS.R_MAX - PARAMS.R_MIN) * exp(-room.lambda * arrival_time) + PARAMS.R_MIN;
    
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

% 【修改点】自动根据房间尺寸分配 Lambda
function R = build_complex_building()
    % 定义两种 Lambda
    % 大房间 10min = 600s -> lambda ~ 0.005
    lambda_large = 3 / (10 * 60);
    % 小房间 6min = 360s -> lambda ~ 0.0083
    lambda_small = 3 / (6 * 60);

    R = []; cnt = 0; z=2.5;
    % F2 (6个小，1个大)
    % F2-R1 (小)
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R1'; r.rect=[0, 9, 4.5, 6]; r.door=[0.5, 9, z]; 
    r.lambda = lambda_small; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R2'; r.rect=[4.5, 9, 4.5, 6]; r.door=[6.75, 9, z]; 
    r.lambda = lambda_small; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R3'; r.rect=[12, 9, 4.5, 6]; r.door=[14.25, 9, z]; 
    r.lambda = lambda_small; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R4'; r.rect=[16.5, 9, 4.5, 6]; r.door=[18.75, 9, z]; 
    r.lambda = lambda_small; R=[R, r];
    
    % F2-Big (大) 宽9m
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-Big'; r.rect=[0, 0, 9, 6]; r.door=[8.5, 6, z]; 
    r.lambda = lambda_large; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R6'; r.rect=[12, 0, 4.5, 6]; r.door=[14.25, 6, z]; 
    r.lambda = lambda_small; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F2-R7'; r.rect=[16.5, 0, 4.5, 6]; r.door=[18.75, 6, z]; 
    r.lambda = lambda_small; R=[R, r];
    
    z=0;
    % F1 (4个大) 宽9m
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-L-Top'; r.rect=[0, 9, 9, 6]; r.door=[8.5, 9, z]; 
    r.lambda = lambda_large; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-R-Top'; r.rect=[12, 9, 9, 6]; r.door=[12.5, 9, z]; 
    r.lambda = lambda_large; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-L-Bot'; r.rect=[0, 0, 9, 6]; r.door=[8.5, 6, z]; 
    r.lambda = lambda_large; R=[R, r];
    
    cnt=cnt+1; r.id=cnt; r.floor_z=z; r.label='F1-R-Bot'; r.rect=[12, 0, 9, 6]; r.door=[12.5, 6, z]; 
    r.lambda = lambda_large; R=[R, r];
end

function plot_3d_building(sol, title_str)
    global ROOMS
    figure('Color','w','Position',[600,100,800,600]); hold on; axis equal; view(3); grid on;
    for i = 1:length(ROOMS)
        r = ROOMS(i);
        patch([r.rect(1), r.rect(1)+r.rect(3), r.rect(1)+r.rect(3), r.rect(1)], ...
              [r.rect(2), r.rect(2), r.rect(2)+r.rect(4), r.rect(2)+r.rect(4)], ...
              [r.floor_z, r.floor_z, r.floor_z, r.floor_z], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5);
        plot3(r.door(1), r.door(2), r.door(3), 'ro', 'MarkerSize', 4, 'LineWidth', 2);
    end
    colors = lines(length(sol.responders));
    for k = 1:length(sol.responders)
        trace = sol.responders(k).trace;
        if ~isempty(trace)
            plot3(trace(:,1), trace(:,2), trace(:,3), '.-', 'LineWidth', 1.5, 'Color', colors(k,:));
        end
    end
    title(title_str); xlabel('X'); ylabel('Y'); zlabel('Floor');
end

function plot_2d_floor_plans(sol)
    global ROOMS STAIRS
    figure('Color','w','Position',[100,100,800,900]);
    subplot(2,1,1); hold on; axis equal; grid on; title('Floor 2'); xlabel('X'); ylabel('Y');
    rectangle('Position',[-2,-2,25,20], 'EdgeColor','none', 'FaceColor',[0.98 0.98 0.98]);
    for i = 1:length(ROOMS)
        r = ROOMS(i);
        if r.floor_z > 1
            rectangle('Position', r.rect, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
            text(r.rect(1)+r.rect(3)/2, r.rect(2)+r.rect(4)/2, r.label, 'Horiz', 'center');
            plot(r.door(1), r.door(2), 'ro', 'MarkerSize', 6, 'LineWidth', 2, 'MarkerFaceColor', 'w');
        end
    end
    plot(STAIRS.LOC(1), STAIRS.LOC(2), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
    colors = lines(length(sol.responders));
    for k = 1:length(sol.responders)
        trace = sol.responders(k).trace;
        if ~isempty(trace)
            idx = abs(trace(:,3) - 2.5) < 0.1;
            x_plot = trace(:,1); y_plot = trace(:,2);
            x_plot(~idx) = NaN; y_plot(~idx) = NaN; 
            plot(x_plot, y_plot, '.-', 'LineWidth', 1.5, 'Color', colors(k,:));
        end
    end
    xlim([-1, 22]); ylim([-1, 16]);

    subplot(2,1,2); hold on; axis equal; grid on; title('Floor 1'); xlabel('X'); ylabel('Y');
    rectangle('Position',[-2,-2,25,20], 'EdgeColor','none', 'FaceColor',[0.98 0.98 0.98]); 
    for i = 1:length(ROOMS)
        r = ROOMS(i);
        if r.floor_z < 1
            rectangle('Position', r.rect, 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
            text(r.rect(1)+r.rect(3)/2, r.rect(2)+r.rect(4)/2, r.label, 'Horiz', 'center');
            plot(r.door(1), r.door(2), 'ro', 'MarkerSize', 6, 'LineWidth', 2, 'MarkerFaceColor', 'w');
        end
    end
    plot(STAIRS.LOC(1), STAIRS.LOC(2), 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'y');
    for k = 1:length(sol.responders)
        trace = sol.responders(k).trace;
        if ~isempty(trace)
            idx = abs(trace(:,3) - 0) < 0.1;
            x_plot = trace(:,1); y_plot = trace(:,2);
            x_plot(~idx) = NaN; y_plot(~idx) = NaN;
            plot(x_plot, y_plot, '.-', 'LineWidth', 1.5, 'Color', colors(k,:));
        end
    end
    xlim([-1, 22]); ylim([-1, 16]);
end