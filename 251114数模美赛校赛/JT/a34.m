function search_rescue_sweep()
    clc; clear; close all;

    %% === 1. 全局参数与初始化 ===
    global GRID RES_SCALE MAP_W MAP_H;
    
    % 空间定义
    RES_SCALE = 5; % 5 grid cells per meter (0.2m resolution)
    MAP_W = 85;    % 包含仓库(0-30) + 一层(30-45) + 二层平铺(50-80)
    MAP_H = 25;
    GRID = zeros(ceil(MAP_H * RES_SCALE), ceil(MAP_W * RES_SCALE)); % 0=空, 1=墙/障碍
    
    % 速度参数 (结构体: Responder, Rescuee)
    % 格式: [直线, 转弯, 楼梯]
    VEL.Res = [1.2, 1.0, 1.1]; 
    VEL.Vic = [1.0, 0.8, 0.9]; % Victim (Rescuee)
    
    % 烟雾参数
    SMOKE.T_crit_office = 6 * 60;   % 360s
    SMOKE.T_crit_wh = 30 * 60;      % 1800s
    SMOKE.lambda_office = 3 / SMOKE.T_crit_office;
    SMOKE.lambda_wh = 3 / SMOKE.T_crit_wh;
    SMOKE.R_max_wh = sqrt(28^2 + 22^2);
    SMOKE.R_max_room = sqrt(6^2 + 5^2);
    SMOKE.R_min = 0.5;
    
    fprintf('正在构建复杂障碍物地图 (严格复刻 a31.m)...\n');
    build_environment_strict();
    
    %% === 2. 定义搜索任务与关键点 ===
    % 坐标系统: [x, y]
    % 楼梯接口点 (用于连接两层)
    Stair_L1 = [32.5, 1.5]; % 一层楼梯
    Stair_L2 = [52.5, 1.5]; % 二层楼梯 (平铺坐标)
    
    % 定义需要搜索的区域 (Tasks)
    % Priority: 1=高危(孕妇婴幼儿), 2=普通
    % 关键点生成策略: 房间四角 + 门后 + 障碍物角
    
    Tasks = [];
    
    % --- 任务 1: Apt 1 (孕妇 - Priority 1) ---
    % 位于二层 (平铺X: 50+), y=3~7
    apt1_corners = [50.5, 3.5; 50.5, 6.5; 55.5, 6.5; 55.5, 3.5]; % 房间四角
    apt1_door_back = [56.5, 5.5]; % 门后
    Tasks = add_task(Tasks, 'Apt 1 (Pregnant)', 1, [56.5, 5], [apt1_door_back; apt1_corners], 'office');
    
    % --- 任务 2: Apt 4 (婴幼儿 - Priority 1) ---
    % 位于二层, y=15~19
    apt4_corners = [50.5, 15.5; 50.5, 18.5; 55.5, 18.5; 55.5, 15.5];
    apt4_door_back = [56.5, 17.5]; 
    Tasks = add_task(Tasks, 'Apt 4 (Infant)', 1, [56.5, 17], [apt4_door_back; apt4_corners], 'office');
    
    % --- 任务 3: Warehouse (复杂障碍 - Priority 2) ---
    % 仓库入口: (0, 11) 或 (0, 15) 等
    % 必须搜索: 障碍物角落 (根据 a31.m 障碍物位置手动提取关键探测点)
    wh_entry = [0, 11];
    % 提取的关键检查点 (覆盖所有死角)
    wh_points = [
        1.0, 1.0;   27.0, 1.0;  27.0, 21.0; 1.0, 21.0; % 仓库大四角
        12.0, 18.0; 14.0, 19.5; % 门前阻挡物后
        4.0, 11.0;  10.0, 15.0; % 左侧货架迷宫区
        15.0, 5.0;  20.0, 10.0; 24.0, 10.0; % 中间岛屿与右侧屏风后
        0.5, 0.5;   16.0, 18.0  % 已知人员位置
    ];
    Tasks = add_task(Tasks, 'Warehouse Sweep', 2, wh_entry, wh_points, 'warehouse');
    
    % --- 任务 4: Shop 2 (Priority 2) ---
    % 一层 X=28+, Shop 2 y=3+5.33=8.33
    shop2_entry = [37.5, 8.33 + 2.6];
    shop2_pts = [29.0, 9.0; 29.0, 13.0; 36.0, 13.0; 36.0, 9.0; 30.0, 11.3]; % 角落+人员
    Tasks = add_task(Tasks, 'Shop 2', 2, shop2_entry, shop2_pts, 'office');

    %% === 3. 执行搜救仿真 ===
    % 排序任务: 优先级高的先走
    [~, idx] = sort([Tasks.prio]);
    SortedTasks = Tasks(idx);
    
    current_pos = [0, 11]; % 响应者初始位置 (仓库左门外)
    current_time = 0;      % 累计时间 (秒)
    
    Full_Trace = current_pos;
    fprintf('\n=== 开始搜救任务 (Search Phase) ===\n');
    fprintf('起始时间延迟: 0s (立即进入)\n');
    
    for i = 1:length(SortedTasks)
        T = SortedTasks(i);
        fprintf('>>> [Task %d]前往: %s (优先级 %d)\n', i, T.name, T.prio);
        
        % 1. 移动到目标区域入口 (Move to Entry)
        [path_to_entry, dist_move] = plan_path_smart(current_pos, T.entry, Stair_L1, Stair_L2);
        time_move = calc_movement_time(path_to_entry, VEL.Res, false); % 纯跑动
        
        current_time = current_time + time_move;
        current_pos = T.entry;
        Full_Trace = [Full_Trace; path_to_entry];
        
        % 2. 执行区域扫视 (Sweep Area)
        % 计算当前烟雾视距 R(t)
        if strcmp(T.type, 'warehouse')
            lambda = SMOKE.lambda_wh; Rmax = SMOKE.R_max_wh;
        else
            lambda = SMOKE.lambda_office; Rmax = SMOKE.R_max_room;
        end
        R_curr = (Rmax - SMOKE.R_min) * exp(-lambda * current_time) + SMOKE.R_min;
        
        fprintf('    到达时刻: %.1fs | 视距 R(t): %.2fm\n', current_time, R_curr);
        
        % 生成扫视路径: 遍历该区域所有关键点
        % 策略: 最近邻贪心 (Nearest Neighbor) 访问所有点以覆盖死角
        sweep_path = generate_sweep_path(T.entry, T.points);
        
        % 烟雾惩罚系数: 视距越低，为了看清需要走得越近/越慢，或需要更密的路径
        % 此处简化为: 路径本身已强制经过死角，但移动速度会因小心翼翼而变慢
        vis_factor = min(1.0, max(0.5, R_curr / 3.0)); % 视距<3m时速度受影响
        
        % 判定是否为救援带出阶段 (如果是孕妇/婴儿，找到后需带出)
        % 模拟: 走完所有点(确保清场)后，如果有Target，最后一段路是带人
        time_sweep = calc_movement_time(sweep_path, VEL.Res, false) / vis_factor;
        
        current_time = current_time + time_sweep;
        current_pos = sweep_path(end, :);
        Full_Trace = [Full_Trace; sweep_path];
        
        fprintf('    扫视完成. 耗时: %.1fs. 当前总时间: %.1fs\n', time_sweep, current_time);
        
        % 如果是高危任务，模拟“带人撤离到门口交接”的时间
        if T.prio == 1
            % 假设最后在某一点找到了人，带回门口
            [path_out, ~] = plan_path_smart(current_pos, T.entry, Stair_L1, Stair_L2);
            % 受伤/孕妇速度
            time_extract = calc_movement_time(path_out, VEL.Vic, true); 
            current_time = current_time + time_extract;
            current_pos = T.entry;
            Full_Trace = [Full_Trace; path_out];
            fprintf('    人员(孕妇/婴幼儿)已移交至门口. 额外耗时: %.1fs\n', time_extract);
        end
    end
    
    fprintf('\n=== 所有区域搜索完毕 ===\n');
    fprintf('总耗时: %.2f 秒 (%.2f 分钟)\n', current_time, current_time/60);
    
    %% === 4. 绘图 ===
    plot_results(Full_Trace, SortedTasks, Stair_L1, Stair_L2);
end

%% ================= 辅助函数 =================

function tasks = add_task(tasks, name, prio, entry, pts, type)
    new_t = struct('name', name, 'prio', prio, 'entry', entry, 'points', pts, 'type', type);
    if isempty(tasks)
        tasks = new_t;
    else
        tasks(end+1) = new_t;
    end
end

function path = generate_sweep_path(start_pt, points)
    % 简单的TSP贪心算法：每次去最近的未访问点
    curr = start_pt;
    rem_pts = points;
    path = [];
    
    while ~isempty(rem_pts)
        dists = sum((rem_pts - curr).^2, 2);
        [~, min_idx] = min(dists);
        next_pt = rem_pts(min_idx, :);
        
        % A* 寻路到下一点
        [segment, ~] = a_star_search(curr, next_pt);
        path = [path; segment];
        
        curr = next_pt;
        rem_pts(min_idx, :) = [];
    end
end

function t = calc_movement_time(path, vel_profile, is_stairs_region)
    % vel_profile: [v_straight, v_turn, v_stair]
    if isempty(path), t=0; return; end
    t = 0;
    for k = 2:size(path, 1)
        p_curr = path(k,:);
        p_prev = path(k-1,:);
        dist = norm(p_curr - p_prev);
        
        v = vel_profile(1); % 默认直线
        
        % 检测是否在楼梯区域 (X坐标 30~40 或 50~60 且是特定的楼梯区域)
        % 简单判定: 路径点是否跨层连接线(这里简化处理，统一用直线/转弯判定)
        
        % 转向判定
        if k > 2
            p_pre2 = path(k-2,:);
            vec1 = p_prev - p_pre2;
            vec2 = p_curr - p_prev;
            % 如果向量方向改变(点积<1)，视为转弯
            cos_theta = dot(vec1, vec2) / (norm(vec1)*norm(vec2) + 1e-6);
            if cos_theta < 0.99
                v = vel_profile(2);
            end
        end
        
        t = t + dist / v;
    end
end

function [path, dist] = plan_path_smart(p_start, p_end, s1, s2)
    % 智能判断是否需要走楼梯
    is_f1_s = p_start(1) < 45;
    is_f1_e = p_end(1) < 45;
    
    if is_f1_s == is_f1_e
        [path, dist] = a_star_search(p_start, p_end);
    else
        % 跨层
        if is_f1_s % F1 -> F2
            [p1, ~] = a_star_search(p_start, s1);
            [p2, ~] = a_star_search(s2, p_end);
            path = [p1; p2];
            dist = norm(p_start-s1) + norm(s2-p_end) + 5; % 5m楼梯等效
        else % F2 -> F1
            [p1, ~] = a_star_search(p_start, s2);
            [p2, ~] = a_star_search(s1, p_end);
            path = [p1; p2];
            dist = norm(p_start-s2) + norm(s1-p_end) + 5;
        end
    end
end

function [path, total_dist] = a_star_search(start_p, end_p)
    global GRID RES_SCALE
    
    s_node = max([1,1], min(size(GRID), ceil([start_p(2), start_p(1)] * RES_SCALE)));
    e_node = max([1,1], min(size(GRID), ceil([end_p(2), end_p(1)] * RES_SCALE)));
    
    map = GRID > 0; % 二值化障碍
    
    % 使用 Distance Transform 快速生成无碰撞路径 (模拟 A*)
    % 确保起点终点不在墙内
    if map(s_node(1), s_node(2)), map(s_node(1), s_node(2))=0; end
    if map(e_node(1), e_node(2)), map(e_node(1), e_node(2))=0; end
    
    D = bwdistgeodesic(~map, s_node(2), s_node(1), 'quasi-euclidean');
    
    path_idx = [];
    curr = e_node;
    if isinf(D(curr(1), curr(2)))
        path = [start_p; end_p]; total_dist = norm(start_p-end_p); return;
    end
    
    iter = 0;
    while norm(curr - s_node) > 0 && iter < 5000
        path_idx = [curr; path_idx];
        % 梯度下降找最短路
        min_v = inf; next_n = curr;
        for i=-1:1, for j=-1:1
            if i==0&&j==0, continue; end
            nr = curr(1)+i; nc = curr(2)+j;
            if nr>0 && nr<=size(map,1) && nc>0 && nc<=size(map,2)
                if D(nr,nc) < min_v
                    min_v = D(nr,nc); next_n = [nr,nc];
                end
            end
        end, end
        curr = next_n;
        iter = iter + 1;
    end
    path_idx = [s_node; path_idx];
    
    % 降采样优化路径点
    path_idx = path_idx(1:2:end, :);
    path = [path_idx(:,2), path_idx(:,1)] / RES_SCALE;
    total_dist = sum(sqrt(sum(diff(path).^2,2)));
end

function build_environment_strict()
    global GRID RES_SCALE
    fill_rect = @(x,y,w,h) set_grid(x,y,w,h,1);
    
    % === 1. 仓库 (0-28) ===
    % 墙体
    fill_rect(0,0,28,0.5); fill_rect(0,21.5,28,0.5); % 上下
    fill_rect(0,0,0.5,7); fill_rect(0,15,0.5,7); % 左墙开门
    fill_rect(27.5,0,0.5,22); % 右墙
    
    % 内部障碍 (复刻 a31.m)
    fill_rect(3, 16, 8, 2); fill_rect(3, 12, 2, 4); fill_rect(6, 13, 2, 2); % 倒L组
    fill_rect(3, 4, 8, 3); fill_rect(3, 8, 3, 2); % 迷宫座
    fill_rect(13, 14, 2, 4); fill_rect(13, 4, 2, 4); % 岛
    fill_rect(17, 10, 2, 6); fill_rect(17, 17, 2, 2); fill_rect(17, 2, 2, 2); % 中轴
    fill_rect(22, 5, 2, 12); % 右屏风
    fill_rect(13, 19, 4, 1); % 上门挡
    fill_rect(25, 18, 1, 3); % 楼梯前挡
    
    % === 2. 一层商铺与楼梯 (28-40) ===
    fill_rect(28,0,9,3); fill_rect(28,19,9,3); % 楼梯实体
    shop_h = 16/3;
    for i=1:3, y=3+(i-1)*shop_h; fill_rect(28,y,9,0.2); fill_rect(28,y+shop_h,9,0.2); end
    fill_rect(37, 3, 0.2, 16); % 商铺前脸
    
    % === 3. 二层公寓 (50-80, 平移后的 28-58) ===
    base_x = 50;
    fill_rect(base_x,0,9,3); fill_rect(base_x,19,9,3); % 二层楼梯井
    for i=1:4
        y=3+(i-1)*4;
        fill_rect(base_x,y,6,0.2); fill_rect(base_x,y+4,6,0.2); % 隔墙
        fill_rect(base_x,y,0.2,4); fill_rect(base_x+6,y,0.2,4); % 前后墙
    end
end

function set_grid(x,y,w,h,val)
    global GRID RES_SCALE
    c = max(1, ceil([x, x+w]*RES_SCALE));
    r = max(1, ceil([y, y+h]*RES_SCALE));
    GRID(r(1):r(2), c(1):c(2)) = val;
end

function plot_results(path, tasks, s1, s2)
    global GRID RES_SCALE
    figure('Color','w','Position',[50,50,1200,400]); hold on; axis equal;
    
    % 绘制障碍
    [r,c] = find(GRID);
    plot(c/RES_SCALE, r/RES_SCALE, 'k.', 'Color', [0.8,0.8,0.8]);
    
    % 绘制楼梯
    plot(s1(1), s1(2), 'ms', 'MarkerSize',8, 'LineWidth',2);
    plot(s2(1), s2(2), 'ms', 'MarkerSize',8, 'LineWidth',2);
    text(s1(1), s1(2)-1, 'Stair F1'); text(s2(1), s2(2)-1, 'Stair F2');
    
    % 绘制路径
    plot(path(:,1), path(:,2), 'b-', 'LineWidth', 1.5);
    
    % 绘制任务点
    for i=1:length(tasks)
        t=tasks(i);
        % 入口
        plot(t.entry(1), t.entry(2), 'go', 'MarkerFaceColor','g');
        % 检查点
        pts = t.points;
        plot(pts(:,1), pts(:,2), 'r.', 'MarkerSize', 8);
        
        % 连线示意
        if t.prio==1, col='r'; else, col='k'; end
        text(t.entry(1), t.entry(2)+1.5, t.name, 'Color', col, 'FontWeight','bold');
    end
    
    title('搜楼阶段全覆盖路径仿真 (Search Phase Trace)');
    xlabel('X (m) [左:仓库 | 中:一层 | 右:二层]'); ylabel('Y (m)');
    ylim([-2, 25]);
end