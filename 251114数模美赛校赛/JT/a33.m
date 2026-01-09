function comprehensive_sweep_simulation()
    % =========================================================================
    % 全面扫楼与搜救模拟系统 (Comprehensive Sweep & Rescue System)
    % =========================================================================
    % 核心功能:
    % 1. 100% 复刻 a31.m 的复杂仓库与建筑布局 (障碍物、门、楼梯)。
    % 2. 实现“全覆盖扫楼” (Sweeping Process)：
    %    - 消防员对仓库、商铺、公寓进行地毯式搜索 (Zigzag路径)。
    %    - 自动避开障碍物，确保视线覆盖每个角落。
    % 3. 物理运动模型 (基于用户指定速度)：
    %    - 响应人员 (Responder): 直线 1.2 m/s, 转弯 1.0 m/s, 上下楼 1.1 m/s。
    %    - 待救援者 (Rescuee):   直线 1.0 m/s, 转弯 0.8 m/s, 上下楼 0.9 m/s。
    % 4. 优先策略:
    %    - 优先搜索 Apt 4 (婴幼儿) 和 Apt 1 (孕妇)。
    %    - 医疗队负责 Apt 1，安保负责 Shop 2，消防员负责全场扫楼。
    % =========================================================================

    clc; clear; close all;
    
    %% 1. 系统参数定义
    global SPEEDS OBSTACLES WAREHOUSE RIGHT_BLDG
    
    % --- 速度参数 (m/s) ---
    SPEEDS.Res_Str   = 1.2; % 响应者直线
    SPEEDS.Res_Turn  = 1.0; % 响应者转弯
    SPEEDS.Res_Stair = 1.1; % 响应者楼梯
    
    SPEEDS.Vic_Str   = 1.0; % 待救援直线
    SPEEDS.Vic_Turn  = 0.8; % 待救援转弯
    SPEEDS.Vic_Stair = 0.9; % 待救援楼梯
    
    % --- 几何参数 (完全复刻 a31.m) ---
    % 仓库 (F1 Only)
    WAREHOUSE.W = 28; WAREHOUSE.H = 22;
    
    % 右侧建筑 (F1 & F2)
    RIGHT_BLDG.Offset = 28; 
    RIGHT_BLDG.Shop_W = 9; RIGHT_BLDG.Shop_H = 16/3;
    RIGHT_BLDG.Room_W = 6; RIGHT_BLDG.Room_H = 4; 
    RIGHT_BLDG.Hall_W = 3;
    
    % --- 初始化障碍物列表 (用于扫楼避障) ---
    OBSTACLES = define_obstacles_a31();
    
    %% 2. 定义任务与扫楼区域
    % 扫楼区域定义 (Zone Definitions)
    % 格式: [x, y, w, h, floor_z, id]
    
    % F1 仓库分区 (由于障碍物复杂，分为几个大区进行扫描)
    Zones(1) = struct('name', 'Warehouse Main', 'rect', [1, 1, 26, 20], 'z', 0, 'prio', 2);
    
    % F1 商铺
    for i = 1:3
        y_s = 3 + (i-1)*RIGHT_BLDG.Shop_H;
        Zones(end+1) = struct('name', sprintf('Shop %d', i), ...
            'rect', [RIGHT_BLDG.Offset+0.5, y_s+0.5, RIGHT_BLDG.Shop_W-1, RIGHT_BLDG.Shop_H-1], ...
            'z', 0, 'prio', 2);
    end
    
    % F2 公寓 (重点: Apt 1 & Apt 4)
    % Apt 1 (孕妇 - High Priority)
    Zones(end+1) = struct('name', 'Apt 1 (Pregnant)', ...
        'rect', [RIGHT_BLDG.Offset+0.5, 3+0.5, RIGHT_BLDG.Room_W-1, RIGHT_BLDG.Room_H-1], ...
        'z', 3, 'prio', 1);
        
    % Apt 2, 3
    for i = 2:3
        y_a = 3 + (i-1)*RIGHT_BLDG.Room_H;
        Zones(end+1) = struct('name', sprintf('Apt %d', i), ...
            'rect', [RIGHT_BLDG.Offset+0.5, y_a+0.5, RIGHT_BLDG.Room_W-1, RIGHT_BLDG.Room_H-1], ...
            'z', 3, 'prio', 2);
    end
    
    % Apt 4 (婴幼儿 - High Priority)
    Zones(end+1) = struct('name', 'Apt 4 (Infant)', ...
        'rect', [RIGHT_BLDG.Offset+0.5, 3+3*RIGHT_BLDG.Room_H+0.5, RIGHT_BLDG.Room_W-1, RIGHT_BLDG.Room_H-1], ...
        'z', 3, 'prio', 1);

    %% 3. 核心计算：生成路径与时间
    fprintf('=== 扫楼与搜救模拟开始 ===\n');
    
    SimulationResults = [];
    
    % 策略：
    % Team A (Firefighter): 负责 F1 仓库复杂区域 (最耗时)
    % Team B (Medical/Police): 负责 F2 公寓 (优先 Apt 1 & 4)
    % Team C (Security): 负责 F1 商铺
    
    % --- Team A: Warehouse Sweep ---
    % 自动生成避障的之字形(Zigzag)搜索路径
    [path_wh, len_wh] = generate_sweep_path(Zones(1).rect, OBSTACLES, 1.5); % 1.5m 间距
    [t_wh, ~] = calculate_physics_time(path_wh, 'Responder');
    SimulationResults = log_result(SimulationResults, 'Firefighter A', 'Warehouse Sweep', t_wh, path_wh, [0.8, 0.2, 0.2]);
    
% --- Team B: F2 Apartments Sweep (Priority Order) ---
    % 修正逻辑：强制通过楼道(x=35.5)和门(y=3, y=19)，防止穿墙
    
    % 1. 楼梯路径: 地面 -> 楼梯间 -> 穿过下方的门(y=3) -> 进入楼道
    % x=35.5 是楼道和门的中心线
    pt_ground   = [35.5, 0, 0];   % 地面入口
    pt_stair_top= [35.5, 1.5, 3]; % 楼梯顶部
    pt_door_bot = [35.5, 3, 3];   % === 关键点: 下方的门 ===
    pt_hall_bot = [35.5, 4, 3];   % 进入楼道内部
    
    path_entry = [pt_ground; pt_stair_top; pt_door_bot; pt_hall_bot];
    
    % 2. 生成各房间扫楼路径 (Z=3)
    [p_a1, ~] = generate_sweep_path(Zones(5).rect, [], 1.0); p_a1(:,3)=3; % Apt 1
    [p_a4, ~] = generate_sweep_path(Zones(8).rect, [], 1.0); p_a4(:,3)=3; % Apt 4
    [p_a2, ~] = generate_sweep_path(Zones(6).rect, [], 1.0); p_a2(:,3)=3; % Apt 2
    [p_a3, ~] = generate_sweep_path(Zones(7).rect, [], 1.0); p_a3(:,3)=3; % Apt 3
    
    % 3. 定义房间到楼道的连接点 (用于串联路径)
    % 房间都在左侧，楼道在右侧，通过 x=34 的墙上的门连接
    to_hall = @(y) [34.5, y, 3; 35.5, y, 3]; % 从房间出来到楼道中心
    from_hall = @(y) [35.5, y, 3; 34.5, y, 3]; % 从楼道中心进房间
    
    % 4. 串联完整路径
    % 顺序: 入口 -> Apt 1 -> 回楼道 -> 走楼道 -> Apt 4 -> ...
    full_path_f2 = [
        path_entry; 
        from_hall(5); p_a1; to_hall(5);       % 扫 Apt 1 (y~5)
        [35.5, 17, 3];                        % 沿楼道走到上方
        from_hall(17); p_a4; to_hall(17);     % 扫 Apt 4 (y~17)
        [35.5, 9, 3];                         % 回到中间
        from_hall(9);  p_a2; to_hall(9);      % 扫 Apt 2
        from_hall(13); p_a3; to_hall(13)      % 扫 Apt 3
    ];
    
    [t_f2, ~] = calculate_physics_time(full_path_f2, 'Responder');
    SimulationResults = log_result(SimulationResults, 'Firefighter B', 'F2 Priority Sweep', t_f2, full_path_f2, [0.2, 0.2, 0.8]);
    % --- Team C: Shops Sweep ---
    path_shops = [];
    for i = 2:4 % Shop Indices in Zones
        [p_s, ~] = generate_sweep_path(Zones(i).rect, [], 1.2);
        path_shops = [path_shops; p_s];
    end
    [t_shops, ~] = calculate_physics_time(path_shops, 'Responder');
    SimulationResults = log_result(SimulationResults, 'Security Team', 'Shops Sweep', t_shops, path_shops, [0.2, 0.6, 0.2]);
    
    %% 4. 结果显示
    fprintf('------------------------------------------------------------\n');
    fprintf('%-15s | %-20s | %-10s\n', 'Team', 'Task', 'Time (s)');
    fprintf('------------------------------------------------------------\n');
    for i = 1:length(SimulationResults)
        fprintf('%-15s | %-20s | %.2f s\n', ...
            SimulationResults(i).Team, SimulationResults(i).Task, SimulationResults(i).Time);
    end
    fprintf('------------------------------------------------------------\n');
    
    %% 5. 绘图 (2D & 3D)
    visualize_operations(SimulationResults);
end

% =========================================================================
% 辅助函数：障碍物定义 (a31.m 复刻)
% =========================================================================
function obs = define_obstacles_a31()
    % 格式: [x, y, w, h]
    obs = [];
    % Area A
    obs = [obs; 3, 16, 8, 2];
    obs = [obs; 3, 12, 2, 4];
    obs = [obs; 6, 13, 2, 2];
    % Area B
    obs = [obs; 3, 4, 8, 3];
    obs = [obs; 3, 8, 3, 2];
    % Area C (Islands)
    obs = [obs; 13, 14, 2, 4];
    obs = [obs; 13, 4, 2, 4];
    obs = [obs; 17, 10, 2, 6];
    obs = [obs; 17, 17, 2, 2];
    obs = [obs; 17, 2, 2, 2];
    % Area D
    obs = [obs; 22, 5, 2, 12];
    % Door Blockers
    obs = [obs; 13, 19, 4, 1];
    obs = [obs; 25, 18, 1, 3];
end

% =========================================================================
% 辅助函数：扫描路径生成 (Zigzag / Lawnmower)
% =========================================================================
function [path, total_dist] = generate_sweep_path(rect, obstacles, step_size)
    % 在给定矩形区域内生成之字形路径，并避开 obstacles
    x0 = rect(1); y0 = rect(2); w = rect(3); h = rect(4);
    
    xs = x0 : step_size : (x0 + w);
    ys = y0 : step_size : (y0 + h);
    
    path = [];
    
    % 简单的网格扫描算法
    for i = 1:length(xs)
        x = xs(i);
        if mod(i, 2) == 1
            y_scan = ys; % 向上
        else
            y_scan = flip(ys); % 向下
        end
        
        for y = y_scan
            % 碰撞检测
            is_collision = false;
            if ~isempty(obstacles)
                % 检查点 (x,y) 是否在任意障碍物内
                % Obstacles: [ox, oy, ow, oh]
                for k = 1:size(obstacles, 1)
                    obs = obstacles(k, :);
                    if x >= obs(1)-0.5 && x <= obs(1)+obs(3)+0.5 && ...
                       y >= obs(2)-0.5 && y <= obs(2)+obs(4)+0.5
                        is_collision = true;
                        break;
                    end
                end
            end
            
            if ~is_collision
                path = [path; x, y, 0]; % 默认为 Z=0，外部调用可修改
            end
        end
    end
    
    % 计算距离
    total_dist = 0;
    if size(path, 1) > 1
        d = diff(path(:, 1:2));
        total_dist = sum(sqrt(sum(d.^2, 2)));
    end
end

% =========================================================================
% 辅助函数：物理时间计算
% =========================================================================
function [total_time, log_str] = calculate_physics_time(path, role)
    global SPEEDS
    
    if strcmp(role, 'Responder')
        v_str = SPEEDS.Res_Str;
        v_turn = SPEEDS.Res_Turn;
        v_stair = SPEEDS.Res_Stair;
    else
        v_str = SPEEDS.Vic_Str;
        v_turn = SPEEDS.Vic_Turn;
        v_stair = SPEEDS.Vic_Stair;
    end
    
    total_time = 0;
    if size(path, 1) < 2, return; end
    
    for i = 1:size(path, 1)-1
        p1 = path(i, :);
        p2 = path(i+1, :);
        
        dist = norm(p1 - p2);
        dz = abs(p1(3) - p2(3));
        
        % 1. 移动时间
        if dz > 0.5 % 上下楼
            t_seg = dist / v_stair;
        else % 平面移动
            t_seg = dist / v_str;
        end
        total_time = total_time + t_seg;
        
        % 2. 转弯时间惩罚
        % 如果向量方向发生显著改变，视为转弯
        if i > 1
            p0 = path(i-1, :);
            vec1 = p1 - p0; vec1 = vec1 / (norm(vec1)+eps);
            vec2 = p2 - p1; vec2 = vec2 / (norm(vec2)+eps);
            
            cos_theta = dot(vec1, vec2);
            if cos_theta < 0.8 % 角度变化较大
                % 假设转弯动作导致 1米的路程需要用 v_turn 速度完成，而不是 v_str
                % 时间增量 = 1/v_turn - 1/v_str
                turn_penalty = (1/v_turn - 1/v_str); 
                if turn_penalty > 0, total_time = total_time + turn_penalty; end
            end
        end
    end
    log_str = sprintf('%.2f s', total_time);
end

function res = log_result(res_list, team, task, time, path, color)
    new_entry.Team = team;
    new_entry.Task = task;
    new_entry.Time = time;
    new_entry.Path = path;
    new_entry.Color = color;
    if isempty(res_list)
        res = new_entry;
    else
        res = [res_list, new_entry];
    end
end

% =========================================================================
% 绘图功能
% =========================================================================
function visualize_operations(results)
    % 创建图形
    figure('Color', 'w', 'Name', 'Rescue Sweep Operation', 'Position', [50, 50, 1200, 800]);
    
    % --- 1. 3D 视图 ---
    subplot(2, 2, [1, 3]); 
    hold on; axis equal; grid on; box on;
    title('3D Comprehensive Sweep Visualization', 'FontSize', 14);
    view(45, 30);
    
    % 绘制建筑结构 (3D)
    draw_building_structure_3d();
    
    % 绘制路径
    for i = 1:length(results)
        p = results(i).Path;
        plot3(p(:,1), p(:,2), p(:,3), '-', 'Color', results(i).Color, 'LineWidth', 1.5);
        % 起终点
        plot3(p(1,1), p(1,2), p(1,3), 'o', 'MarkerFaceColor', 'g', 'MarkerSize', 6);
        plot3(p(end,1), p(end,2), p(end,3), 's', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
    end
    legend_str = {results.Task};
    % legend(legend_str, 'Location', 'bestoutside'); % 可选
    
    % --- 2. F2 平面图 ---
    subplot(2, 2, 2);
    hold on; axis equal; grid on; box on;
    title('Floor 2: Apartments (High Priority)', 'FontSize', 12);
    draw_floor_2d(2);
    for i = 1:length(results)
        p = results(i).Path;
        if any(abs(p(:,3) - 3) < 0.5) % Show F2 paths
            mask = abs(p(:,3) - 3) < 0.5;
            plot(p(mask,1), p(mask,2), '-', 'Color', results(i).Color, 'LineWidth', 1);
        end
    end
    xlim([20, 45]); ylim([-2, 24]);
    
    % --- 3. F1 平面图 ---
    subplot(2, 2, 4);
    hold on; axis equal; grid on; box on;
    title('Floor 1: Warehouse & Shops', 'FontSize', 12);
    draw_floor_2d(1);
    for i = 1:length(results)
        p = results(i).Path;
        if any(abs(p(:,3) - 0) < 0.5) % Show F1 paths
            mask = abs(p(:,3) - 0) < 0.5;
            plot(p(mask,1), p(mask,2), '-', 'Color', results(i).Color, 'LineWidth', 1);
        end
    end
    xlim([-2, 40]); ylim([-2, 24]);
end

function draw_building_structure_3d()
    % 简化的3D结构绘制
    global OBSTACLES
    % 地面
    patch([0 28 28 0], [0 0 22 22], [0 0 0 0], [0.9 0.9 0.9], 'FaceAlpha', 0.5);
    % 二楼地板
    patch([28 37 37 28], [0 0 22 22], [3 3 3 3], [0.8 0.8 0.8], 'FaceAlpha', 0.5);
    
    % 绘制障碍物 (高度设为2米)
    for k = 1:size(OBSTACLES, 1)
        o = OBSTACLES(k, :);
        draw_block_3d(o(1), o(2), o(3), o(4), 2, [0.6 0.6 0.6]);
    end
end

function draw_block_3d(x, y, w, h, z, col)
    vert = [x y 0; x+w y 0; x+w y+h 0; x y+h 0; ...
            x y z; x+w y z; x+w y+h z; x y+h z];
    faces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 5 6 7 8; 1 2 3 4];
    patch('Vertices', vert, 'Faces', faces, 'FaceColor', col, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end

function draw_floor_2d(floor_idx)
    global OBSTACLES
    if floor_idx == 1
        rectangle('Position', [0 0 28 22], 'LineWidth', 2); % Warehouse
        % Draw Obstacles
        for k = 1:size(OBSTACLES, 1)
            o = OBSTACLES(k, :);
            rectangle('Position', o, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k');
        end
        % Draw Shops
        x_off = 28;
        for i=1:3, y=3+(i-1)*16/3; rectangle('Position',[x_off, y, 9, 16/3], 'EdgeColor','b'); end
    else
        x_off = 28;
        % Draw Apts
        for i=1:4, y=3+(i-1)*4; rectangle('Position',[x_off, y, 6, 4], 'EdgeColor','b'); end
        % Hallway
        rectangle('Position',[x_off+6, 3, 3, 16], 'FaceColor',[0.95 0.95 0.95]);
        % === [新增代码: 绘制楼道两端的门] ===
        % 下门 (连接下楼梯)
        plot([34.5, 36.5], [3, 3], 'r-', 'LineWidth', 3);
        % 上门 (连接上楼梯)
        plot([34.5, 36.5], [19, 19], 'r-', 'LineWidth', 3);
        % =================================
    end
end