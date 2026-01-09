function total_time = rescue_core_v12(input_n_responders, input_start_delay, input_v_speed, do_plot)
    % ======================================
    % 核心仿真函数 V12 (支持延迟灵敏度分析)
    % 
    % 输入: 
    %   input_n_responders: 响应者总数 (整数, e.g., 6)
    %   input_start_delay:  开始救援前的延迟时间 (秒, e.g., 60) <--- [关键修改]
    %   input_v_speed:      人员基础移动速度 (浮点数, e.g., 1.2)
    %   do_plot:            true=绘图/打印详日志, false=静默运行(只计算)
    % 输出:
    %   total_time:         完成所有任务的总耗时 (秒)
    % ======================================
    
    % --- 1. 仿真配置 ---
    N_RESPONDERS = input_n_responders;
    
    % V12 策略: 动态计算仓库搜救人数 (约占 60%, 至少 1 人)
    N_WAREHOUSE_RESPONDERS = max(1, floor(N_RESPONDERS * 0.6)); 
    
    % 使用输入的延迟时间
    START_DELAY = input_start_delay;         
    RESOLUTION = 0.25;          
    
    % 几何常量
    X_offset = 28;            
    shop_h = 16/3; 
    room_h = 4;
    
    if do_plot
        fprintf('--- 启动混合救援仿真 (V12 Core) ---\n');
        fprintf('响应者总数: %d, 延迟: %.1fs, 速度: %.2f m/s, 仓库并行数: %d\n', ...
                 N_RESPONDERS, START_DELAY, input_v_speed, N_WAREHOUSE_RESPONDERS);
    end

    % --- 1.1 定义全局参数 (动态化) ---
    global PARAMS;
    PARAMS.V_RESP_STRAIGHT = input_v_speed;       % <--- 动态输入
    PARAMS.V_RESP_TURN = input_v_speed * 0.85;    % 转向速度随动
    PARAMS.V_RESP_STAIRS = input_v_speed * 0.9;   % 楼梯速度随动
    PARAMS.V_MIN_SMOKE = 0.2;                     % 烟雾中最低速度保持不变
    PARAMS.R_MIN = 0.5;
    PARAMS.R_THRESH_SWEEP = 2.5;
    PARAMS.SWEEP_ETA = 0.3;
    PARAMS.SWEEP_KAPPA = 1.4;
    PARAMS.SWEEP_DELTA_MIN = 0.5;
    PARAMS.FLOOR_HEIGHT = 2.5; 
    
    % --- 2. 构建环境 ---
    if do_plot, fprintf('正在构建 3D 环境...\n'); end
    [ZONES, BUILDING_GRAPH, WAREHOUSE_GRID, WH_OBSTACLES, ALL_ENTRY_NODES] = build_environment_v11(RESOLUTION);

    % --- 3. 定义救援任务 ---
    tasks_data_static = {
        % 高优先级 (已知人员)
        'Infant',    'Apt 4',   [29, 19],    'Injured', nan; 
        'Pregnant',  'Apt 1',   [29, 4],     'Injured', nan; 
        'Shop Person', 'Shop 2',  [30, 11.33], 'Normal',  nan; 
        
        % 高优先级 (关键搜索点)
        'WH Search 1 (P1)', 'Warehouse', [0.5, 0.5],  'Search', nan; 
        'WH Search 2 (P2)', 'Warehouse', [16, 18],   'Search', nan; 

        % 中优先级 (扫楼 - 剩余房间)
        'Search Apt 2',  'Apt 2',   [X_offset+3, 3+room_h*1.5],   'Search', nan;
        'Search Apt 3',  'Apt 3',   [X_offset+3, 3+room_h*2.5],   'Search', nan;
        'Search Shop 1', 'Shop 1',  [X_offset+4.5, 3+shop_h/2],  'Search', nan;
        'Search Shop 3', 'Shop 3',  [X_offset+4.5, 3+shop_h*2.5],'Search', nan;
    };
    
    % 动态生成并行仓库扫描任务 (根据 N_WAREHOUSE_RESPONDERS)
    tasks_data_wh_sweep = cell(N_WAREHOUSE_RESPONDERS, 5);
    for z = 1:N_WAREHOUSE_RESPONDERS
        tasks_data_wh_sweep(z,:) = {sprintf('WH Sweep Zone %d', z), 'Warehouse', [14, 11], 'Sweep', z};
    end

    % 组合任务
    tasks_data = [
        tasks_data_static(1:5,:);   
        tasks_data_static(6:9,:);   
        tasks_data_wh_sweep         
    ];
    
    if do_plot, fprintf('任务生成完毕: %d 个任务.\n', size(tasks_data, 1)); end

    % --- 3.2 任务结构体初始化 ---
    TARGETS = struct('name', {}, 'zone_id', {}, 'pos_m', {}, 'pos_grid', {}, ...
                   'type', {}, 'status', {}, 'zone_idx_for_sweep', {});
    for i = 1:size(tasks_data, 1) 
        TARGETS(i).name = tasks_data{i, 1};
        zone_idx = find(strcmp({ZONES.name}, tasks_data{i, 2}));
        TARGETS(i).zone_id = zone_idx;
        TARGETS(i).pos_m = tasks_data{i, 3};
        TARGETS(i).type = tasks_data{i, 4};
        TARGETS(i).status = 'Pending';
        TARGETS(i).zone_idx_for_sweep = tasks_data{i, 5}; 
        
        if strcmp(ZONES(zone_idx).type, 'Warehouse')
            node = find_closest_walkable(TARGETS(i).pos_m, WAREHOUSE_GRID.grid, RESOLUTION);
            TARGETS(i).pos_grid = node;
        else
            TARGETS(i).pos_grid = []; 
        end
    end

    % --- 4. 仿真执行 ---
    % 动态分配起始点
    responders = struct('id', {}, 'start_node', {}, 'time_free', {}, 'log', {});
    num_entries = length(ALL_ENTRY_NODES);
    for i = 1:N_RESPONDERS
        responders(i).id = i;
        entry_idx = mod(i-1, num_entries) + 1;
        responders(i).start_node = ALL_ENTRY_NODES{entry_idx};
        responders(i).time_free = START_DELAY; 
    end
    
    task_queue = 1:length(TARGETS); 
    all_paths = {}; 
    
    while ~isempty(task_queue)
        current_task_idx = task_queue(1);
        task_queue(1) = []; 
        
        target = TARGETS(current_task_idx);
        target_zone = ZONES(target.zone_id);
        
        [~, resp_idx] = min([responders.time_free]);
        t_start = responders(resp_idx).time_free;
        
        if do_plot, fprintf('  -> [T=%.0fs] R%d 分配: %s\n', t_start, resp_idx, target.name); end

        % --- 路径计算 ---
        
        % 1. 决定高层图目标节点
        high_level_target_node = '';
        if strcmp(target_zone.type, 'Warehouse')
            high_level_target_node = target_zone.graph_node; 
        else
            high_level_target_node = [target_zone.graph_node, '_Door']; 
        end

        % 2. 高层图路径
        [path_high, time_high_level] = find_path_high_level_3d(responders(resp_idx).start_node, ...
                                            high_level_target_node, BUILDING_GRAPH, ZONES, t_start);
        
        if isinf(time_high_level)
            if do_plot, fprintf('     !! 严重错误: 无路径.\n'); end
            continue; 
        end
        
        % 3. 低层路径
        if strcmp(target_zone.type, 'Warehouse')
            
            start_node_name = responders(resp_idx).start_node;
            entry_door_node_name = '';
            
            % 3.1 找到 A* 起点
            if length(path_high) == 1
                target_pos_m = (target.pos_grid - 0.5) * RESOLUTION;
                wh_door_names = {'WH_Door_Left', 'WH_Door_Top', 'WH_Door_Bottom', 'WH_Door_Stairs1', 'WH_Door_Stairs2'};
                best_door_name = '';
                min_dist = inf;
                for i = 1:length(wh_door_names)
                    door_name = wh_door_names{i};
                    door_pos_m = BUILDING_GRAPH.Nodes.(door_name).pos(1:2); 
                    dist = norm(door_pos_m - target_pos_m(1:2)); 
                    if dist < min_dist
                        min_dist = dist;
                        best_door_name = door_name;
                    end
                end
                entry_door_node_name = best_door_name;
                [path_to_door, time_to_door] = find_path_high_level_3d(start_node_name, ...
                    entry_door_node_name, BUILDING_GRAPH, ZONES, t_start);
                t_arr_zone = t_start + time_to_door; 
                task_total_time = time_to_door;     
                path_segments = {path_to_door};     
            else
                entry_door_node_name = path_high(end-1).name; 
                t_arr_zone = t_start + time_high_level; 
                task_total_time = time_high_level;
                path_segments = {path_high};
            end
            
            wh_entry_pos_m = BUILDING_GRAPH.Nodes.(entry_door_node_name).pos;
            wh_entry_grid = find_closest_walkable(wh_entry_pos_m(1:2), WAREHOUSE_GRID.grid, RESOLUTION);
            
            % 3.2 根据任务类型选择 A* 还是 弓字形
            if strcmp(target.type, 'Search')
                target_grid = target.pos_grid;
                [path_low_nodes, time_to_target_low] = find_path_A_star_warehouse(wh_entry_grid, ...
                                                        target_grid, WAREHOUSE_GRID, ZONES, t_arr_zone, RESOLUTION);
                if isempty(path_low_nodes)
                    continue;
                end
                t_arr_person = t_arr_zone + time_to_target_low; 
                task_total_time = task_total_time + time_to_target_low;
                path_segments{end+1} = path_low_nodes;
                sweep_time = 5.0; 
            
            elseif strcmp(target.type, 'Sweep')
                zone_index = target.zone_idx_for_sweep;
                [grid_h, grid_w] = size(WAREHOUSE_GRID.grid);
                zone_width_grid = floor(grid_w / N_WAREHOUSE_RESPONDERS);
                x_start_grid = (zone_index - 1) * zone_width_grid + 3;
                sweep_start_grid = [x_start_grid, grid_h-3]; 
                
                [path_to_sweep_start, time_to_sweep_start] = find_path_A_star_warehouse(wh_entry_grid, ...
                                                        sweep_start_grid, WAREHOUSE_GRID, ZONES, t_arr_zone, RESOLUTION);
                
                if isempty(path_to_sweep_start)
                    continue;
                end
                
                t_arr_sweep_start = t_arr_zone + time_to_sweep_start;
                task_total_time = task_total_time + time_to_sweep_start;
                path_segments{end+1} = path_to_sweep_start;
                
                [sweep_path_nodes, sweep_time] = calculate_warehouse_sweep_v11(sweep_start_grid, ...
                                                    WAREHOUSE_GRID, ZONES, t_arr_sweep_start, RESOLUTION, ...
                                                    zone_index, N_WAREHOUSE_RESPONDERS);
                
                task_total_time = task_total_time + sweep_time;
                path_segments{end+1} = sweep_path_nodes;
            end

            task_total_time = task_total_time + sweep_time;

        else 
            % Apt/Shop (周界)
            t_arr_zone = t_start + time_high_level;
            task_total_time = time_high_level;
            path_segments = {path_high};

            [sweep_time, sweep_path_local] = calculate_sweep_time_v11(target_zone, t_arr_zone);
            task_total_time = task_total_time + sweep_time;

            sweep_path_global = sweep_path_local;
            sweep_path_global(:,1) = sweep_path_local(:,1) + target_zone.rect(1);
            sweep_path_global(:,2) = sweep_path_local(:,2) + target_zone.rect(2);
            path_segments{end+1} = sweep_path_global;
        end
        
        % 4. 更新响应者状态
        responders(resp_idx).time_free = t_start + task_total_time;
        if strcmp(target_zone.type, 'Warehouse')
            responders(resp_idx).start_node = target_zone.graph_node; 
        else
            responders(resp_idx).start_node = high_level_target_node; 
        end
        
        TARGETS(current_task_idx).status = 'Completed';
        all_paths{end+1} = struct('resp_id', resp_idx, 'segments', {path_segments}, ...
            'target_name', target.name, 'zone_id', target.zone_id, 'total_time', task_total_time);
    end
    
    % --- 5. 退出逻辑 ---
    if do_plot, fprintf('--- 计算退出路径 ---\n'); end
    
    for i = 1:N_RESPONDERS
        resp = responders(i);
        t_start = resp.time_free;
        start_node = resp.start_node;
        
        best_exit_node = '';
        min_time = inf;
        
        for j = 1:length(ALL_ENTRY_NODES)
            exit_node = ALL_ENTRY_NODES{j}; 
            [~, time_to_exit] = find_path_high_level_3d(start_node, exit_node, BUILDING_GRAPH, ZONES, t_start);
            if time_to_exit < min_time
                min_time = time_to_exit;
                best_exit_node = exit_node;
            end
        end
        
        [path_exit, time_to_exit] = find_path_high_level_3d(start_node, best_exit_node, BUILDING_GRAPH, ZONES, t_start);
        responders(i).time_free = t_start + time_to_exit;
        all_paths{end+1} = struct('resp_id', resp.id, 'segments', {{path_exit}}, ...
            'target_name', 'Exit', 'zone_id', 1, 'total_time', time_to_exit);
    end
    
    % --- 6. 计算结果 ---
    total_time = max([responders.time_free]) - START_DELAY;
    
    % 仅在 do_plot 为 true 时绘图
    if do_plot
        fprintf('--- 仿真结束. 总耗时: %.2f s ---\n', total_time);
        plot_simulation_3D_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, RESOLUTION, N_RESPONDERS);
        plot_2d_floor_plans_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, RESOLUTION, N_RESPONDERS);
    end
end


% ===================================================================
% ===================================================================
%                        辅助函数 (HELPER FUNCTIONS)
% ===================================================================
% ===================================================================

% ========================
% 1. BUILD_ENVIRONMENT
% ========================
function [ZONES, BUILDING_GRAPH, WAREHOUSE_GRID, WH_OBSTACLES, ALL_ENTRY_NODES] = build_environment_v11(res)
    global PARAMS;
    Z_F1 = 0.0; 
    Z_F2 = PARAMS.FLOOR_HEIGHT; 

    ZONES = struct('id', {}, 'name', {}, 'rect', {}, 'type', {}, ...
                   'T_crit', {}, 'R_max', {}, 'graph_node', {}, 'z_level', {});
    
    W_wh = 28; H_wh = 22; X_offset = 28;
    shop_w = 9; shop_h = 16/3; 
    room_w = 6; room_h = 4; hall_w = 3;

    ZONES(end+1).id = 1; ZONES(end).name = 'Warehouse'; ZONES(end).rect = [0, 0, W_wh, H_wh];
    ZONES(end).type = 'Warehouse'; ZONES(end).T_crit = 30 * 60; ZONES(end).R_max = norm([W_wh, H_wh]);
    ZONES(end).graph_node = 'Warehouse'; ZONES(end).z_level = Z_F1;
    
    for i = 1:3
        y_s = 3 + (i-1)*shop_h;
        ZONES(end+1).id = 1+i; ZONES(end).name = sprintf('Shop %d', i); ZONES(end).rect = [X_offset, y_s, shop_w, shop_h];
        ZONES(end).type = 'Shop'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([shop_w, shop_h]);
        ZONES(end).graph_node = sprintf('Shop%d', i); ZONES(end).z_level = Z_F1;
    end

    for i = 1:4
        y_a = 3 + (i-1)*room_h;
        ZONES(end+1).id = 4+i; ZONES(end).name = sprintf('Apt %d', i); ZONES(end).rect = [X_offset, y_a, room_w, room_h];
        ZONES(end).type = 'Apt'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([room_w, room_h]);
        ZONES(end).graph_node = sprintf('Apt%d', i); ZONES(end).z_level = Z_F2;
    end
    
    ZONES(end+1).id = 9; ZONES(end).name = 'Hallway'; ZONES(end).rect = [X_offset + room_w, 3, hall_w, 16];
    ZONES(end).type = 'Hallway'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([hall_w, 16]);
    ZONES(end).graph_node = 'Hallway'; ZONES(end).z_level = Z_F2;

    ZONES(end+1).id = 10; ZONES(end).name = 'Stairs 1'; ZONES(end).rect = [X_offset, 0, 9, 3];
    ZONES(end).type = 'Stairs'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([9, 3]);
    ZONES(end).graph_node = 'Stairs1'; ZONES(end).z_level = Z_F1;
    
    ZONES(end+1).id = 11; ZONES(end).name = 'Stairs 2'; ZONES(end).rect = [X_offset, 19, 9, 3];
    ZONES(end).type = 'Stairs'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([9, 3]);
    ZONES(end).graph_node = 'Stairs2'; ZONES(end).z_level = Z_F1;

    % --- WAREHOUSE_GRID ---
    grid_w = ceil(W_wh / res);
    grid_h = ceil(H_wh / res);
    grid = zeros(grid_h, grid_w); 
    grid_zone_id = ones(grid_h, grid_w); 
    
    function [g, gz] = draw_grid_rect(g, gz, x, y, w, h, val, zone_id, res)
        x1 = max(1, floor(x / res) + 1);
        y1 = max(1, floor(y / res) + 1);
        x2 = min(size(g, 2), ceil((x+w) / res));
        y2 = min(size(g, 1), ceil((y+h) / res));
        g(y1:y2, x1:x2) = val;
        if zone_id > 0, gz(y1:y2, x1:x2) = zone_id; end
    end
    
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, 0, 0, W_wh, res, 1, 1, res); 
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, 0, H_wh-res, W_wh, res, 1, 1, res); 
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, 0, 0, res, H_wh, 1, 1, res); 
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, W_wh-res, 0, res, H_wh, 1, 1, res); 
    
    WH_OBSTACLES = {
        [3, 16, 8, 2], [3, 12, 2, 4], [6, 13, 2, 2], [3, 4, 8, 3], [3, 8, 3, 2], ...
        [13, 14, 2, 4], [13, 4, 2, 4], [17, 10, 2, 6], [17, 17, 2, 2], [17, 2, 2, 2], ...
        [22, 5, 2, 12], [13, 19, 4, 1], [25, 18, 1, 3]
    };
    for i=1:length(WH_OBSTACLES)
        obs = WH_OBSTACLES{i};
        [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, obs(1), obs(2), obs(3), obs(4), 2, 1, res);
    end

    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, 0, 7, res, 8, 0, 1, res); 
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, 13, H_wh-res, 2, res, 0, 1, res);
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, 13, 0, 2, res, 0, 1, res);
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, W_wh-res, 0.5, res, 2, 0, 1, res);
    [grid, grid_zone_id] = draw_grid_rect(grid, grid_zone_id, W_wh-res, 19.5, res, 2, 0, 1, res);
    
    WAREHOUSE_GRID.grid = grid;
    WAREHOUSE_GRID.grid_zone_id = grid_zone_id;
    WAREHOUSE_GRID.origin = [0, 0];
    WAREHOUSE_GRID.res = res;

    % --- BUILDING_GRAPH ---
    G.Nodes = struct();
    G.Edges = {};
    ALL_ENTRY_NODES = {}; 
    
    function G_Nodes = add_node(G_Nodes, name, pos_3d, zone_id, type)
        G_Nodes.(name).name = name;
        G_Nodes.(name).pos = pos_3d; 
        G_Nodes.(name).zone_id = zone_id;
        G_Nodes.(name).type = type; 
    end
    
    function G_Edges = add_edge(G_Edges, G_Nodes, n1_name, n2_name, type)
        n1 = G_Nodes.(n1_name); n2 = G_Nodes.(n2_name);
        if strcmp(type, 'Stairs')
            dist = norm(n1.pos(1:2) - n2.pos(1:2)) + abs(n1.pos(3) - n2.pos(3));
        else
            dist = norm(n1.pos - n2.pos);
        end
        G_Edges{end+1} = struct('n1', n1_name, 'n2', n2_name, 'dist', dist, 'type', type); 
    end

    G.Nodes = add_node(G.Nodes, 'Entry_WH_Left', [-1, 11, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_WH_Top', [14, 23, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_WH_Bottom', [14, -1, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Shop1', [X_offset+shop_w+1, 3+shop_h/2, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Shop2', [X_offset+shop_w+1, 3+shop_h*1.5, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Shop3', [X_offset+shop_w+1, 3+shop_h*2.5, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Stairs1', [X_offset+shop_w+1, 1.5, Z_F1], 0, 'Entry'); 
    G.Nodes = add_node(G.Nodes, 'Entry_Stairs2', [X_offset+shop_w+1, 20.5, Z_F1], 0, 'Entry');
    
    G.Nodes = add_node(G.Nodes, 'Warehouse', [14, 11, Z_F1], 1, 'Room'); 
    G.Nodes = add_node(G.Nodes, 'Shop1', [X_offset+shop_w/2, 3+shop_h/2, Z_F1], 2, 'Room');
    G.Nodes = add_node(G.Nodes, 'Shop2', [X_offset+shop_w/2, 3+shop_h*1.5, Z_F1], 3, 'Room');
    G.Nodes = add_node(G.Nodes, 'Shop3', [X_offset+shop_w/2, 3+shop_h*2.5, Z_F1], 4, 'Room');
    G.Nodes = add_node(G.Nodes, 'Apt1', [X_offset+room_w/2, 3+room_h*0.5, Z_F2], 5, 'Room');
    G.Nodes = add_node(G.Nodes, 'Apt2', [X_offset+room_w/2, 3+room_h*1.5, Z_F2], 6, 'Room');
    G.Nodes = add_node(G.Nodes, 'Apt3', [X_offset+room_w/2, 3+room_h*2.5, Z_F2], 7, 'Room');
    G.Nodes = add_node(G.Nodes, 'Apt4', [X_offset+room_w/2, 3+room_h*3.5, Z_F2], 8, 'Room');
    G.Nodes = add_node(G.Nodes, 'Hallway', [X_offset+room_w+hall_w/2, 11, Z_F2], 9, 'Transition');
    G.Nodes = add_node(G.Nodes, 'Stairs1', [X_offset+shop_w/2, 1.5, Z_F1], 10, 'Transition');
    G.Nodes = add_node(G.Nodes, 'Stairs2', [X_offset+shop_w/2, 20.5, Z_F1], 11, 'Transition');
    
    G.Nodes = add_node(G.Nodes, 'WH_Door_Left', [0, 11, Z_F1], 1, 'Door');
    G.Nodes = add_node(G.Nodes, 'WH_Door_Top', [14, 22, Z_F1], 1, 'Door');
    G.Nodes = add_node(G.Nodes, 'WH_Door_Bottom', [14, 0, Z_F1], 1, 'Door');
    G.Nodes = add_node(G.Nodes, 'WH_Door_Stairs1', [28, 1.5, Z_F1], 1, 'Door');
    G.Nodes = add_node(G.Nodes, 'WH_Door_Stairs2', [28, 20.5, Z_F1], 1, 'Door');
    G.Nodes = add_node(G.Nodes, 'Shop1_Door', [X_offset+shop_w, 3+shop_h/2, Z_F1], 2, 'Door');
    G.Nodes = add_node(G.Nodes, 'Shop2_Door', [X_offset+shop_w, 3+shop_h*1.5, Z_F1], 3, 'Door');
    G.Nodes = add_node(G.Nodes, 'Shop3_Door', [X_offset+shop_w, 3+shop_h*2.5, Z_F1], 4, 'Door');
    G.Nodes = add_node(G.Nodes, 'Apt1_Door', [X_offset+room_w, 3+room_h*0.5, Z_F2], 5, 'Door');
    G.Nodes = add_node(G.Nodes, 'Apt2_Door', [X_offset+room_w, 3+room_h*1.5, Z_F2], 6, 'Door');
    G.Nodes = add_node(G.Nodes, 'Apt3_Door', [X_offset+room_w, 3+room_h*2.5, Z_F2], 7, 'Door');
    G.Nodes = add_node(G.Nodes, 'Apt4_Door', [X_offset+room_w, 3+room_h*3.5, Z_F2], 8, 'Door');
    G.Nodes = add_node(G.Nodes, 'Hall_Door_Bottom', [X_offset+room_w+hall_w/2, 3, Z_F2], 9, 'Door');
    G.Nodes = add_node(G.Nodes, 'Hall_Door_Top', [X_offset+room_w+hall_w/2, 19, Z_F2], 9, 'Door');

    all_nodes = fieldnames(G.Nodes);
    entry_nodes = {};
    for i = 1:length(all_nodes)
        if strcmp(G.Nodes.(all_nodes{i}).type, 'Entry')
            entry_nodes{end+1} = all_nodes{i};
        end
    end
    ALL_ENTRY_NODES = entry_nodes;
    
    for i = 1:length(entry_nodes)
        for j = (i+1):length(entry_nodes)
            G.Edges = add_edge(G.Edges, G.Nodes, entry_nodes{i}, entry_nodes{j}, 'Hall'); 
        end
    end
    
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_WH_Left', 'WH_Door_Left', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_WH_Top', 'WH_Door_Top', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_WH_Bottom', 'WH_Door_Bottom', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_Stairs1', 'Stairs1', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_Stairs2', 'Stairs2', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_Shop1', 'Shop1_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_Shop2', 'Shop2_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Entry_Shop3', 'Shop3_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Shop1_Door', 'Shop1', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Shop2_Door', 'Shop2', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Shop3_Door', 'Shop3', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Warehouse', 'WH_Door_Left', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Warehouse', 'WH_Door_Top', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Warehouse', 'WH_Door_Bottom', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Warehouse', 'WH_Door_Stairs1', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Warehouse', 'WH_Door_Stairs2', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'WH_Door_Stairs1', 'Stairs1', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'WH_Door_Stairs2', 'Stairs2', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hall_Door_Bottom', 'Hallway', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hall_Door_Top', 'Hallway', 'Hall'); 
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hallway', 'Apt1_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hallway', 'Apt2_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hallway', 'Apt3_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hallway', 'Apt4_Door', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt1_Door', 'Apt1', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt2_Door', 'Apt2', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt3_Door', 'Apt3', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt4_Door', 'Apt4', 'Hall');

    G.Edges = add_edge(G.Edges, G.Nodes, 'Stairs1', 'Hall_Door_Bottom', 'Stairs');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Stairs2', 'Hall_Door_Top', 'Stairs');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hall_Door_Bottom', 'Hallway', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Hall_Door_Top', 'Hallway', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt1_Door', 'Hallway', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt2_Door', 'Hallway', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt3_Door', 'Hallway', 'Hall');
    G.Edges = add_edge(G.Edges, G.Nodes, 'Apt4_Door', 'Hallway', 'Hall');

    BUILDING_GRAPH = G;
end


% ========================
% 2. PATHFINDING
% ========================
function [path_nodes, total_time] = find_path_high_level_3d(start_node_name, end_node_name, G, ZONES, t_start)
    nodes = fieldnames(G.Nodes);
    num_nodes = length(nodes);
    dist = inf(num_nodes, 1);
    prev = zeros(num_nodes, 1);
    name_to_idx = containers.Map(nodes, 1:num_nodes);
    idx_to_name = nodes;
    
    start_idx = name_to_idx(start_node_name);
    end_idx = name_to_idx(end_node_name);
    
    dist(start_idx) = 0;
    Q = 1:num_nodes;
    
    while ~isempty(Q)
        [~, u_idx_in_Q] = min(dist(Q));
        u_idx = Q(u_idx_in_Q);
        Q(u_idx_in_Q) = []; 
        
        if u_idx == end_idx, break; end
        
        u_name = idx_to_name{u_idx};
        neighbors = {};
        for i = 1:length(G.Edges)
            edge = G.Edges{i};
            if strcmp(edge.n1, u_name), neighbors{end+1} = edge.n2;
            elseif strcmp(edge.n2, u_name), neighbors{end+1} = edge.n1;
            end
        end
        neighbors = unique(neighbors);
        
        for i = 1:length(neighbors)
            v_name = neighbors{i};
            v_idx = name_to_idx(v_name);
            if ~any(Q == v_idx), continue; end
            
            edge_time = get_edge_time_3d(G, ZONES, u_name, v_name, t_start + dist(u_idx));
            alt = dist(u_idx) + edge_time;
            if alt < dist(v_idx)
                dist(v_idx) = alt;
                prev(v_idx) = u_idx;
            end
        end
    end
    
    path_nodes = [];
    curr_idx = end_idx;
    if prev(curr_idx) ~= 0 || curr_idx == start_idx
        while curr_idx ~= 0
            node_name = idx_to_name{curr_idx};
            path_nodes = [G.Nodes.(node_name); path_nodes];
            curr_idx = prev(curr_idx);
        end
    end
    total_time = dist(end_idx);
end


function [path_nodes_grid, total_time] = find_path_A_star_warehouse(start_node, end_node, WH_GRID, ZONES, t_start, res)
    global PARAMS;
    grid = WH_GRID.grid;
    [grid_h, grid_w] = size(grid);
    
    OPEN = zeros(20000, 7); 
    h_start = norm(end_node - start_node) * res / PARAMS.V_RESP_STRAIGHT;
    OPEN(1, :) = [start_node(1), start_node(2), 0, h_start, h_start, 0, 0];
    OPEN_count = 1;
    
    CLOSED = zeros(grid_h, grid_w); 
    G_cost = inf(grid_h, grid_w);
    G_cost(start_node(2), start_node(1)) = 0;
    Parent = zeros(grid_h, grid_w, 2); 
    
    moves = [ -1,0,1; 1,0,1; 0,-1,1; 0,1,1; -1,-1,sqrt(2); -1,1,sqrt(2); 1,-1,sqrt(2); 1,1,sqrt(2) ];
    path_nodes_grid = []; total_time = inf; found_path = false;

    while OPEN_count > 0
        [~, current_idx_in_open] = min(OPEN(1:OPEN_count, 5));
        current_node = OPEN(current_idx_in_open, :);
        cx = current_node(1); cy = current_node(2);
        
        OPEN(current_idx_in_open, :) = OPEN(OPEN_count, :);
        OPEN(OPEN_count, :) = 0; 
        OPEN_count = OPEN_count - 1;
        
        if CLOSED(cy, cx) == 1, continue; end
        CLOSED(cy, cx) = 1; 
        
        if cx == end_node(1) && cy == end_node(2)
            total_time = current_node(3); found_path = true; break; 
        end
        
        for i = 1:size(moves, 1)
            nx = cx + moves(i, 1); ny = cy + moves(i, 2);
            if nx < 1 || nx > grid_w || ny < 1 || ny > grid_h || grid(ny, nx) > 0 || CLOSED(ny, nx) == 1, continue; end
            
            segment_dist_m = moves(i, 3) * res;
            t_current = t_start + current_node(3); 
            v_dynamic = get_dynamic_speed(ZONES(1), t_current, PARAMS.V_RESP_STRAIGHT);
            segment_time = segment_dist_m / v_dynamic;
            new_g = current_node(3) + segment_time;
            
            if new_g < G_cost(ny, nx)
                G_cost(ny, nx) = new_g;
                Parent(ny, nx, :) = [cx, cy];
                h = norm(end_node - [nx, ny]) * res / PARAMS.V_RESP_STRAIGHT; 
                f = new_g + h;
                
                open_idx = 0;
                for j = 1:OPEN_count
                    if OPEN(j, 1) == nx && OPEN(j, 2) == ny, open_idx = j; break; end
                end
                
                if open_idx == 0 
                    OPEN_count = OPEN_count + 1;
                    if OPEN_count > size(OPEN, 1), OPEN = [OPEN; zeros(10000, 7)]; end
                    OPEN(OPEN_count, :) = [nx, ny, new_g, h, f, cx, cy];
                else
                    OPEN(open_idx, 3) = new_g; OPEN(open_idx, 5) = f;
                    OPEN(open_idx, 6) = cx; OPEN(open_idx, 7) = cy;
                end
            end
        end
    end
    
    if found_path
        path_nodes_grid = [end_node(1), end_node(2)];
        curr_p = end_node;
        while curr_p(1) ~= start_node(1) || curr_p(2) ~= start_node(2)
            parent_p = squeeze(Parent(curr_p(2), curr_p(1), :))';
            path_nodes_grid = [parent_p; path_nodes_grid];
            curr_p = parent_p;
            if isempty(curr_p) || (curr_p(1) == 0 && curr_p(2) == 0), break; end
        end
    else
        path_nodes_grid = []; total_time = inf;
    end
end


% ========================
% 3. SPEED MODELS
% ========================
function edge_time = get_edge_time_3d(G, ZONES, n1_name, n2_name, t_arrival_n1)
    edge = [];
    for i = 1:length(G.Edges)
        e = G.Edges{i};
        if (strcmp(e.n1, n1_name) && strcmp(e.n2, n2_name)) || (strcmp(e.n1, n2_name) && strcmp(e.n2, n1_name))
            edge = e; break;
        end
    end
    if isempty(edge), edge_time = inf; return; end
    
    [v_base, ~] = get_base_speed(G.Nodes.(n1_name), G.Nodes.(n2_name), edge.type);
    
    zone_id_1 = G.Nodes.(n1_name).zone_id;
    zone_id_2 = G.Nodes.(n2_name).zone_id;
    zone_to_check = [];
    if zone_id_1 > 0 && zone_id_2 > 0
        if ZONES(zone_id_1).T_crit < ZONES(zone_id_2).T_crit, zone_to_check = ZONES(zone_id_1);
        else, zone_to_check = ZONES(zone_id_2); end
    elseif zone_id_1 > 0, zone_to_check = ZONES(zone_id_1);
    elseif zone_id_2 > 0, zone_to_check = ZONES(zone_id_2);
    end
    
    v_dynamic = get_dynamic_speed(zone_to_check, t_arrival_n1, v_base);
    edge_time = edge.dist / v_dynamic;
end

function [v_base, edge_type] = get_base_speed(~, ~, edge_type)
    global PARAMS;
    if strcmp(edge_type, 'Stairs'), v_base = PARAMS.V_RESP_STAIRS; return; end
    v_base = PARAMS.V_RESP_STRAIGHT;
end

function v_dynamic = get_dynamic_speed(zone, t_current, v_base)
    global PARAMS;
    if isempty(zone), v_dynamic = v_base; return; end
    
    T_crit = zone.T_crit; R_max = zone.R_max; R_min = PARAMS.R_MIN;
    lambda = 3 / T_crit; 
    R_curr = (R_max - R_min) * exp(-lambda * t_current) + R_min;
    
    if R_curr >= 2.0, v_dynamic = v_base;
    else
        v_min_smoke = PARAMS.V_MIN_SMOKE;
        v_dynamic = v_min_smoke + (v_base - v_min_smoke) * (R_curr - R_min) / (2.0 - R_min);
        v_dynamic = max(v_min_smoke, v_dynamic); 
    end
end

function [T_sweep, sweep_path_pts] = calculate_sweep_time_v11(zone, arrival_time)
    global PARAMS;
    T_crit = zone.T_crit; R_max = zone.R_max; R_min = PARAMS.R_MIN; lambda = 3 / T_crit;
    R_arr = (R_max - R_min) * exp(-lambda * arrival_time) + R_min;
    v_arr = get_dynamic_speed(zone, arrival_time, PARAMS.V_RESP_STRAIGHT);

    W = zone.rect(3); H = zone.rect(4);
    door_pos = [W, H/2]; 
    if strcmp(zone.type, 'Hallway'), door_pos = [W/2, 0]; end
    padding = 0.1;

    if R_arr > PARAMS.R_THRESH_SWEEP
        path_nodes = [door_pos; padding, padding; W-padding, padding; W-padding, H-padding; padding, H-padding; door_pos];
        L_peri = 0;
        for i = 1:(size(path_nodes, 1) - 1)
            L_peri = L_peri + norm(path_nodes(i+1,:) - path_nodes(i,:));
        end
        T_sweep = L_peri / v_arr;
        sweep_path_pts = path_nodes;
    else
        delta = max(PARAMS.SWEEP_DELTA_MIN, PARAMS.SWEEP_KAPPA * R_arr); 
        y_levels = (0 + delta/2) : delta : (H - delta/2);
        if isempty(y_levels), y_levels = [H/2]; end
        
        L_zigzag = 0; prev_pt = door_pos; path_nodes = [door_pos];
        for k = 1:length(y_levels)
            y = y_levels(k);
            if mod(k, 2) == 1, pts = [padding, y; W-padding, y]; else, pts = [W-padding, y; padding, y]; end
            path_nodes = [path_nodes; pts];
            L_zigzag = L_zigzag + norm(pts(1,:) - prev_pt) + norm(pts(2,:) - pts(1,:));
            prev_pt = pts(2,:);
        end
        L_zigzag = L_zigzag + norm(prev_pt - door_pos);
        path_nodes = [path_nodes; door_pos];
        T_sweep = L_zigzag / (v_arr * PARAMS.SWEEP_ETA);
        sweep_path_pts = path_nodes;
    end
end

function [sweep_path_nodes, total_time] = calculate_warehouse_sweep_v11(start_grid_node, WH_GRID, ZONES, t_start, res, zone_index, total_zones)
    global PARAMS;
    grid = WH_GRID.grid; [grid_h, grid_w] = size(grid);
    
    zone_width_grid = floor(grid_w / total_zones);
    x_start_grid = (zone_index - 1) * zone_width_grid + 1;
    x_end_grid = zone_index * zone_width_grid;
    if zone_index == total_zones, x_end_grid = grid_w; end
    
    step_size = max(1, round(1.0 / res));
    sweep_path_nodes = [start_grid_node]; 
    total_time = 0; t_current = t_start;
    last_node_overall = start_grid_node; direction = 1; 
    
    for y = (grid_h - 1) : -step_size : 1
        if direction == 1, x_range = x_start_grid : 1 : x_end_grid; else, x_range = x_end_grid : -1 : x_start_grid; end
        last_node_in_row = []; 
        for x = x_range
            if grid(y, x) == 0 
                current_node = [x, y];
                if isempty(last_node_in_row)
                    [path_seg, time_seg] = find_path_A_star_warehouse(last_node_overall, current_node, WH_GRID, ZONES, t_current, res);
                    if ~isempty(path_seg)
                        sweep_path_nodes = [sweep_path_nodes; path_seg(2:end, :)];
                        total_time = total_time + time_seg; t_current = t_current + time_seg;
                    end
                else
                    dist_m = norm(current_node - last_node_in_row) * res;
                    v_dynamic = get_dynamic_speed(ZONES(1), t_current, PARAMS.V_RESP_STRAIGHT);
                    time_seg = dist_m / v_dynamic;
                    total_time = total_time + time_seg; t_current = t_current + time_seg;
                    sweep_path_nodes = [sweep_path_nodes; current_node];
                end
                last_node_in_row = current_node; last_node_overall = current_node;
            else
                last_node_in_row = [];
            end
        end
        direction = direction * -1;
    end
end


% ========================
% 4. UTILITY & PLOTTING
% ========================
function node = find_closest_walkable(coord_m, grid, res)
    start_node = round(coord_m / res);
    start_node(1) = max(1, min(size(grid, 2), start_node(1) + 1));
    start_node(2) = max(1, min(size(grid, 1), start_node(2) + 1));
    
    if grid(start_node(2), start_node(1)) == 0, node = start_node; return; end
    sz = 1;
    while sz < 20 
        for dx = -sz:sz
            for dy = -sz:sz
                if abs(dx) ~= sz && abs(dy) ~= sz, continue; end
                nx = start_node(1) + dx; ny = start_node(2) + dy;
                if nx < 1 || nx > size(grid, 2) || ny < 1 || ny > size(grid, 1), continue; end
                if grid(ny, nx) == 0, node = [nx, ny]; return; end
            end
        end
        sz = sz + 1;
    end
    node = start_node; 
end


function plot_simulation_3D_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, res, N_RESPONDERS)
    figure('Color', 'w', 'Name', '3D 救援路径仿真');
    hold on; grid on; axis equal; view(3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    col_wall = [0.1 0.1 0.1]; col_obs = [0.6 0.6 0.6];
    
    function draw_patch_3d(rect, z_base, h, col, alpha)
        x1=rect(1); y1=rect(2); x2=rect(1)+rect(3); y2=rect(2)+rect(4);
        z1=z_base; z2=z_base+h;
        v = [x1,y1,z1; x2,y1,z1; x2,y2,z1; x1,y2,z1; x1,y1,z2; x2,y1,z2; x2,y2,z2; x1,y2,z2];
        f = [1,2,3,4; 5,6,7,8; 1,2,6,5; 2,3,7,6; 3,4,8,7; 4,1,5,8];    
        patch('Vertices',v,'Faces',f,'FaceColor',col,'FaceAlpha',alpha,'EdgeColor','k','LineWidth',0.1);
    end

    for i = 1:length(ZONES)
        draw_patch_3d(ZONES(i).rect, ZONES(i).z_level, -0.05, [0.9 0.9 0.9], 0.8); 
    end
    for i=1:length(WH_OBSTACLES), draw_patch_3d(WH_OBSTACLES{i}, 0, 2.5, col_obs, 0.6); end

    if isempty(all_paths), max_resp_id = 1; else
        all_ids = cellfun(@(x) x.resp_id, all_paths); max_resp_id = max(max(all_ids), N_RESPONDERS); 
    end
    colors = lines(max(1, max_resp_id));

    for i = 1:length(all_paths)
        path = all_paths{i}; col = colors(path.resp_id, :);
        z_level = 0; if path.zone_id > 0, z_level = ZONES(path.zone_id).z_level; end
        
        for s = 1:length(path.segments)
            seg = path.segments{s};
            if isstruct(seg), pts = vertcat(seg.pos); plot3(pts(:,1), pts(:,2), pts(:,3), 'o-', 'Color', col);
            else
                pts_m = seg;
                if strcmp(path.target_name, 'Warehouse') || strncmp(path.target_name, 'WH', 2), pts_m = (seg - 0.5) * res; end
                
                % --- 修复：动态判断每个点的 Z ---
                pts_z = zeros(size(pts_m,1),1);
                for pi = 1:size(pts_m,1)
                    x = pts_m(pi,1); y = pts_m(pi,2);
                    z_guess = 0;
                    for zz = 1:length(ZONES)
                        r = ZONES(zz).rect;
                        if x>=r(1) && x<=r(1)+r(3) && y>=r(2) && y<=r(2)+r(4)
                            z_guess = ZONES(zz).z_level; break;
                        end
                    end
                    pts_z(pi) = z_guess;
                end
                plot3(pts_m(:,1), pts_m(:,2), pts_z, '.-', 'Color', col);
            end
        end
    end
    xlim([-2, 40]); ylim([-2, 24]); zlim([0, 6]);
end

function plot_2d_floor_plans_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, res, N_RESPONDERS)
    figure('Color', 'w', 'Name', '2D 路径平面图', 'Position', [200, 200, 1000, 500]);
    if isempty(all_paths), max_resp_id = 1; else, all_ids = cellfun(@(x) x.resp_id, all_paths); max_resp_id = max(max(all_ids), N_RESPONDERS); end
    colors = lines(max(1, max_resp_id));

    subplot(1, 2, 1); hold on; axis equal; title('Floor 1'); xlim([-2, 40]); ylim([-2, 24]);
    for i = 1:length(ZONES), if ZONES(i).z_level == 0, rectangle('Position', ZONES(i).rect, 'EdgeColor','k'); end, end
    for i=1:length(WH_OBSTACLES), rectangle('Position', WH_OBSTACLES{i}, 'FaceColor',[0.6 0.6 0.6]); end
    
    for i = 1:length(all_paths)
        path = all_paths{i}; col = colors(path.resp_id, :);
        z_level = 0; if path.zone_id > 0, z_level = ZONES(path.zone_id).z_level; end
        if z_level > 0, continue; end
        for s = 1:length(path.segments)
            seg = path.segments{s};
            if isstruct(seg), pts = vertcat(seg.pos); plot(pts(:,1), pts(:,2), 'Color', col);
            else, pts_m = (seg - 0.5) * res; plot(pts_m(:,1), pts_m(:,2), 'Color', col); end
        end
    end

    subplot(1, 2, 2); hold on; axis equal; title('Floor 2'); xlim([26, 40]); ylim([-2, 24]);
    for i = 1:length(ZONES), if ZONES(i).z_level > 0, rectangle('Position', ZONES(i).rect, 'EdgeColor','k'); end, end
    for i = 1:length(all_paths)
        path = all_paths{i}; col = colors(path.resp_id, :);
        z_level = 0; if path.zone_id > 0, z_level = ZONES(path.zone_id).z_level; end
        if z_level == 0, continue; end
        for s = 1:length(path.segments)
            seg = path.segments{s};
            if isstruct(seg), pts = vertcat(seg.pos); plot(pts(:,1), pts(:,2), 'Color', col);
            else, pts_m = seg; plot(pts_m(:,1), pts_m(:,2), 'Color', col); end
        end
    end
end