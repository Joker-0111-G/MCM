function rescue_simulation_hybrid_v11_FINAL()
    % ======================================
    % 主函数：复杂建筑救援仿真 (a31.m 基础) - V11 (最终版)
    % 
    % V11 最终更新:
    % 1. (FIX) 修复 V9 的绘图 bug (N_RESPONDERS 变量作用域问题).
    % 2. (ADD) plot_2d_floor_plans_v11 现在接收所有仿真数据.
    % 3. (ADD) 2D 平面图 (Figure 2) 现在也会绘制所有响应者的
    %           一层和二层路径.
    % ======================================
    clc; clear; close all;
    
    % --- 1. 仿真输入 ---
    N_RESPONDERS = 6;           % V9: 响应者总数 (现在完全可变)
    N_WAREHOUSE_RESPONDERS = 4; % V9: 希望并行扫描仓库的响应者数量
    START_DELAY = 60.0;         % 响应延迟 (秒)
    RESOLUTION = 0.25;          % 网格分辨率 (米)
    
    fprintf('--- 启动混合救援仿真 (V11 - 3D + 2D 路径图) ---\n');
    fprintf('响应者总数: %d, 仓库并行数: %d, 延迟: %.1f s\n', ...
             N_RESPONDERS, N_WAREHOUSE_RESPONDERS, START_DELAY);

    % --- 1.2 定义几何常量 ---
    X_offset = 28;            
    shop_h = 16/3; 
    room_h = 4;
    
    % --- 1.1 定义全局参数 ---
    global PARAMS;
    PARAMS.V_RESP_STRAIGHT = 1.2;
    PARAMS.V_RESP_TURN = 1.0;
    PARAMS.V_RESP_STAIRS = 1.1;
    PARAMS.V_MIN_SMOKE = 0.2; 
    PARAMS.R_MIN = 0.5;
    PARAMS.R_THRESH_SWEEP = 2.5;
    PARAMS.SWEEP_ETA = 0.3;
    PARAMS.SWEEP_KAPPA = 1.4;
    PARAMS.SWEEP_DELTA_MIN = 0.5;
    PARAMS.FLOOR_HEIGHT = 2.5; 
    
    % --- 2. 构建环境 ---
    fprintf('正在构建 3D 环境 (V11)...\n');
    [ZONES, BUILDING_GRAPH, WAREHOUSE_GRID, WH_OBSTACLES, ALL_ENTRY_NODES] = build_environment_v11(RESOLUTION);
    fprintf('环境构建完毕. 区域数: %d, 入口数: %d\n', length(ZONES), length(ALL_ENTRY_NODES));

    % --- 3. 定义救援任务 (V9 逻辑) ---
    % 任务格式: { 'Name', 'Zone', [x,y], 'Type', ZoneIdx (仅用于Sweep) }
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
    
    % V9: 动态生成并行仓库扫描任务
    tasks_data_wh_sweep = cell(N_WAREHOUSE_RESPONDERS, 5);
    for z = 1:N_WAREHOUSE_RESPONDERS
        tasks_data_wh_sweep(z,:) = {sprintf('WH Sweep Zone %d', z), 'Warehouse', [14, 11], 'Sweep', z};
    end

    % 组合任务: 优先 -> 扫楼 -> 仓库扫描
    tasks_data = [
        tasks_data_static(1:5,:);   % 5 个高优先级
        tasks_data_static(6:9,:);   % 4 个中优先级 (扫楼)
        tasks_data_wh_sweep         % 低优先级 (仓库并行扫描)
    ];
    
    fprintf('任务生成完毕: %d 个任务 (包括 %d 个并行仓库扫描).\n', size(tasks_data, 1), N_WAREHOUSE_RESPONDERS);

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
        TARGETS(i).zone_idx_for_sweep = tasks_data{i, 5}; % V9 新增
        
        if strcmp(ZONES(zone_idx).type, 'Warehouse')
            node = find_closest_walkable(TARGETS(i).pos_m, WAREHOUSE_GRID.grid, RESOLUTION);
            TARGETS(i).pos_grid = node;
        else
            TARGETS(i).pos_grid = []; 
        end
    end

    % --- 4. 仿真执行 ---
    fprintf('开始分配任务并规划 3D 路径...\n');
    
    % V9: 动态分配起始点
    responders = struct('id', {}, 'start_node', {}, 'time_free', {}, 'log', {});
    num_entries = length(ALL_ENTRY_NODES);
    for i = 1:N_RESPONDERS
        responders(i).id = i;
        % 循环分配入口
        entry_idx = mod(i-1, num_entries) + 1;
        responders(i).start_node = ALL_ENTRY_NODES{entry_idx};
        responders(i).time_free = START_DELAY; 
        fprintf('  R%d 在 %s 准备就绪.\n', i, responders(i).start_node);
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
        
        fprintf('  -> [T=%.0fs] R%d 分配: %s (在 %s)\n', t_start, resp_idx, target.name, target_zone.name);

        % --- 路径计算 (V9 逻辑) ---
        
        % 1. 决定高层图目标节点
        high_level_target_node = '';
        if strcmp(target_zone.type, 'Warehouse')
            high_level_target_node = target_zone.graph_node; % 'Warehouse'
        else
            high_level_target_node = [target_zone.graph_node, '_Door']; % 'Apt4_Door'
        end

        % 2. 高层图路径 (Dijkstra)
        [path_high, time_high_level] = find_path_high_level_3d(responders(resp_idx).start_node, ...
                                            high_level_target_node, BUILDING_GRAPH, ZONES, t_start);
        
        if isinf(time_high_level)
            fprintf('     !! 严重错误: 无法从 %s 找到路径到 %s.\n', ...
                responders(resp_idx).start_node, high_level_target_node);
            continue; 
        end
        
        % 3. 低层路径
        
        % --- 仓库 (A* 或 弓字形) ---
        if strcmp(target_zone.type, 'Warehouse')
            
            start_node_name = responders(resp_idx).start_node;
            entry_door_node_name = '';
            
            % 3.1 找到 A* 起点 (物理门)
            if length(path_high) == 1
                % 情况1: 响应者已在 'Warehouse' (逻辑节点)
                target_pos_m = (target.pos_grid - 0.5) * RESOLUTION;
                wh_door_names = {'WH_Door_Left', 'WH_Door_Top', 'WH_Door_Bottom', 'WH_Door_Stairs1', 'WH_Door_Stairs2'};
                best_door_name = '';
                min_dist = inf;
                
                for i = 1:length(wh_door_names)
                    door_name = wh_door_names{i};
                    door_pos_m = BUILDING_GRAPH.Nodes.(door_name).pos(1:2); % 2D pos
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
                fprintf('     高层图 (WH内部->门) 耗时: %.1f s (T_arr_zone: %.1f s)\n', time_to_door, t_arr_zone);
            
            else
                % 情况2: 响应者从外部进入
                entry_door_node_name = path_high(end-1).name; 
                t_arr_zone = t_start + time_high_level; 
                task_total_time = time_high_level;
                path_segments = {path_high};
                fprintf('     高层图 (外部->门) 耗时: %.1f s (T_arr_zone: %.1f s)\n', time_high_level, t_arr_zone);
            end
            
            wh_entry_pos_m = BUILDING_GRAPH.Nodes.(entry_door_node_name).pos;
            wh_entry_grid = find_closest_walkable(wh_entry_pos_m(1:2), WAREHOUSE_GRID.grid, RESOLUTION);
            
            % 3.2 根据任务类型选择 A* 还是 弓字形
            if strcmp(target.type, 'Search')
                % --- A* 寻路到关键点 ---
                target_grid = target.pos_grid;
                fprintf('     仓库内部 A* (T_arr: %.1f s): \n', t_arr_zone);
                fprintf('       从 %s (Grid: %d, %d)\n', entry_door_node_name, wh_entry_grid(1), wh_entry_grid(2));
                fprintf('       到 %s (Grid: %d, %d)\n', target.name, target_grid(1), target_grid(2));

                [path_low_nodes, time_to_target_low] = find_path_A_star_warehouse(wh_entry_grid, ...
                                                        target_grid, WAREHOUSE_GRID, ZONES, t_arr_zone, RESOLUTION);
                if isempty(path_low_nodes)
                    fprintf('     !! 警告: 仓库A* 未找到路径到 %s!!\n', target.name);
                    continue;
                end
                
                t_arr_person = t_arr_zone + time_to_target_low; 
                task_total_time = task_total_time + time_to_target_low;
                path_segments{end+1} = path_low_nodes;
                
                fprintf('     仓库内部 A* 耗时: %.1f s (T_arr_search_pt: %.1f s)\n', time_to_target_low, t_arr_person);
                sweep_time = 5.0; % 搜索点确认时间
            
            elseif strcmp(target.type, 'Sweep')
                % --- V9 弓字形 (Zigzag) 并行扫描 ---
                zone_index = target.zone_idx_for_sweep;
                fprintf('     仓库内部 Zigzag 扫描 [Zone %d/%d] (T_arr: %.1f s): \n', ...
                    zone_index, N_WAREHOUSE_RESPONDERS, t_arr_zone);
                fprintf('       从 %s (Grid: %d, %d) 开始\n', entry_door_node_name, wh_entry_grid(1), wh_entry_grid(2));

                % 1. A* 寻路到扫描起点
                [grid_h, grid_w] = size(WAREHOUSE_GRID.grid);
                % V9: 动态计算此区域的 x_start
                zone_width_grid = floor(grid_w / N_WAREHOUSE_RESPONDERS);
                x_start_grid = (zone_index - 1) * zone_width_grid + 3;
                
                sweep_start_grid = [x_start_grid, grid_h-3]; % 此区域的左上角
                
                [path_to_sweep_start, time_to_sweep_start] = find_path_A_star_warehouse(wh_entry_grid, ...
                                                        sweep_start_grid, WAREHOUSE_GRID, ZONES, t_arr_zone, RESOLUTION);
                if isempty(path_to_sweep_start)
                    fprintf('     !! 警告: 无法 A* 到扫描起点 !!\n');
                    continue;
                end
                
                t_arr_sweep_start = t_arr_zone + time_to_sweep_start;
                task_total_time = task_total_time + time_to_sweep_start;
                path_segments{end+1} = path_to_sweep_start;
                fprintf('     A* 到扫描起点耗时: %.1f s\n', time_to_sweep_start);
                
                % 2. 执行弓字形扫描
                [sweep_path_nodes, sweep_time] = calculate_warehouse_sweep_v11(sweep_start_grid, ...
                                                    WAREHOUSE_GRID, ZONES, t_arr_sweep_start, RESOLUTION, ...
                                                    zone_index, N_WAREHOUSE_RESPONDERS);
                
                task_total_time = task_total_time + sweep_time;
                path_segments{end+1} = sweep_path_nodes;
                t_arr_person = t_arr_sweep_start + sweep_time; % 扫描结束时间
                fprintf('     仓库 Zigzag (Zone %d) 扫描耗时: %.1f s (T_end_sweep: %.1f s)\n', ...
                    zone_index, sweep_time, t_arr_person);
            end

            task_total_time = task_total_time + sweep_time;

        % --- Apt/Shop (周界) ---
        else 
            t_arr_zone = t_start + time_high_level;
            task_total_time = time_high_level;
            path_segments = {path_high};

            fprintf('     高层图 (->门) 耗时: %.1f s (T_arr_door: %.1f s)\n', time_high_level, t_arr_zone);
            
            [sweep_time, sweep_path_local] = calculate_sweep_time_v11(target_zone, t_arr_zone);
            
            t_arr_person = t_arr_zone + sweep_time;
            task_total_time = task_total_time + sweep_time;

            sweep_path_global = sweep_path_local;
            sweep_path_global(:,1) = sweep_path_local(:,1) + target_zone.rect(1);
            sweep_path_global(:,2) = sweep_path_local(:,2) + target_zone.rect(2);
            
            path_segments{end+1} = sweep_path_global;
            
            fprintf('     周界搜索耗时: %.1f s (T_arr_person: %.1f s)\n', sweep_time, t_arr_person);
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
        
        fprintf('     任务 %s 完成. R%d 可用 T=%.0f s\n', ...
            target.name, resp_idx, responders(resp_idx).time_free);
    end
    
    % --- 5. 退出逻辑 ---
    fprintf('--- 所有任务完成, 开始计算退出路径 ---\n');
    
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
            
        fprintf('  -> R%d 从 %s 退出, 耗时: %.1f s. 最终 T=%.0f s\n', ...
            resp.id, start_node, time_to_exit, responders(i).time_free);
    end
    
    % --- 6. 结果汇总与绘图 ---
    fprintf('--- 仿真结束 ---\n');
    total_time = max([responders.time_free]) - START_DELAY;
    fprintf('所有救援+退出完成. 总救援耗时 (从响应开始): %.2f s\n', total_time);
    for i=1:N_RESPONDERS
        fprintf('响应者 %d 结束时间: %.2f s (绝对时间)\n', i, responders(i).time_free);
    end
    
    % --- V10-Fix: 绘图函数作用域错误 ---
    plot_simulation_3D_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, RESOLUTION, N_RESPONDERS);
    
    % --- V11-Add: 调用 2D 平面图 (传入所有路径) ---
    plot_2d_floor_plans_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, RESOLUTION, N_RESPONDERS);
    % --- End V11-Add ---
    
end

% ===================================================================
% ===================================================================
%                        辅助函数 (HELPER FUNCTIONS)
% ===================================================================
% ===================================================================


% ========================
% 1. BUILD_ENVIRONMENT (V11)
% ========================
function [ZONES, BUILDING_GRAPH, WAREHOUSE_GRID, WH_OBSTACLES, ALL_ENTRY_NODES] = build_environment_v11(res)
    % (V10: 重命名)
    global PARAMS;
    Z_F1 = 0.0; 
    Z_F2 = PARAMS.FLOOR_HEIGHT; 

    ZONES = struct('id', {}, 'name', {}, 'rect', {}, 'type', {}, ...
                   'T_crit', {}, 'R_max', {}, 'graph_node', {}, 'z_level', {});
    
    W_wh = 28; H_wh = 22; X_offset = 28;
    shop_w = 9; shop_h = 16/3; 
    room_w = 6; room_h = 4; hall_w = 3;

    % Z-1: 仓库
    ZONES(end+1).id = 1; ZONES(end).name = 'Warehouse'; ZONES(end).rect = [0, 0, W_wh, H_wh];
    ZONES(end).type = 'Warehouse'; ZONES(end).T_crit = 30 * 60; ZONES(end).R_max = norm([W_wh, H_wh]);
    ZONES(end).graph_node = 'Warehouse'; ZONES(end).z_level = Z_F1;
    
    % Z-2,3,4: 商铺
    for i = 1:3
        y_s = 3 + (i-1)*shop_h;
        ZONES(end+1).id = 1+i; ZONES(end).name = sprintf('Shop %d', i); ZONES(end).rect = [X_offset, y_s, shop_w, shop_h];
        ZONES(end).type = 'Shop'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([shop_w, shop_h]);
        ZONES(end).graph_node = sprintf('Shop%d', i); ZONES(end).z_level = Z_F1;
    end

    % Z-5,6,7,8: 公寓
    for i = 1:4
        y_a = 3 + (i-1)*room_h;
        ZONES(end+1).id = 4+i; ZONES(end).name = sprintf('Apt %d', i); ZONES(end).rect = [X_offset, y_a, room_w, room_h];
        ZONES(end).type = 'Apt'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([room_w, room_h]);
        ZONES(end).graph_node = sprintf('Apt%d', i); ZONES(end).z_level = Z_F2;
    end
    
    % Z-9: 楼道
    ZONES(end+1).id = 9; ZONES(end).name = 'Hallway'; ZONES(end).rect = [X_offset + room_w, 3, hall_w, 16];
    ZONES(end).type = 'Hallway'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([hall_w, 16]);
    ZONES(end).graph_node = 'Hallway'; ZONES(end).z_level = Z_F2;

    % Z-10, 11: 楼梯
    ZONES(end+1).id = 10; ZONES(end).name = 'Stairs 1'; ZONES(end).rect = [X_offset, 0, 9, 3];
    ZONES(end).type = 'Stairs'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([9, 3]);
    ZONES(end).graph_node = 'Stairs1'; ZONES(end).z_level = Z_F1;
    
    ZONES(end+1).id = 11; ZONES(end).name = 'Stairs 2'; ZONES(end).rect = [X_offset, 19, 9, 3];
    ZONES(end).type = 'Stairs'; ZONES(end).T_crit = 6 * 60; ZONES(end).R_max = norm([9, 3]);
    ZONES(end).graph_node = 'Stairs2'; ZONES(end).z_level = Z_F1;

    
    % --- 1.2 构建 WAREHOUSE_GRID ---
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
        if zone_id > 0
            gz(y1:y2, x1:x2) = zone_id;
        end
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

    % --- 1.3 构建 BUILDING_GRAPH (3D 坐标) ---
    G.Nodes = struct();
    G.Edges = {};
    ALL_ENTRY_NODES = {}; % V9: 用于返回
    
    function G_Nodes = add_node(G_Nodes, name, pos_3d, zone_id, type)
        G_Nodes.(name).name = name;
        G_Nodes.(name).pos = pos_3d; % [x, y, z]
        G_Nodes.(name).zone_id = zone_id;
        G_Nodes.(name).type = type; 
    end
    
    function G_Edges = add_edge(G_Edges, G_Nodes, n1_name, n2_name, type)
        n1 = G_Nodes.(n1_name);
        n2 = G_Nodes.(n2_name);
        if strcmp(type, 'Stairs')
            dist = norm(n1.pos(1:2) - n2.pos(1:2)) + abs(n1.pos(3) - n2.pos(3));
        else
            dist = norm(n1.pos - n2.pos);
        end
        G_Edges{end+1} = struct('n1', n1_name, 'n2', n2_name, 'dist', dist, 'type', type); 
    end

    % 1. 入口节点 (外部) (Z=0)
    G.Nodes = add_node(G.Nodes, 'Entry_WH_Left', [-1, 11, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_WH_Top', [14, 23, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_WH_Bottom', [14, -1, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Shop1', [X_offset+shop_w+1, 3+shop_h/2, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Shop2', [X_offset+shop_w+1, 3+shop_h*1.5, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Shop3', [X_offset+shop_w+1, 3+shop_h*2.5, Z_F1], 0, 'Entry');
    G.Nodes = add_node(G.Nodes, 'Entry_Stairs1', [X_offset+shop_w+1, 1.5, Z_F1], 0, 'Entry'); 
    G.Nodes = add_node(G.Nodes, 'Entry_Stairs2', [X_offset+shop_w+1, 20.5, Z_F1], 0, 'Entry');
    
    % 2. 内部逻辑节点 (Zone 的 "中心")
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
    
    % 3. "门" 节点
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

    % === [新增 F2 楼道与楼梯的连接门] ===
    % X = X_offset + room_w + hall_w/2 = 35.5 (楼道中心)
    G.Nodes = add_node(G.Nodes, 'Hall_Door_Bottom', [X_offset+room_w+hall_w/2, 3, Z_F2], 9, 'Door');
    G.Nodes = add_node(G.Nodes, 'Hall_Door_Top', [X_offset+room_w+hall_w/2, 19, Z_F2], 9, 'Door');

    % 4. 定义边 (连接)
    all_nodes = fieldnames(G.Nodes);
    entry_nodes = {};
    for i = 1:length(all_nodes)
        if strcmp(G.Nodes.(all_nodes{i}).type, 'Entry')
            entry_nodes{end+1} = all_nodes{i};
        end
    end
    % V9: 赋值给返回变量
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
% 2. PATHFINDING (Dijkstra, A*)
% ========================

function [path_nodes, total_time] = find_path_high_level_3d(start_node_name, end_node_name, G, ZONES, t_start)
    % (无变化)
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
        
        if u_idx == end_idx
            break; 
        end
        
        u_name = idx_to_name{u_idx};
        
        neighbors = {};
        for i = 1:length(G.Edges)
            edge = G.Edges{i};
            if strcmp(edge.n1, u_name)
                neighbors{end+1} = edge.n2;
            elseif strcmp(edge.n2, u_name)
                neighbors{end+1} = edge.n1;
            end
        end
        neighbors = unique(neighbors);
        
        for i = 1:length(neighbors)
            v_name = neighbors{i};
            v_idx = name_to_idx(v_name);
            
            if ~any(Q == v_idx)
                continue;
            end
            
            edge_time = get_edge_time_3d(G, ZONES, u_name, v_name, t_start + dist(u_idx));
            
            alt = dist(u_idx) + edge_time;
            if alt < dist(v_idx)
                dist(v_idx) = alt;
                prev(v_idx) = u_idx;
            end
        end
    end
    
    path_nodes = [];
    path_names = {};
    curr_idx = end_idx;
    if prev(curr_idx) ~= 0 || curr_idx == start_idx
        while curr_idx ~= 0
            node_name = idx_to_name{curr_idx};
            path_nodes = [G.Nodes.(node_name); path_nodes];
            path_names = [node_name; path_names];
            curr_idx = prev(curr_idx);
        end
    end
    total_time = dist(end_idx);
end


function [path_nodes_grid, total_time] = find_path_A_star_warehouse(start_node, end_node, WH_GRID, ZONES, t_start, res)
    % (无变化)
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
    
    moves = [
        -1,  0, 1; 1,  0, 1; 0, -1, 1; 0,  1, 1;
        -1, -1, sqrt(2); -1,  1, sqrt(2); 1, -1, sqrt(2); 1,  1, sqrt(2)
    ];
    
    path_nodes_grid = [];
    total_time = inf;
    found_path = false;

    while OPEN_count > 0
        [~, current_idx_in_open] = min(OPEN(1:OPEN_count, 5));
        current_node = OPEN(current_idx_in_open, :);
        cx = current_node(1); cy = current_node(2);
        
        OPEN(current_idx_in_open, :) = OPEN(OPEN_count, :);
        OPEN(OPEN_count, :) = 0; 
        OPEN_count = OPEN_count - 1;
        
        if CLOSED(cy, cx) == 1
            continue; 
        end
        CLOSED(cy, cx) = 1; 
        
        if cx == end_node(1) && cy == end_node(2)
            total_time = current_node(3); 
            found_path = true;
            break; 
        end
        
        for i = 1:size(moves, 1)
            nx = cx + moves(i, 1);
            ny = cy + moves(i, 2);
            
            if nx < 1 || nx > grid_w || ny < 1 || ny > grid_h
                continue;
            end
            if grid(ny, nx) == 1 || grid(ny, nx) == 2
                continue;
            end
            if CLOSED(ny, nx) == 1
                continue;
            end
            
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
                    if OPEN(j, 1) == nx && OPEN(j, 2) == ny
                        open_idx = j;
                        break;
                    end
                end
                
                if open_idx == 0 
                    OPEN_count = OPEN_count + 1;
                    if OPEN_count > size(OPEN, 1)
                        OPEN = [OPEN; zeros(10000, 7)];
                    end
                    OPEN(OPEN_count, :) = [nx, ny, new_g, h, f, cx, cy];
                else
                    OPEN(open_idx, 3) = new_g;
                    OPEN(open_idx, 5) = f;
                    OPEN(open_idx, 6) = cx;
                    OPEN(open_idx, 7) = cy;
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
            if isempty(curr_p) || (curr_p(1) == 0 && curr_p(2) == 0)
                break; 
            end
        end
    else
        path_nodes_grid = []; 
        total_time = inf;
    end
end


% ========================
% 3. TIME & SPEED MODELS
% ========================

function edge_time = get_edge_time_3d(G, ZONES, n1_name, n2_name, t_arrival_n1)
    % (无变化)
    global PARAMS;
    
    edge = [];
    for i = 1:length(G.Edges)
        e = G.Edges{i};
        if (strcmp(e.n1, n1_name) && strcmp(e.n2, n2_name)) || ...
           (strcmp(e.n1, n2_name) && strcmp(e.n2, n1_name))
            edge = e;
            break;
        end
    end
    if isempty(edge)
        edge_time = inf;
        return;
    end
    
    [v_base, ~] = get_base_speed(G.Nodes.(n1_name), G.Nodes.(n2_name), edge.type);
    
    zone_id_1 = G.Nodes.(n1_name).zone_id;
    zone_id_2 = G.Nodes.(n2_name).zone_id;
    
    zone_to_check = [];
    if zone_id_1 > 0 && zone_id_2 > 0
        if ZONES(zone_id_1).T_crit < ZONES(zone_id_2).T_crit
            zone_to_check = ZONES(zone_id_1);
        else
            zone_to_check = ZONES(zone_id_2);
        end
    elseif zone_id_1 > 0
        zone_to_check = ZONES(zone_id_1);
    elseif zone_id_2 > 0
        zone_to_check = ZONES(zone_id_2);
    else
        zone_to_check = [];
    end
    
    v_dynamic = get_dynamic_speed(zone_to_check, t_arrival_n1, v_base);
    edge_time = edge.dist / v_dynamic;
end

function [v_base, edge_type] = get_base_speed(~, ~, edge_type)
    % (无变化)
    global PARAMS;
    if strcmp(edge_type, 'Stairs')
        v_base = PARAMS.V_RESP_STAIRS;
        return;
    end
    v_base = PARAMS.V_RESP_STRAIGHT;
end

function v_dynamic = get_dynamic_speed(zone, t_current, v_base)
    % (无变化)
    global PARAMS;
    
    if isempty(zone)
        v_dynamic = v_base; 
        return;
    end
    
    T_crit = zone.T_crit;
    R_max = zone.R_max;
    R_min = PARAMS.R_MIN;
    lambda = 3 / T_crit; 
    
    R_curr = (R_max - R_min) * exp(-lambda * t_current) + R_min;
    
    if R_curr >= 2.0
        v_dynamic = v_base;
    else
        v_min_smoke = PARAMS.V_MIN_SMOKE;
        v_dynamic = v_min_smoke + (v_base - v_min_smoke) * (R_curr - R_min) / (2.0 - R_min);
        v_dynamic = max(v_min_smoke, v_dynamic); 
    end
end


function [T_sweep, sweep_path_pts] = calculate_sweep_time_v11(zone, arrival_time)
    % (V11: 重命名)
    global PARAMS;
    
    T_crit = zone.T_crit;
    R_max = zone.R_max;
    R_min = PARAMS.R_MIN;
    lambda = 3 / T_crit;
    
    R_arr = (R_max - R_min) * exp(-lambda * arrival_time) + R_min;
    v_base = PARAMS.V_RESP_STRAIGHT;
    v_arr = get_dynamic_speed(zone, arrival_time, v_base);

    W = zone.rect(3);
    H = zone.rect(4);
    
    if (strcmp(zone.type, 'Apt'))
        door_pos = [W, H/2]; 
    elseif (strcmp(zone.type, 'Shop'))
        door_pos = [W, H/2]; 
    else
        door_pos = [W/2, 0]; 
    end
    
    padding = 0.1; % 不贴墙

    if R_arr > PARAMS.R_THRESH_SWEEP
        % 模式 A: 周界
        c1 = [padding, padding]; 
        c2 = [W-padding, padding]; 
        c3 = [W-padding, H-padding]; 
        c4 = [padding, H-padding]; 
        
        path_nodes = [door_pos; c1; c2; c3; c4; door_pos];
        
        L_peri = 0;
        for i = 1:(size(path_nodes, 1) - 1)
            L_peri = L_peri + norm(path_nodes(i+1,:) - path_nodes(i,:));
        end
        
        T_sweep = L_peri / v_arr;
        sweep_path_pts = path_nodes;
        
    else
        % 模式 B: Zigzag
        delta_min = PARAMS.SWEEP_DELTA_MIN;
        kappa = PARAMS.SWEEP_KAPPA;
        eta = PARAMS.SWEEP_ETA;
        
        delta = max(delta_min, kappa * R_arr); 
        
        y_levels = (0 + delta/2) : delta : (H - delta/2);
        if isempty(y_levels), y_levels = [H/2]; end
        if y_levels(end) < (H - delta)
             y_levels = [y_levels, H - delta/2];
        end

        L_zigzag = 0;
        prev_pt = door_pos;
        path_nodes = [door_pos];
        
        for k = 1:length(y_levels)
            y = y_levels(k);
            x_left = 0 + padding; x_right = W - padding;
            
            if mod(k, 2) == 1
                pts = [x_left, y; x_right, y]; 
            else
                pts = [x_right, y; x_left, y]; 
            end
            
            path_nodes = [path_nodes; pts];
            L_zigzag = L_zigzag + norm(pts(1,:) - prev_pt);
            L_zigzag = L_zigzag + norm(pts(2,:) - pts(1,:));
            prev_pt = pts(2,:);
        end
        
        L_zigzag = L_zigzag + norm(prev_pt - door_pos);
        path_nodes = [path_nodes; door_pos];
        
        T_sweep = L_zigzag / (v_arr * eta);
        sweep_path_pts = path_nodes;
    end
end


function [sweep_path_nodes, total_time] = calculate_warehouse_sweep_v11(start_grid_node, WH_GRID, ZONES, t_start, res, zone_index, total_zones)
    % (V11: 重命名)
    
    global PARAMS;
    grid = WH_GRID.grid;
    [grid_h, grid_w] = size(grid);
    
    % V9: 计算此区域的 X 边界
    zone_width_grid = floor(grid_w / total_zones);
    x_start_grid = (zone_index - 1) * zone_width_grid + 1;
    x_end_grid = zone_index * zone_width_grid;
    if zone_index == total_zones
        x_end_grid = grid_w; % 确保最后一个区域覆盖所有剩余部分
    end
    
    sweep_resolution_m = 1.0; 
    step_size = max(1, round(sweep_resolution_m / res));
    
    sweep_path_nodes = [start_grid_node]; 
    total_time = 0;
    
    t_current = t_start;
    last_node_overall = start_grid_node; 

    direction = 1; 
    
    for y = (grid_h - 1) : -step_size : 1
        
        x_range = [];
        % V9: 只在分配的 X 区域内扫描
        if direction == 1
            x_range = x_start_grid : 1 : x_end_grid; % L -> R
        else
            x_range = x_end_grid : -1 : x_start_grid; % R -> L
        end
        
        last_node_in_row = []; 
            
        for x = x_range
            if grid(y, x) == 0 
                current_node = [x, y];
                
                if isempty(last_node_in_row)
                    % --- "跳跃" ---
                    [path_seg, time_seg] = find_path_A_star_warehouse(last_node_overall, ...
                                               current_node, WH_GRID, ZONES, t_current, res);
                    
                    if isempty(path_seg)
                        continue; 
                    end
                    
                    sweep_path_nodes = [sweep_path_nodes; path_seg(2:end, :)];
                    total_time = total_time + time_seg;
                    t_current = t_current + time_seg;

                else
                    % --- "连续" ---
                    dist_m = norm(current_node - last_node_in_row) * res;
                    v_dynamic = get_dynamic_speed(ZONES(1), t_current, PARAMS.V_RESP_STRAIGHT);
                    time_seg = dist_m / v_dynamic;
                    
                    total_time = total_time + time_seg;
                    t_current = t_current + time_seg;
                    sweep_path_nodes = [sweep_path_nodes; current_node];
                end
                
                last_node_in_row = current_node;
                last_node_overall = current_node;

            else
                % 撞墙
                last_node_in_row = [];
            end
        end
        
        direction = direction * -1;
    end
end


% ========================
% 4. UTILITY & PLOTTING (V11)
% ========================

function node = find_closest_walkable(coord_m, grid, res)
    % (无变化)
    start_node = round(coord_m / res);
    start_node(1) = max(1, min(size(grid, 2), start_node(1) + 1));
    start_node(2) = max(1, min(size(grid, 1), start_node(2) + 1));
    
    if grid(start_node(2), start_node(1)) == 0
        node = start_node; 
        return;
    end
    
    sz = 1;
    while sz < 20 
        for dx = -sz:sz
            for dy = -sz:sz
                if abs(dx) ~= sz && abs(dy) ~= sz
                    continue; 
                end
                
                nx = start_node(1) + dx;
                ny = start_node(2) + dy;
                
                if nx < 1 || nx > size(grid, 2) || ny < 1 || ny > size(grid, 1)
                    continue;
                end
                
                if grid(ny, nx) == 0
                    node = [nx, ny];
                    return;
                end
            end
        end
        sz = sz + 1;
    end
    node = start_node; 
end


function plot_simulation_3D_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, res, N_RESPONDERS)
    % (V10: 重命名, V9-Fix)
    
    global PARAMS;
    H_FLOOR = PARAMS.FLOOR_HEIGHT;
    H_WALL = PARAMS.FLOOR_HEIGHT; 
    H_OBS = PARAMS.FLOOR_HEIGHT;  

    figure('Color', 'w', 'Name', '3D 救援路径仿真 (V11)', 'Position', [100, 100, 1400, 900]);
    hold on; grid on; axis equal;
    title('3D 混合路径救援仿真 (V11 - 并行扫描)', 'FontSize', 14);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    view(3); 
    
    col_wall = [0.1 0.1 0.1];
    col_door = [1 0 0];
    col_obs = [0.6 0.6 0.6];
    col_person_inj = [1 0 0]; 
    col_person_norm = [0 0 1]; 
    col_stair = [0.95 0.95 0.8];
    col_floor = [0.9 0.9 0.9];
    
    % --- 3D 绘制助手 ---
    function draw_patch_3d(rect, z_base, h, col, alpha)
        x1=rect(1); y1=rect(2); x2=rect(1)+rect(3); y2=rect(2)+rect(4);
        z1=z_base; z2=z_base+h;
        v = [x1, y1, z1; x2, y1, z1; x2, y2, z1; x1, y2, z1; ...
             x1, y1, z2; x2, y1, z2; x2, y2, z2; x1, y2, z2];
        f = [1, 2, 3, 4; 5, 6, 7, 8; 1, 2, 6, 5; ...
             2, 3, 7, 6; 3, 4, 8, 7; 4, 1, 5, 8];    
        patch('Vertices', v, 'Faces', f, 'FaceColor', col, 'FaceAlpha', alpha, 'EdgeColor', 'k', 'LineWidth', 0.1);
    end

    % --- 1. 绘制 3D 环境 ---
    for i = 1:length(ZONES)
        r = ZONES(i).rect;
        z = ZONES(i).z_level;
        col = col_floor;
        if strcmp(ZONES(i).type, 'Stairs'), col = col_stair; end
        draw_patch_3d(r, z, -0.05, col, 0.8); 
    end
    
    X_offset = 28; room_w = 6;
    draw_patch_3d([0, 0, 28, 22], 0, H_WALL, col_wall, 0.05); 
    draw_patch_3d([28, 0, 9, 22], 0, H_WALL, col_wall, 0.05); 
    draw_patch_3d([28, 3, 6, 16], H_FLOOR, H_WALL, col_wall, 0.1); 
    for i = 1:3
        y_a = 3 + i*4;
        draw_patch_3d([X_offset, y_a, room_w, 0.1], H_FLOOR, H_WALL, col_wall, 0.1);
    end
    
    for i=1:length(WH_OBSTACLES)
        draw_patch_3d(WH_OBSTACLES{i}, 0, H_OBS, col_obs, 0.6);
    end

    % --- 2. 绘制 3D 目标点 ---
    for i = 1:length(TARGETS)
        pos = TARGETS(i).pos_m;
        z = ZONES(TARGETS(i).zone_id).z_level;
        
        if strcmp(TARGETS(i).type, 'Injured')
            col = col_person_inj;
            plot3(pos(1), pos(2), z, 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', 10);
            text(pos(1), pos(2), z+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold');
            
        elseif strcmp(TARGETS(i).type, 'Normal')
            col = col_person_norm;
            plot3(pos(1), pos(2), z, 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', 10);
            text(pos(1), pos(2), z+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold');
            
        elseif strcmp(TARGETS(i).type, 'Search') || strcmp(TARGETS(i).type, 'Sweep')
            col = [0.1 0.8 0.1]; % 绿色
            plot3(pos(1), pos(2), z, 'x', 'Color', col, 'MarkerSize', 12, 'LineWidth', 2.5);
            text(pos(1), pos(2), z+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold', 'FontSize', 8, 'Clipping', 'on');
        end
    end
    
    % 绘制 "真实" 位置 (参考)
    scatter3(0.5, 0.5, 0, 80, col_person_norm, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeColor', 'k');
    scatter3(16, 18, 0, 80, col_person_norm, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeColor', 'k');

    % --- 3. 绘制 3D 路径 ---
    if isempty(all_paths)
        max_resp_id = 1; 
    else
        all_ids = cellfun(@(x) x.resp_id, all_paths);
        % V10-Fix: (第 1122 行) N_RESPONDERS 现在可用
        max_resp_id = max(max(all_ids), N_RESPONDERS); 
    end
    colors = lines(max(1, max_resp_id));

    for i = 1:length(all_paths)
        path = all_paths{i};
        col = colors(path.resp_id, :);
        
        % 确定 Z 轴
        z_level = 0;
        if path.zone_id > 0 && path.zone_id <= length(ZONES)
             z_level = ZONES(path.zone_id).z_level;
        end
        if strcmp(path.target_name, 'Exit')
            start_node_name = path.segments{1}(1).name;
            z_level = BUILDING_GRAPH.Nodes.(start_node_name).pos(3);
        end
        
        current_target_zone_type = '';
        if path.zone_id > 0 && path.zone_id <= length(ZONES)
            current_target_zone_type = ZONES(path.zone_id).type;
        end


        for s = 1:length(path.segments)
            seg = path.segments{s};
            
            if isstruct(seg) % 高层图路径 (节点 struct)
                pts = vertcat(seg.pos); % [x, y, z]
                plot3(pts(:,1), pts(:,2), pts(:,3), 'o-', 'Color', col, 'LineWidth', 2.5, 'MarkerSize', 5, 'MarkerFaceColor', 'w');
            
            else % 低层A* / 周界 / Zigzag 路径 (矩阵)
                pts_m = seg; % 默认为米制 (Apt/Shop)
                if strcmp(current_target_zone_type, 'Warehouse')
                     pts_m = (seg - 0.5) * res; 
                end
                
                % pts_z = ones(size(pts_m, 1), 1) * z_level;
                % plot3(pts_m(:,1), pts_m(:,2), pts_z, '.-', 'Color', col, 'LineWidth', 2.0, 'MarkerSize', 3);
                    % --- 修复：动态判断每个点的 Z (根据 Zones.rect) ---
    pts_z = zeros(size(pts_m,1),1);
    for pi = 1:size(pts_m,1)
        x = pts_m(pi,1); 
        y = pts_m(pi,2);
        z_guess = 0;

        % 优先依照区域判断 Z
        for zz = 1:length(ZONES)
            r = ZONES(zz).rect;
            if x>=r(1) && x<=r(1)+r(3) && ...
               y>=r(2) && y<=r(2)+r(4)
                z_guess = ZONES(zz).z_level;
                break;
            end
        end

        pts_z(pi) = z_guess;
    end

    plot3(pts_m(:,1), pts_m(:,2), pts_z, '.-', ...
        'Color', col, 'LineWidth', 2.0, 'MarkerSize', 3);
            end
        end
        
        end_node = path.segments{end};
        end_pos_3d = [];
        if isstruct(end_node)
            end_pos_3d = end_node(end).pos; 
        else
            end_pos_2d = end_node(end,:); 
            if strcmp(current_target_zone_type, 'Warehouse')
                 end_pos_2d = (end_pos_2d - 0.5) * res;
            end
            end_pos_3d = [end_pos_2d, z_level];
        end
        
        text(end_pos_3d(1), end_pos_3d(2), end_pos_3d(3)+0.2, ...
            sprintf('R%d -> %s (%.0fs)', path.resp_id, path.target_name, path.total_time), ...
            'Color', 'k', 'BackgroundColor', [1 1 1 0.7], 'FontSize', 8, 'FontWeight', 'bold', 'Clipping', 'on');
    end

    xlim([-2, 40]); ylim([-2, 24]); zlim([0, 6]);
end

% ========================
% V11: 2D 路径平面图函数
% ========================
function plot_2d_floor_plans_v11(ZONES, BUILDING_GRAPH, WH_OBSTACLES, TARGETS, all_paths, res, N_RESPONDERS)
    % V11: 在 2D 平面图上绘制路径

    fprintf('正在打开 2D 路径平面图窗口...\n');
    
    % --- 创建图形窗口 ---
    figure('Color', 'w', 'Name', '建筑 2D 路径平面图 (一层 & 二层)', 'Position', [200, 200, 1400, 700]);

    % --- 1. 定义常量 ---
    col_wall = [0 0 0];       % 墙壁 (黑)
    col_door = [1 0 0];       % 门 (红)
    col_obs  = [0.6 0.6 0.6]; % 障碍物 (深灰)
    col_stair = [0.95 0.95 0.8]; % 楼梯 (淡黄)
    col_hall = [0.95 0.95 0.95]; % 楼道 (浅灰)
    
    col_person_inj = [1 0 0]; 
    col_person_norm = [0 0 1]; 

    % V11: 获取颜色
    if isempty(all_paths)
        max_resp_id = 1; 
    else
        all_ids = cellfun(@(x) x.resp_id, all_paths);
        max_resp_id = max(max(all_ids), N_RESPONDERS); 
    end
    colors = lines(max(1, max_resp_id));
    
    % --- 2. 子图 1: 一层平面图 (仓库 & 商铺) ---
    subplot(1, 2, 1);
    hold on; axis equal; box on;
    title('一层路径图 (Floor 1: Warehouse & Shops)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X (m)'); ylabel('Y (m)');
    
    % 绘制一层区域 (Warehouse, Shops, Stairs)
    for i = 1:length(ZONES)
        if ZONES(i).z_level == 0 % 只绘制 Z=0 (一层)
            r = ZONES(i).rect;
            col = 'w';
            if strcmp(ZONES(i).type, 'Stairs'), col = col_stair; end
            draw_rect_2d(r(1), r(2), r(3), r(4), col, 1.5);
            
            % 添加标签
            if ~strcmp(ZONES(i).type, 'Warehouse')
                text(r(1)+r(3)/2, r(2)+r(4)/2, ZONES(i).name, 'Horiz', 'center');
            end
        end
    end
    
    % 绘制仓库内部障碍物
    for i=1:length(WH_OBSTACLES)
        obs = WH_OBSTACLES{i};
        draw_rect_2d(obs(1), obs(2), obs(3), obs(4), col_obs, 1);
    end
    
    % 绘制门
    plot([0, 0], [7, 15], 'Color', col_door, 'LineWidth', 4);    % 左大门
    plot([13, 15], [22, 22], 'Color', col_door, 'LineWidth', 4); % 上小门
    plot([13, 15], [0, 0], 'Color', col_door, 'LineWidth', 4);   % 下小门
    plot([28, 28], [0.5, 2.5], 'Color', col_door, 'LineWidth', 4); % 楼梯门1
    plot([28, 28], [19.5, 21.5], 'Color', col_door, 'LineWidth', 4);% 楼梯门2
    
    X_offset = 28; shop_w = 9; shop_h = 16/3;
    plot([X_offset+9, X_offset+9], [0.5, 2.5], 'Color', col_door, 'LineWidth', 4); % 楼梯1 出口
    plot([X_offset+9, X_offset+9], [19.5, 21.5], 'Color', col_door, 'LineWidth', 4); % 楼梯2 出口
    
    for i = 1:3
        y_s = 3 + (i-1)*shop_h;
        d_cen = y_s + shop_h/2;
        plot([X_offset+shop_w, X_offset+shop_w], [d_cen-1.5, d_cen+1.5], 'Color', col_door, 'LineWidth', 4);
    end
    
    % --- V11: 绘制一层 目标点 ---
    for i = 1:length(TARGETS)
        z = ZONES(TARGETS(i).zone_id).z_level;
        if z > 0, continue; end % 跳过二层
        
        pos = TARGETS(i).pos_m;
        if strcmp(TARGETS(i).type, 'Injured')
            col = col_person_inj;
            plot(pos(1), pos(2), 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
            text(pos(1), pos(2)+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold');
        elseif strcmp(TARGETS(i).type, 'Normal')
            col = col_person_norm;
            plot(pos(1), pos(2), 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
            text(pos(1), pos(2)+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold');
        elseif strcmp(TARGETS(i).type, 'Search') || strcmp(TARGETS(i).type, 'Sweep')
            col = [0.1 0.8 0.1]; % 绿色
            plot(pos(1), pos(2), 'x', 'Color', col, 'MarkerSize', 10, 'LineWidth', 2);
            text(pos(1), pos(2)+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold', 'FontSize', 8);
        end
    end
    
    % 绘制 "真实" 位置 (参考)
    scatter(0.5, 0.5, 80, col_person_norm, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeColor', 'k');
    scatter(16, 18, 80, col_person_norm, 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeColor', 'k');

    % --- V11: 绘制一层 路径 ---
    for i = 1:length(all_paths)
        path = all_paths{i};
        col = colors(path.resp_id, :);
        
        % 确定 Z 轴
        z_level = 0;
        if path.zone_id > 0 && path.zone_id <= length(ZONES)
             z_level = ZONES(path.zone_id).z_level;
        end
        if strcmp(path.target_name, 'Exit')
            start_node_name = path.segments{1}(1).name;
            z_level = BUILDING_GRAPH.Nodes.(start_node_name).pos(3);
        end
        
        % 如果 Z > 0 (二层路径), 则跳过
        if z_level > 0, continue; end
        
        current_target_zone_type = '';
        if path.zone_id > 0 && path.zone_id <= length(ZONES)
            current_target_zone_type = ZONES(path.zone_id).type;
        end

        for s = 1:length(path.segments)
            seg = path.segments{s};
            if isstruct(seg) % 高层图路径 (节点 struct)
                pts = vertcat(seg.pos); % [x, y, z]
                plot(pts(:,1), pts(:,2), 'o-', 'Color', col, 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'w');
            else % 低层A* / 周界 / Zigzag 路径 (矩阵)
                pts_m = seg; % 默认为米制 (Shop)
                if strcmp(current_target_zone_type, 'Warehouse')
                     pts_m = (seg - 0.5) * res; % A* 或 Zigzag 路径
                end
                plot(pts_m(:,1), pts_m(:,2), '.-', 'Color', col, 'LineWidth', 1.5, 'MarkerSize', 3);
            end
        end
        
        % 绘制终点标签
        end_node = path.segments{end};
        end_pos_2d = [];
        if isstruct(end_node)
            end_pos_2d = end_node(end).pos(1:2); 
        else
            end_pos_2d = end_node(end,:); 
            if strcmp(current_target_zone_type, 'Warehouse')
                 end_pos_2d = (end_pos_2d - 0.5) * res;
            end
        end
        text(end_pos_2d(1), end_pos_2d(2)+0.2, ...
            sprintf('R%d->%s', path.resp_id, path.target_name), ...
            'Color', 'k', 'BackgroundColor', [1 1 1 0.7], 'FontSize', 7, 'FontWeight', 'bold', 'Clipping', 'on');
    end
    
    xlim([-2, 40]); ylim([-2, 24]);

    % --- 3. 子图 2: 二层平面图 (公寓) ---
    subplot(1, 2, 2);
    hold on; axis equal; box on;
    title('二层路径图 (Floor 2: Apartments)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X (m)'); ylabel('Y (m)');
    
    % 绘制二层区域 (Apts, Hallway)
    for i = 1:length(ZONES)
        if ZONES(i).z_level > 0 % 只绘制 Z > 0 (二层)
            r = ZONES(i).rect;
            col = 'w';
            if strcmp(ZONES(i).type, 'Hallway'), col = col_hall; end
            draw_rect_2d(r(1), r(2), r(3), r(4), col, 1.5);
            
            % 添加标签
            text_pos = [r(1)+r(3)/2, r(2)+r(4)/2];
            if strcmp(ZONES(i).type, 'Hallway')
                text(text_pos(1), text_pos(2), ZONES(i).name, 'Rotation', 90, 'Horiz', 'center');
            else
                text(text_pos(1), text_pos(2), ZONES(i).name, 'Horiz', 'center');
            end
        end
    end
    
    % 绘制二层楼梯口 (参考)
    draw_rect_2d(X_offset, 0, 9, 3, col_stair, 1);
    text(X_offset+4.5, 1.5, 'Stairs 1 (F2)', 'Horiz', 'center');
    draw_rect_2d(X_offset, 19, 9, 3, col_stair, 1);
    text(X_offset+4.5, 20.5, 'Stairs 2 (F2)', 'Horiz', 'center');
    
    % === [新增代码: 绘制楼道两端的门 (防止视觉穿墙)] ===
    % 下方门 (连接 Stairs 1): y=3, x=34.5-36.5
    plot([34.5, 36.5], [3, 3], 'Color', col_door, 'LineWidth', 4);
    
    % 上方门 (连接 Stairs 2): y=19, x=34.5-36.5
    plot([34.5, 36.5], [19, 19], 'Color', col_door, 'LineWidth', 4);
    % ===============================================

    % 绘制公寓门
    room_w = 6; room_h = 4;
    for i = 1:4
        y_a = 3 + (i-1)*room_h;
        d_cen = y_a + room_h/2;
        plot([X_offset+room_w, X_offset+room_w], [d_cen-0.5, d_cen+0.5], 'Color', col_door, 'LineWidth', 4);
    end
    
    % --- V11: 绘制二层 目标点 ---
    for i = 1:length(TARGETS)
        z = ZONES(TARGETS(i).zone_id).z_level;
        if z == 0, continue; end % 跳过一层
        
        pos = TARGETS(i).pos_m;
        if strcmp(TARGETS(i).type, 'Injured')
            col = col_person_inj;
            plot(pos(1), pos(2), 'o', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
            text(pos(1), pos(2)+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold');
        elseif strcmp(TARGETS(i).type, 'Search')
            col = [0.1 0.8 0.1]; % 绿色
            plot(pos(1), pos(2), 'x', 'Color', col, 'MarkerSize', 10, 'LineWidth', 2);
            text(pos(1), pos(2)+0.5, TARGETS(i).name, 'Color', col, 'FontWeight', 'bold', 'FontSize', 8);
        end
    end
    
    % --- V11: 绘制二层 路径 ---
    for i = 1:length(all_paths)
        path = all_paths{i};
        col = colors(path.resp_id, :);
        
        % 确定 Z 轴
        z_level = 0;
        if path.zone_id > 0 && path.zone_id <= length(ZONES)
             z_level = ZONES(path.zone_id).z_level;
        end
        if strcmp(path.target_name, 'Exit')
            start_node_name = path.segments{1}(1).name;
            z_level = BUILDING_GRAPH.Nodes.(start_node_name).pos(3);
        end
        
        % 如果 Z = 0 (一层路径), 则跳过
        if z_level == 0, continue; end
        
        current_target_zone_type = '';
        if path.zone_id > 0 && path.zone_id <= length(ZONES)
            current_target_zone_type = ZONES(path.zone_id).type;
        end

        for s = 1:length(path.segments)
            seg = path.segments{s};
            if isstruct(seg) % 高层图路径 (节点 struct)
                pts = vertcat(seg.pos); % [x, y, z]
                plot(pts(:,1), pts(:,2), 'o-', 'Color', col, 'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'w');
            else % 低层周界搜索路径 (矩阵)
                pts_m = seg; % 默认为米制 (Apt)
                plot(pts_m(:,1), pts_m(:,2), '.-', 'Color', col, 'LineWidth', 1.5, 'MarkerSize', 3);
            end
        end
        
        % 绘制终点标签
        end_node = path.segments{end};
        end_pos_2d = [];
        if isstruct(end_node)
            end_pos_2d = end_node(end).pos(1:2); 
        else
            end_pos_2d = end_node(end,:); 
        end
        text(end_pos_2d(1), end_pos_2d(2)+0.2, ...
            sprintf('R%d->%s', path.resp_id, path.target_name), ...
            'Color', 'k', 'BackgroundColor', [1 1 1 0.7], 'FontSize', 7, 'FontWeight', 'bold', 'Clipping', 'on');
    end

    xlim([26, 40]); ylim([-2, 24]);
    
end

% --- 辅助绘图函数 ---
function draw_rect_2d(x, y, w, h, col, lw)
    % Helper function to draw rectangles in 2D
    if ischar(col) && col == 'w'
        rectangle('Position', [x, y, w, h], 'EdgeColor', 'k', 'LineWidth', lw);
    else
        rectangle('Position', [x, y, w, h], 'FaceColor', col, 'EdgeColor', 'k', 'LineWidth', lw);
    end
end