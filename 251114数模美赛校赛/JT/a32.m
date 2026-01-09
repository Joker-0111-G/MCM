% 救援过程 拆解出的最需要的部分


function evacuation_complex_final()
    % =========================================================================
    % COMPLEX WAREHOUSE & RESIDENTIAL RESCUE SIMULATION (FINAL)
    % =========================================================================
    % Features:
    % 1. Exact replication of 'a31.m' geometry and obstacles.
    % 2. Specific Resident Profiles:
    %    - Apt 4: Infant Family (Priority 1)
    %    - Apt 1: Pregnant/Injured (Priority 1, Medical Team)
    % 3. Physics:
    %    - Responder: Str 1.2, Turn 1.0, Stair 1.1
    %    - Rescuee:   Str 1.0, Turn 0.8, Stair 0.9
    % 4. Personnel Roles: Security, Medical, Police, Firefighters.
    % =========================================================================
    
    clc; clear; close all;
    
    % --- Global Constraints ---
    SMOKE_LIMIT_MIN = 30; 
    SMOKE_LIMIT_SEC = SMOKE_LIMIT_MIN * 60;
    
    % --- Speed Parameters (m/s) ---
    % [Straight, Turn, Stair]
    SPEEDS.Responder = [1.2, 1.0, 1.1];
    SPEEDS.Rescuee   = [1.0, 0.8, 0.9];
    
    % --- Define Missions (Targets & Profiles) ---
    % Targets coordinates must match the logic of obstacles in a31
    
    % Mission 1: Apt 4 (Infant) - Priority High - Team: Fire/Security
    M(1).name = 'Apt 4 (Infant)';
    M(1).loc = [29, 18, 3]; % x,y,z (F2 is z=3)
    M(1).prio = 1;
    M(1).team = 'Fire/Security';
    M(1).type = 'Normal'; 
    
    % Mission 2: Apt 1 (Pregnant) - Priority High - Team: Medical
    M(2).name = 'Apt 1 (Pregnant)';
    M(2).loc = [29, 4, 3];
    M(2).prio = 1;
    M(2).team = 'Medical';
    M(2).type = 'Injured'; % Handled by Medical
    
    % Mission 3: Shop 2 - Team: Security
    M(3).name = 'Shop 2';
    M(3).loc = [30, 11.3, 0];
    M(3).prio = 2;
    M(3).team = 'Security';
    M(3).type = 'Normal';
    
    % Mission 4: Warehouse Deep Corner - Team: Firefighter 1
    M(4).name = 'WH Corner';
    M(4).loc = [0.5, 0.5, 0];
    M(4).prio = 2;
    M(4).team = 'Firefighter A';
    M(4).type = 'Normal';
    
    % Mission 5: Warehouse Maze - Team: Firefighter 2
    M(5).name = 'WH Maze';
    M(5).loc = [16, 18, 0];
    M(5).prio = 2;
    M(5).team = 'Firefighter B';
    M(5).type = 'Normal';

    % --- Calculate Paths & Times ---
    fprintf('=== 救援任务模拟报告 (烟雾极限: %d 分钟) ===\n', SMOKE_LIMIT_MIN);
    fprintf('%-20s | %-15s | %-10s | %-10s | %-10s\n', 'Target', 'Team', 'Priority', 'Status', 'Total Time');
    fprintf('-------------------------------------------------------------------------------\n');
    
    results = [];
    
    for i = 1:length(M)
        % Generate Route Nodes based on Obstacle Logic (No Wall Clip)
        route_nodes = generate_safe_path(i, M(i).loc);
        
        % Calculate Physics
        [t_in, t_out, dist_log] = calc_mission_physics(route_nodes, SPEEDS);
        
        total_time = t_in + t_out;
        status = 'SUCCESS';
        if total_time > SMOKE_LIMIT_SEC, status = 'FAILED (Smoke)'; end
        
        % Log Results
        res.mission = M(i);
        res.path = route_nodes;
        res.time = total_time;
        res.t_in = t_in;
        res.t_out = t_out;
        results = [results, res];
        
        fprintf('%-20s | %-15s | Level %-4d | %-10s | %.2f s\n', ...
            M(i).name, M(i).team, M(i).prio, status, total_time);
    end
    fprintf('-------------------------------------------------------------------------------\n');
    
    % ================= Visualization =================
    
    % 1. Prepare Figure
    fig = figure('Color', 'w', 'Name', 'Advanced Rescue Simulation', 'Position', [50, 50, 1400, 800]);
    
    % 2. Draw 2D Plans (Subplot 1 & 2)
    % We split 2D into two subplots for clarity: F2 (Top) and F1 (Bottom)
    
    % --- Floor 2 (Apartments) ---
    subplot(2, 2, 1); 
    draw_floor_plan(2); % Draw Geometry
    hold on;
    title('Floor 2: Apartments (High Priority Targets)');
    plot_paths_2d(results, 2); % Plot paths on F2
    
    % --- Floor 1 (Warehouse/Shops) ---
    subplot(2, 2, 3);
    draw_floor_plan(1); % Draw Geometry
    hold on;
    title('Floor 1: Warehouse & Shops (Complex Obstacles)');
    plot_paths_2d(results, 1); % Plot paths on F1
    
    % --- 3D View ---
    subplot(2, 2, [2, 4]);
    draw_building_3d();
    hold on;
    plot_paths_3d(results);
    view(45, 30);
    title('3D Rescue Operation View');
    grid on; axis equal;
    
end

% =========================================================================
% CORE LOGIC: PATH GENERATION (Avoids Known Obstacles)
% =========================================================================
function nodes = generate_safe_path(mission_id, target_loc)
    % Hardcoded Navigation Graph based on a31.m geometry
    % 关键修改：强制通过楼道两端的门 (x=35.5, y=3 和 y=19)
    
    nodes = [];
    
    switch mission_id
        case 1 % Apt 4 (Infant) - F2
            % Path: Right Ext Door -> Top Stair -> Top Door -> Hallway -> Apt 4
            
            % 1. 进大门 (地面)
            start_pt = [37, 20.5, 0]; 
            
            % 2. 走到楼梯底部中心 (x调整为35.5以对齐楼道)
            stair_btm = [35.5, 20.5, 0];
            
            % 3. 上楼 (到达二层楼梯间)
            stair_top = [35.5, 20.5, 3];
            
            % 4. === 关键点: 穿过上方的门 (y=19) ===
            door_node = [35.5, 19, 3];
            
            % 5. 进入楼道 (y=17)
            hall_pt   = [35.5, 17, 3];
            
            % 6. 转向进入房间门口
            apt_entry = [34, 17, 3];
            
            target    = target_loc;
            nodes = [start_pt; stair_btm; stair_top; door_node; hall_pt; apt_entry; target];
            
        case 2 % Apt 1 (Pregnant) - F2
            % Path: Right Ext Door -> Bottom Stair -> Bottom Door -> Hallway -> Apt 1
            
            % 1. 进大门 (地面)
            start_pt = [37, 1.5, 0]; 
            
            % 2. 走到楼梯底部中心 (x调整为35.5)
            stair_btm = [35.5, 1.5, 0];
            
            % 3. 上楼 (到达二层楼梯间)
            stair_top = [35.5, 1.5, 3];
            
            % 4. === 关键点: 穿过下方的门 (y=3) ===
            door_node = [35.5, 3, 3];
            
            % 5. 进入楼道 (y=5)
            hall_pt   = [35.5, 5, 3];
            
            % 6. 转向进入房间门口
            apt_entry = [34, 5, 3];
            
            target    = target_loc;
            nodes = [start_pt; stair_btm; stair_top; door_node; hall_pt; apt_entry; target];
            
        case 3 % Shop 2
            % Direct entry from street
            start_pt = [37, 11.3, 0];
            target = target_loc;
            nodes = [start_pt; target];
            
        case 4 % WH Corner (0.5, 0.5)
            % Blocked by Maze Entry at (3,4). Must hug left wall.
            start_pt = [0, 11, 0]; % Enter Left Main Door
            safe_pt1 = [1.5, 11, 0]; % Step in
            safe_pt2 = [1.5, 1, 0];  % Go down (avoiding obs at x=3)
            target = target_loc;
            nodes = [start_pt; safe_pt1; safe_pt2; target];
            
        case 5 % WH Maze (16, 18)
            % Blocked by Door Blocker at (13,19).
            % Enter Upper Door (14,22).
            start_pt = [14, 22, 0];
            step_in  = [14, 20.5, 0]; 
            % Must go around blocker (x:13-17, y:19-20)
            avoid_l  = [12, 20.5, 0]; % Go left into corridor
            avoid_d  = [12, 18, 0];   % Go down
            target   = target_loc;
            nodes = [start_pt; step_in; avoid_l; avoid_d; target];
    end
end

% =========================================================================
% CORE LOGIC: PHYSICS & TIMING
% =========================================================================
function [t_in, t_out, log] = calc_mission_physics(path, speeds)
    % path: Nx3 matrix [x, y, z]
    % speeds: struct with .Responder and .Rescuee [Str, Turn, Stair]
    
    % 1. Calculate INBOUND time (Responder only)
    t_in = traverse_path(path, speeds.Responder);
    
    % 2. Calculate OUTBOUND time (Responder + Victim)
    % Victim is slower. We assume the speed is limited by the victim.
    % Path is reversed.
    rev_path = flipud(path);
    t_out = traverse_path(rev_path, speeds.Rescuee);
    
    log = sprintf('In: %.1fs, Out: %.1fs', t_in, t_out);
end

function t = traverse_path(pts, speed_prof)
    % Speed Profile: 1=Str, 2=Turn, 3=Stair
    v_str = speed_prof(1);
    v_turn = speed_prof(2);
    v_stair = speed_prof(3);
    
    t = 0;
    if size(pts, 1) < 2, return; end
    
    % Iterate segments
    for i = 1:size(pts, 1)-1
        p1 = pts(i,:);
        p2 = pts(i+1,:);
        
        dist = norm(p2 - p1);
        dz = abs(p2(3) - p1(3));
        
        % Determine type
        if dz > 0.1
            % Stair case
            t = t + dist / v_stair;
        else
            % Flat
            t = t + dist / v_str;
        end
        
        % Turn Penalty (logic: if direction changes from prev segment)
        if i > 1
            p0 = pts(i-1,:);
            vec1 = (p1 - p0); vec1 = vec1 / (norm(vec1)+eps);
            vec2 = (p2 - p1); vec2 = vec2 / (norm(vec2)+eps);
            
            % Dot product < 0.99 means a turn happened
            if dot(vec1, vec2) < 0.99
                % Add 'Turn Penalty'. 
                % Implementation: Assume turning takes 1s equivalent delay or 
                % traverse the corner arc at v_turn.
                % Simplified: Add 1.5s penalty for slowing down/reorienting
                t = t + 1.5 * (v_str / v_turn); 
            end
        end
    end
end

% =========================================================================
% VISUALIZATION FUNCTIONS (Based on a31.m)
% =========================================================================

function draw_floor_plan(floor_idx)
    % Replicates a31.m drawing logic
    
    col_wall = [0 0 0]; col_door = [1 0 0]; col_obs = [0.6 0.6 0.6]; 
    col_stair = [0.95 0.95 0.8];
    
    hold on; axis equal; box on;
    xlabel('X (m)'); ylabel('Y (m)');
    
    X_offset = 28; 
    W_wh = 28; H_wh = 22;
    
    if floor_idx == 1
        % --- F1: Warehouse & Shops ---
        draw_rect(0, 0, W_wh, H_wh, 'w', 2); % Warehouse Outline
        
        % Warehouse Internal Obstacles (Exact a31 copies)
        draw_rect(3, 16, 8, 2, col_obs, 1);  % Area A
        draw_rect(3, 12, 2, 4, col_obs, 1);
        draw_rect(6, 13, 2, 2, col_obs, 1);
        
        draw_rect(3, 4, 8, 3, col_obs, 1);   % Area B
        draw_rect(3, 8, 3, 2, col_obs, 1);
        
        draw_rect(13, 14, 2, 4, col_obs, 1); % Area C Islands
        draw_rect(13, 4, 2, 4, col_obs, 1);
        draw_rect(17, 10, 2, 6, col_obs, 1);
        draw_rect(17, 17, 2, 2, col_obs, 1);
        draw_rect(17, 2, 2, 2, col_obs, 1);
        
        draw_rect(22, 5, 2, 12, col_obs, 1); % Area D
        
        draw_rect(13, 19, 4, 1, col_obs, 1); % Door Blockers
        draw_rect(25, 18, 1, 3, col_obs, 1);
        
        % Warehouse Doors
        plot([0, 0], [7, 15], 'Color', col_door, 'LineWidth', 3);
        plot([13, 15], [22, 22], 'Color', col_door, 'LineWidth', 3);
        plot([28, 28], [0.5, 2.5], 'Color', col_door, 'LineWidth', 3);
        plot([28, 28], [19.5, 21.5], 'Color', col_door, 'LineWidth', 3);
        
        % Right Building F1 (Shops & Stairs)
        draw_rect(X_offset, 0, 9, 3, col_stair, 1); % Stair Down
        draw_rect(X_offset, 19, 9, 3, col_stair, 1); % Stair Up
        
        shop_h = 16/3; shop_w = 9;
        for i=1:3
            y_s = 3 + (i-1)*shop_h;
            draw_rect(X_offset, y_s, shop_w, shop_h, 'w', 1);
            plot([X_offset+shop_w, X_offset+shop_w], [y_s+shop_h/2-1, y_s+shop_h/2+1], 'r-', 'LineWidth',3);
            text(X_offset+4.5, y_s+shop_h/2, sprintf('Shop %d', i), 'Horiz', 'center');
        end
        
        xlim([-2, 40]); ylim([-2, 24]);
        
    else
        % --- F2: Apartments ---
        % Use Phantom WH for context
        rectangle('Position', [0,0,28,22], 'EdgeColor', [0.8 0.8 0.8], 'LineStyle', '--');
        
        draw_rect(X_offset, 0, 9, 3, col_stair, 1);
        draw_rect(X_offset, 19, 9, 3, col_stair, 1);
        
        hall_w = 3; room_w = 6; room_h = 4;
        draw_rect(X_offset + room_w, 3, hall_w, 16, [0.95 0.95 0.95], 1); % Hallway
        
        for i=1:4
            y_a = 3 + (i-1)*room_h;
            draw_rect(X_offset, y_a, room_w, room_h, 'w', 2);
            plot([X_offset+room_w, X_offset+room_w], [y_a+room_h/2-0.5, y_a+room_h/2+0.5], 'r-', 'LineWidth',3);
            
            label = sprintf('Apt %d', i);
            if i==1, label = 'Apt 1 (Pregnant)'; end
            if i==4, label = 'Apt 4 (Infant)'; end
            text(X_offset+3, y_a+2, label, 'Horiz', 'center', 'FontSize', 8);
        end
        xlim([20, 45]); ylim([-2, 24]);
    end
end

function plot_paths_2d(results, floor_idx)
    colors = lines(length(results));
    for k = 1:length(results)
        p = results(k).path;
        z_mask = abs(p(:,3) - (floor_idx==2)*3) < 0.5; % Simple Z filtering
        
        % Exception for Stairs: Show connecting lines partially
        if any(z_mask)
            plot(p(z_mask,1), p(z_mask,2), '.-', 'Color', colors(k,:), 'LineWidth', 2);
            % Plot End Point
            if z_mask(end)
                plot(p(end,1), p(end,2), 'p', 'MarkerSize', 12, 'MarkerFaceColor', colors(k,:), 'MarkerEdgeColor','k');
            end
        end
    end
end

function draw_building_3d()
    % Extrude 2D plan into 3D
    hold on;
    % F1 Floor
    patch([0 28 28 0], [0 0 22 22], [0 0 0 0], [0.9 0.9 0.9], 'FaceAlpha', 0.3);
    % F2 Floor (Right Bldg)
    patch([28 37 37 28], [0 0 22 22], [3 3 3 3], [0.8 0.8 0.8], 'FaceAlpha', 0.5);
    
    % Draw Obstacles as Cubes
    create_block(3, 4, 8, 3, 2, [0.6 0.6 0.6]); % Area B
    create_block(22, 5, 2, 12, 2, [0.6 0.6 0.6]); % Area D
    % ... (simplified set for 3D clarity)
end

function create_block(x, y, w, h, z_h, col)
    % Helper to draw 3D cube
    vert = [x y 0; x+w y 0; x+w y+h 0; x y+h 0; ...
            x y z_h; x+w y z_h; x+w y+h z_h; x y+h z_h];
    faces = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 5 6 7 8; 1 2 3 4];
    patch('Vertices', vert, 'Faces', faces, 'FaceColor', col, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end

function plot_paths_3d(results)
    colors = lines(length(results));
    for k = 1:length(results)
        p = results(k).path;
        plot3(p(:,1), p(:,2), p(:,3), '.-', 'Color', colors(k,:), 'LineWidth', 2);
        text(p(end,1), p(end,2), p(end,3)+0.5, results(k).mission.name, 'Color', colors(k,:), 'FontSize', 8);
    end
end

function draw_rect(x, y, w, h, col, lw)
    if ischar(col) && col == 'w'
        rectangle('Position', [x, y, w, h], 'EdgeColor', 'k', 'LineWidth', lw);
    else
        rectangle('Position', [x, y, w, h], 'FaceColor', col, 'EdgeColor', 'k', 'LineWidth', lw);
    end
end
