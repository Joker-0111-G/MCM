function draw_complex_fixed_warehouse()
    % 创建图形窗口
    figure('Color', 'w', 'Name', 'Complex Fixed Warehouse Model', 'Position', [100, 100, 1200, 600]);

    %% === 参数定义 ===
    col_wall = [0 0 0];       % 墙壁 (黑)
    col_door = [1 0 0];       % 门 (红)
    col_obs  = [0.6 0.6 0.6]; % 障碍物 (深灰，表示实心物体)
    col_person = [0 0 1];     % 待救援人员 (蓝)
    col_stair = [0.95 0.95 0.8]; % 楼梯 (淡黄)
    
    W_wh = 28; H_wh = 22;     % 仓库尺寸
    X_offset = 28;            % 右侧建筑起始X
    
    % 右侧尺寸
    shop_w = 9; shop_h = 16/3; 
    room_w = 6; room_h = 4; hall_w = 3;

    %% =====================
    %  子图 1: 一层平面图 (复杂仓库)
    % =====================
    subplot(1, 2, 1);
    hold on; axis equal; box on;
    title('Floor 1: Complex Warehouse & Shops', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X (m)'); ylabel('Y (m)');
    
    % --- 1. 仓库外框与门 ---
    draw_rect(0, 0, W_wh, H_wh, 'w', 2); 
    
    % 门 (Doors)
    plot([0, 0], [7, 15], 'Color', col_door, 'LineWidth', 4);    % 左大门
    plot([13, 15], [22, 22], 'Color', col_door, 'LineWidth', 4); % 上小门
    plot([13, 15], [0, 0], 'Color', col_door, 'LineWidth', 4);   % 下小门
    % 右侧通往楼梯的门
    plot([28, 28], [0.5, 2.5], 'Color', col_door, 'LineWidth', 4);
    plot([28, 28], [19.5, 21.5], 'Color', col_door, 'LineWidth', 4);

    % --- 2. 仓库内部复杂固定布局 (Hardcoded Complexity) ---
    
    % [区域 A: 左上角 - 倒L型货架组]
    draw_rect(3, 16, 8, 2, col_obs, 1);  % 横条
    draw_rect(3, 12, 2, 4, col_obs, 1);  % 竖条连接
    draw_rect(6, 13, 2, 2, col_obs, 1);  % 散货堆
    
    % [区域 B: 左下角 - 迷宫入口]
    draw_rect(3, 4, 8, 3, col_obs, 1);   % 大底座
    draw_rect(3, 8, 3, 2, col_obs, 1);   % 阻挡块
    
    % [区域 C: 中间 - 岛式阵列 (迫使走S型路线)]
    % 第一列岛
    draw_rect(13, 14, 2, 4, col_obs, 1);
    draw_rect(13, 4, 2, 4, col_obs, 1);
    % 第二列岛
    draw_rect(17, 10, 2, 6, col_obs, 1); % 中间长条
    draw_rect(17, 17, 2, 2, col_obs, 1); % 上方点
    draw_rect(17, 2, 2, 2, col_obs, 1);  % 下方点
    
    % [区域 D: 右侧 - 屏风与隔离带 (保护死角)]
    % 这个长条把右侧隔离成了一条狭窄走廊，增加了到达救援点的距离
    draw_rect(22, 5, 2, 12, col_obs, 1); 
    
    % [区域 E: 门前阻挡物]
    % 在上小门(x=14,y=22)前面放一个障碍，进门必须绕行
    draw_rect(13, 19, 4, 1, col_obs, 1);
    % 在右侧楼梯门前放障碍
    draw_rect(25, 18, 1, 3, col_obs, 1); 

    % --- 3. 仓库待救援人员 (固定在最难到达的死角) ---
    
    % Person 1: 
    % 坐标: (0.5,0.5)
    scatter(0.5,0.5, 80, col_person, 'filled');
    
    % Person 2: 
    % 坐标: (16,18) 
    scatter(16,18, 80, col_person, 'filled');

    % --- 4. 右侧建筑 (楼梯 & 商铺) ---
    draw_rect(X_offset, 0, 9, 3, col_stair, 1); % 下楼梯
    plot([X_offset+9, X_offset+9], [0.5, 2.5], 'Color', col_door, 'LineWidth', 4);
    
    draw_rect(X_offset, 19, 9, 3, col_stair, 1); % 上楼梯
    plot([X_offset+9, X_offset+9], [19.5, 21.5], 'Color', col_door, 'LineWidth', 4);

    % 商铺 1, 2, 3
    for i = 1:3
        y_s = 3 + (i-1)*shop_h;
        draw_rect(X_offset, y_s, shop_w, shop_h, 'w', 2);
        % 门
        d_cen = y_s + shop_h/2;
        plot([X_offset+shop_w, X_offset+shop_w], [d_cen-1.5, d_cen+1.5], 'Color', col_door, 'LineWidth', 4);
        text(X_offset+shop_w/2, d_cen, sprintf('Shop %d', i), 'Horiz', 'center');
    end
    
    % Shop 2 待救援人员 (内部)
    scatter(X_offset + 2, 3 + shop_h + 3, 80, col_person, 'filled');

    xlim([-2, 40]); ylim([-2, 24]);

    %% =====================
    %  子图 2: 二层平面图 (住房)
    % =====================
    subplot(1, 2, 2);
    hold on; axis equal; box on;
    title('Floor 2: Apartments', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X (m)'); ylabel('Y (m)');
    
    % 楼梯参考
    draw_rect(X_offset, 0, 9, 3, col_stair, 1);
    draw_rect(X_offset, 19, 9, 3, col_stair, 1);
    
    % 楼道
    draw_rect(X_offset + room_w, 3, hall_w, 16, [0.95 0.95 0.95], 1);
    text(X_offset + room_w + hall_w/2, 11, 'Hallway', 'Rotation', 90, 'Horiz', 'center');
    
    % === [新增代码: 楼道两端的门] ===
    % 下端门 (连接下楼梯): y=3, x=34.5-36.5
    plot([34.5, 36.5], [3, 3], 'Color', col_door, 'LineWidth', 4);
    
    % 上端门 (连接上楼梯): y=19, x=34.5-36.5
    plot([34.5, 36.5], [19, 19], 'Color', col_door, 'LineWidth', 4);

    % 住房 1-4
    for i = 1:4
        y_a = 3 + (i-1)*room_h;
        draw_rect(X_offset, y_a, room_w, room_h, 'w', 2);
        % 门
        d_cen = y_a + room_h/2;
        plot([X_offset+room_w, X_offset+room_w], [d_cen-0.5, d_cen+0.5], 'Color', col_door, 'LineWidth', 4);
        text(X_offset+room_w/2, d_cen, sprintf('Apt %d', i), 'Horiz', 'center');
    end
    
    % Apt 1 待救援人员 (最内部左下角)
    scatter(X_offset + 1, 3 + 1, 80, col_person, 'filled');
    
    % Apt 4 待救援人员 (最内部左上角)
    scatter(X_offset + 1, 3 + 3*room_h + 3, 80, col_person, 'filled');
    
    xlim([20, 45]); ylim([-2, 24]);
    
    %% === 图例 ===
    h_obs = patch(nan, nan, col_obs);
    h_person = scatter(nan, nan, 80, 'b', 'filled');
    h_door = plot(nan, nan, 'r-', 'LineWidth', 4);
    
    legend([h_obs, h_person, h_door], ...
           {'Fixed Obstacles (复杂固定障碍)', 'Rescuee (角落待救援)', 'Door (门)'}, ...
           'Location', 'southoutside', 'Orientation', 'horizontal');
end

function draw_rect(x, y, w, h, col, lw)
    if ischar(col) && col == 'w'
        rectangle('Position', [x, y, w, h], 'EdgeColor', 'k', 'LineWidth', lw);
    else
        rectangle('Position', [x, y, w, h], 'FaceColor', col, 'EdgeColor', 'k', 'LineWidth', lw);
    end
end