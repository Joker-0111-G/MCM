clear

% 定义常量
Lh = 3.41; % 龙头长度（米）
Lb = 2.20; % 龙身长度（米）
Le = Lb - 2 * 0.275; % 有效长度（米）
p = 0.55; % 螺距（米）
a = p / (2 * pi); % 螺线参数
vh = 1; % 龙头速度（米/秒）
T = 300; % 总时间（秒）
dt = 1; % 时间步长（秒）
n = 223; % 板凳总数

% 初始化数组
t = 0:dt:T;
th = zeros(n, length(t));
x = zeros(n, length(t));
y = zeros(n, length(t));
vx = zeros(n, length(t));
vy = zeros(n, length(t));

% 初始化位置
th(1, 1) = 16 * 2 * pi; % 龙头初始位置在第16圈
for i = 2:n
    if i == 2
        ds = Lh;
    else
        ds = Le;
    end
    dth = ds / (a * th(i-1, 1));
    th(i, 1) = th(i-1, 1) + dth;
end

% 计算初始位置的x和y坐标
for i = 1:n
    x(i, 1) = a * th(i, 1) * cos(th(i, 1));
    y(i, 1) = a * th(i, 1) * sin(th(i, 1));
end

% 主循环
for j = 2:length(t)
    % 计算龙头新的theta（顺时针盘入，theta减小）
    ds = vh * dt;
    dth = ds / (a * th(1, j-1));
    th(1, j) = th(1, j-1) - dth;

    % 计算龙头位置
    x(1, j) = a * th(1, j) * cos(th(1, j));
    y(1, j) = a * th(1, j) * sin(th(1, j));

    % 计算龙头速度
    vx(1, j) = -vh * sin(th(1, j));
    vy(1, j) = vh * cos(th(1, j));

    % 计算龙身和龙尾位置和速度
    for i = 2:n
        if i == 2
            ds = Lh;
        else
            ds = Le;
        end
        dth = ds / (a * th(i-1, j));
        th(i, j) = th(i-1, j) + dth;
        
        x(i, j) = a * th(i, j) * cos(th(i, j));
        y(i, j) = a * th(i, j) * sin(th(i, j));
        
        % 计算速度
        vx(i, j) = (x(i, j) - x(i, j-1)) / dt;
        vy(i, j) = (y(i, j) - y(i, j-1)) / dt;
    end
end

% 保存到result1.1.xlsx
result1 = zeros(2*n, length(t)); % 修改为2*n行，存储位置和速度

for i = 1:n
    result1(2*i-1, :) = x(i, :); % 存储x坐标
    result1(2*i, :) = y(i, :);   % 存储y坐标
end

% 添加龙身速度到result1
velocity = zeros(n, length(t)); % 初始化速度矩阵
for j = 1:length(t)
    for i = 2:n
        velocity(i, j) = sqrt(vx(i, j)^2 + vy(i, j)^2);
    end
end

% 将速度添加到result1矩阵的后面
result1 = [result1; velocity];

writematrix(round(result1', 6), 'result1.1.xlsx');

% 保存论文中需要的表格数据
selT = [0, 60, 120, 180, 240, 300];
selS = [1,  52, 102, 152, 202, 223];

posTable = zeros(14, 6);
velTable = zeros(7, 6);

for i = 1:length(selT)
    tIdx = selT(i) + 1;
    for j = 1:length(selS)
        sIdx = selS(j);
        posTable(2*j-1, i) = x(sIdx, tIdx);
        posTable(2*j, i) = y(sIdx, tIdx);
        velTable(j, i) = sqrt(vx(sIdx, tIdx)^2 + vy(sIdx, tIdx)^2);
    end
end

% 保存位置表格
posTableRounded = round(posTable, 6);
writematrix(posTableRounded, 'Position.xlsx');

% 保存速度表格
velTableRounded = round(velTable, 6);
writematrix(velTableRounded, 'Speed.xlsx');

% 论文展现的位置以及速度
disp('Position:');
disp(posTableRounded);
disp('Speed:');
disp(velTableRounded);

% 绘制运动轨迹图
figure('Position', [100, 100, 800, 600]);
plot(x(1,:), y(1,:), 'r-', 'LineWidth', 2);
hold on;
plot(x(end,:), y(end,:), 'b-', 'LineWidth', 2);
plot(x(:,end), y(:,end), 'g-', 'LineWidth', 2);
title('Trajectory');
xlabel('X (m)');
ylabel('Y (m)');
legend('Head Trajectory', 'Tail Trajectory', 'Final Position');
grid on;
axis equal;
saveas(gcf, 'trajectory.png');
saveas(gcf, 'trajectory.fig');

% 绘制特定时间点的速度向量图

selT = [0, 60, 120, 180, 240, 300];
selS = [1, 2, 52, 102, 152, 202, 223]; % 龙头、龙头后第1、51、101、151、201节和龙尾

for k = 1:length(selT)

    timeIdx = selT(k) + 1; % MATLAB索引从1开始

    % 提取当前时间点的位置和速度  
    xCur = x(:, timeIdx);  
    yCur = y(:, timeIdx);  
    vxCur = vx(:, timeIdx);  
    vyCur = vy(:, timeIdx);  
    
    % 筛选特定段的位置和速度  
    xSel = xCur(selS);  
    ySel = yCur(selS);  
    vxSel = vxCur(selS);  
    vySel = vyCur(selS);  
    
    % 绘制速度向量图  
    figure('Position', [100 + 200*(k-1), 100, 800, 600]); % 偏移窗口位置以便查看所有图形  
    quiver(xSel, ySel, vxSel, vySel, 0.5, 'b');  
    title(sprintf('速度向量图：时间 %d 秒', selT(k)));  
    xlabel('X 坐标 (米)');  
    ylabel('Y 坐标 (米)');  
    axis equal;  
    grid on;  
    
    % 保存图形  
    saveas(gcf, sprintf('SpeedVectors_%d秒.png', selT(k)));  
end
