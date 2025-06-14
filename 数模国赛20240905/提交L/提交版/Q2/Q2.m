clc; clear; close all;  
  
% 定义常量  
Lh = 3.41;    % 龙头长度（米）  
Lb = 2.2;     % 龙身长度（米）  
Leff = Lb - 2*0.275; % 有效长度（米）  
Lext = 0.275; % 外延长度（米）  
Sp = 0.55;    % 螺距（米）  
a = Sp / (2*pi); % 螺线参数  
Vh = 1;        % 龙头速度（米/秒）  
T = 500;       % 模拟总时间（秒）  
dt = 1;        % 时间步长（秒）  
N = 223;       % 板凳总数  
tol = 1;       % 碰撞检测容差  
theta_s = 16*2*pi; % 起始角度（第16圈）  
  
% 初始化数组  
t = 0:dt:T;  
th = zeros(N, length(t));  
x = zeros(N, length(t));  
y = zeros(N, length(t));  
vx = zeros(N, length(t));  
vy = zeros(N, length(t));  
  
% 初始化角度和位置  
th(1) = theta_s;  
x(1) = a * th(1) * cos(th(1));  
y(1) = a * th(1) * sin(th(1));  
  
% 计算后续板凳的位置  
for i = 2:N  
    if i == 2  
        ds = Lh;  
    else  
        ds = Leff;  
    end  
    r_prev = a * th(i-1);  
    r_next = r_prev + ds / (2*pi*(i-1 + theta_s/(2*pi)));  
    th_next = atan2(r_next, a);  
    delta_th = ds / (r_prev + ds/2);  
    th(i) = th(i-1) + delta_th;  
    x(i) = (r_prev + ds/2) * cos(th(i));  
    y(i) = (r_prev + ds/2) * sin(th(i));  
end  
  
% 主循环  
col_time = -1;  
for j = 2:length(t)  
    delta_s = Vh * dt;  
    delta_th = delta_s ./ (a * th(:, j-1));  
    th(:, j) = th(:, j-1) - delta_th;  
    x(:, j) = a * th(:, j) .* cos(th(:, j));  
    y(:, j) = a * th(:, j) .* sin(th(:, j));  
    vx(:, j) = (x(:, j) - x(:, j-1)) / dt;  
    vy(:, j) = (y(:, j) - y(:, j-1)) / dt;  
  
    % 碰撞检测  
    seg_start = [x(:, j) - Lext * cos(th(:, j)), y(:, j) - Lext * sin(th(:, j))];  
    seg_end = [x(:, j) + (Lb + Lext) * cos(th(:, j)), y(:, j) + (Lb + Lext) * sin(th(:, j))];  
    [intersect, ~] = lineSegmentIntersect(reshape([seg_start, seg_end]', 4, [])');  
    intersect = reshape(intersect, N, N);  
    intersect = triu(intersect, 1);  
    if any(intersect(:))  
        col_time = t(j);  
        break;  
    end  
end  
  
% 保存和显示结果  
j_final = length(t);  
result = zeros(N, 3);  
result(:,1) = x(:, j_final);  
result(:,2) = y(:, j_final);  
result(:,3) = sqrt(vx(end, j_final)^2 + vy(end, j_final)^2);  
writematrix(round(result, 6), 'result2.xlsx');  
  
% 特定板凳的位置和速度  
selected = [1, 2, 52, 102, 152, 202, 223];  
pos_table = zeros(2*length(selected), 1);  
vel_table = zeros(length(selected), 1);  
for i = 1:length(selected)  
    idx = selected(i);  
    pos_table(2*i-1) = x(idx, j_final);  
    pos_table(2*i) = y(idx, j_final);  
    vel_table(i) = sqrt(vx(idx, j_final)^2 + vy(idx, j_final)^2);  
end  
  
% 打印和保存表格  
disp('位置：');  
disp(round(pos_table, 6));  
disp('速度：');  
disp(round(vel_table, 6));  
disp(['碰撞时刻：', num2str(col_time), ' 秒']);  
  
% 绘制碰撞时刻的板凳龙形状  
figure;  
plot(x(:, j_final), y(:, j_final), 'b-', 'LineWidth', 2);  
hold on;  
plot(x(1, j_final), y(1, j_final), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'black');  
plot(x(end, j_final), y(end, j_final), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'r');  
title('碰撞时刻的板凳龙形状');  
xlabel('X 坐标 (米)');  
ylabel('Y 坐标 (米)');  
axis equal;  
grid on;  
  
% 线段相交检测函数  
function [I,P] = lineSegmentIntersect(XY)  
    X1 = XY(:,1); Y1 = XY(:,2);  
    X2 = XY(:,3); Y2 = XY(:,4);  
    [X1, X3] = meshgrid(X1, X1);  
    [Y1, Y3] = meshgrid(Y1, Y1);  
    [X2, X4] = meshgrid(X2, X2);  
    [Y2, Y4] = meshgrid(Y2, Y2);  
    D = (X1-X2).*(Y3-Y4) - (Y1-Y2).*(X3-X4);  
    ua = ((X1-X3).*(Y3-Y4) - (Y1-Y3).*(X3-X4)) ./ D;  
    ub = ((X1-X3).*(Y1-Y2) - (Y1-Y3).*(X1-X2)) ./ D;  
    I = (ua >= 0 & ua <= 1 & ub >= 0 & ub <= 1);  
    P = [X1 + ua.*(X2-X1), Y1 + ua.*(Y2-Y1)];  
end