clc; clear; close all;  
  
sp = 1.7; % 螺距（米）  
sc = sp / (2*pi); % 螺线系数  
hl = 341e-2; % 龙头长度  
hhd = hl - 27.5e-2*2; % 龙头把手孔距  
bl = 220e-2; % 龙身和龙尾长度  
bhd = bl - 27.5e-2*2; % 凳子把手孔距  
hv = 1; % 龙头速度  
  
th = 5*2*pi:-0.01:0; % 角度范围  
r = sc * th; % 螺线半径  
xin = r .* cos(th); % 盘入线x坐标  
yin = r .* sin(th); % 盘入线y坐标  
  
figure('Name', '盘入盘出螺线及调头');  
plot(xin, yin, 'r-', 'LineWidth', 0.01); % 盘入螺线  
hold on;  
  
th_out = th - pi; % 盘出螺线角度  
rout = sc * (th_out + pi); % 盘出螺线半径  
xout = rout .* cos(th_out); % 盘出线x坐标  
yout = rout .* sin(th_out); % 盘出线y坐标  
plot(xout, yout, 'b-', 'LineWidth', 0.01); % 盘出螺线  
  
% 绘制调头区域  
tr = 4.5; % 调头区半径  
th_c = linspace(0, 2*pi, 100);  
xc = tr * cos(th_c); % 圆x坐标  
yc = tr * sin(th_c); % 圆y坐标  
plot(xc, yc, 'k--', 'LineWidth', 2); % 绘制调头区域  
  
xlabel('x (m)');  
ylabel('y (m)');  
title('盘入盘出螺线及调头');  
legend('盘入螺线', '盘出螺线', '边界');  
axis equal;  
grid on;  
  
th_in_end = tr / sc; % 盘入螺线终点角度  
th_out_start = th_in_end - pi; % 盘出螺线起点角度  
  
% 计算盘入螺线终点斜率  
slope_end = (sc*sin(th_in_end) + tr*cos(th_in_end)) / ...  
            (sc*cos(th_in_end) - tr*sin(th_in_end));  
  
th_max1 = atan(-1/slope_end) + pi; % 第一段圆弧  
th_delta = atan(tan(th_in_end)) + pi - th_max1; % 角度差  
  
rc1_c2 = tr / cos(th_delta); % 圆心距  
rc2 = rc1_c2 / 3; % 第二段圆弧半径  
rc1 = rc2 * 2; % 第一段圆弧半径  
  
phi = 2 * th_delta; % 圆弧角度  
arc_len_c1 = rc1 * (pi - phi); % 第一段圆弧长度  
arc_len_c2 = rc2 * (pi - phi); % 第二段圆弧长度  
  
th_min1 = th_max1 - arc_len_c1 / rc1; % 第一段圆弧最小角度  
th_min2 = th_min1 - pi; % 第二段圆弧最小角度  
th_max2 = th_min2 + arc_len_c2 / rc2; % 第二段圆弧最大角度  
  
% 第一段圆弧圆心坐标  
xc1 = tr * cos(th_in_end) + rc1 * cos(th_max1 - pi);  
yc1 = tr * sin(th_in_end) + rc1 * sin(th_max1 - pi);  
  
% 第二段圆弧圆心坐标  
xc2 = tr * cos(th_out_start) - rc2 * cos(th_max2);  
yc2 = tr * sin(th_out_start) - rc2 * sin(th_max2);  
  
% 绘制调头曲线  
t1 = linspace(th_min1, th_max1, 50);  
xarc1 = xc1 + rc1 * cos(t1);  
yarc1 = yc1 + rc1 * sin(t1);  
plot(xarc1, yarc1, 'g-', 'LineWidth', 2);  
  
t2 = linspace(th_min2, th_max2, 50);  
xarc2 = xc2 + rc2 * cos(t2);  
yarc2 = yc2 + rc2 * sin(t2);  
plot(xarc2, yarc2, 'm-', 'LineWidth', 2);  
  
plot([xc1, xc2], [yc1, yc2], 'ko', 'MarkerFaceColor', 'k'); % 绘制圆心  
  
legend('盘入螺线', '盘出螺线', '调头边界', '调头曲线1', '调头曲线2', '圆心');  
  
ts = 1; % 时间步长（秒）  
tot_seg = 223; % 总段数  
  
% 盘入阶段  
tin = 0:ts:100;  
dtheta_dt = @(t, th) 1 ./ (sc * sqrt(1 + th.^2));  
[tin, th_in] = ode45(dtheta_dt, tin, th_in_end);  
xhead_in = sc * th_in .* cos(th_in);  
yhead_in = sc * th_in .* sin(th_in);  
  
% 调头阶段  
tturn = 0:ts:(arc_len_c1 + arc_len_c2);  
th_c1 = th_max1 - tturn(tturn <= arc_len_c1) / rc1;  
th_c2 = th_min2 + (tturn(tturn > arc_len_c1) - arc_len_c1) / rc2;  
xhead_turn = [rc1 * cos(th_c1) + xc1, rc2 * cos(th_c2) + xc2];  
yhead_turn = [rc1 * sin(th_c1) + yc1, rc2 * sin(th_c2) + yc2];  
  
% 盘出阶段  
tout = 0:ts:(100 - length(tturn)*ts);  
dtheta_dt_out = @(t, th) 1 ./ (sc * sqrt(1 + (th + pi).^2));  
[tout, th_out] = ode45(dtheta_dt_out, tout, th_out_start);  
xhead_out = sc * (th_out + pi) .* cos(th_out);  
yhead_out = sc * (th_out + pi) .* sin(th_out);  
  
% 合并所有轨迹  
ttotal = [-flip(tin); tturn'; tout + tturn(end)] - 100;  
xhead_total = [flip(xhead_in); xhead_turn'; xhead_out];  
yhead_total = [flip(yhead_in); yhead_turn'; yhead_out];  
  
% 计算速度  
vx = diff(xhead_total) / ts;  
vy = diff(yhead_total) / ts;  
vtotal = sqrt(vx.^2 + vy.^2);  
  
% 辅助函数（保持原名，但内部变量名简化）  
function [x, y] = calculate_next_point_spiral_in(sp, x1, y1, th1, d)  
    k = sp / (2*pi);  
    fun = @(th) (k*th.*cos(th)-x1).^2 + (k*th.*sin(th)-y1).^2 - d^2;  
    options = optimset('Display', 'off');  
    th = fsolve(fun, th1 + 0.1, options);  
    while th <= th1 || abs(k*th - k*th1) > sp/2  
        th = fsolve(fun, th + 0.1, options);  
    end  
    x = k * th * cos(th);  
    y = k * th * sin(th);  
end  
  
function [x, y] = calculate_next_point_spiral_out(spiral_pitch, x1, y1, theta1, d)  
    k = spiral_pitch / (2*pi); % 计算螺线系数  
    % 定义距离方程，该方程描述了在螺线上距离(x1, y1)为d的点  
    fun = @(theta) ((k*(theta+pi).*cos(theta) - x1).^2 + ...  
                    (k*(theta+pi).*sin(theta) - y1).^2 - d^2);  
    options = optimoptions('fsolve', 'Display', 'off');
    % 初始猜测为theta1 - 0.1，因为螺线是向外扩展的，所以尝试减小角度  
    theta = fsolve(fun, theta1 - 0.1, options);  
      
    % 检查解是否在预期范围内，如果不是，则重新求解  
    while theta >= theta1 || abs(k*theta - k*theta1) > spiral_pitch/2  
        theta = fsolve(fun, theta - 0.1, options);  
    end  
      
    % 根据求解得到的theta计算下一个点的坐标  
    x = k * (theta + pi) * cos(theta);  
    y = k * (theta + pi) * sin(theta);  
end
function [x, y] = calculate_next_point_arc(x1, y1, theta1, d, r, xc, yc)  
  
    % 计算角度增量，这里使用了正弦定理的近似（假设d远小于r）  
    delta_theta = 2 * asin(d / (2 * r));  
    % 计算新的角度  
    theta = theta1 + delta_theta;  
    % 根据新的角度和圆心计算点的坐标  
    x = xc + r * cos(theta);  
    y = yc + r * sin(theta);  
end