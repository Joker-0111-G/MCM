function [c, ceq] = constraintFunction_3smoke(x)
    global drone_max_speed; % 获取无人机最大速度
    P0 = [17800, 0]; % 无人机起点XY坐标

    % --- 解析变量 ---
    t1 = x(1); P1 = [x(2), x(3)];
    t2 = x(5); P2 = [x(6), x(7)];
    t3 = x(9); P3 = [x(10), x(11)];

    % --- 不等式约束 c(i) <= 0 ---
    c = zeros(5, 1);

    % 1. 时序约束
    c(1) = (t1 + 1) - t2; % t2 >= t1 + 1  =>  t1 + 1 - t2 <= 0
    c(2) = (t2 + 1) - t3; % t3 >= t2 + 1  =>  t2 + 1 - t3 <= 0

    % 2. 可达性约束
    % 从 P0到P1
    dist1 = norm(P1 - P0);
    c(3) = dist1 / t1 - drone_max_speed;
    
    % 从 P1到P2
    dist2 = norm(P2 - P1);
    c(4) = dist2 / (t2 - t1) - drone_max_speed;

    % 从 P2到P3
    dist3 = norm(P3 - P2);
    c(5) = dist3 / (t3 - t2) - drone_max_speed;
    
    % --- 等式约束 ceq(i) = 0 ---
    ceq = []; % 本问题没有等式约束
end