clear; clc;
tic; %计时

global M0 v_m O T_c r_t h_t v_down r_s t_e_ggg dt g; 
global A0_1 A0_2 A0_3; %无人机   初始   位置

% 导弹   信息
M0 = [20000, 0, 2000];   %导弹初始坐标
v_m = 300;               %导弹速度
O = [0, 0, 0];      

A0_1 = [17800, 0, 1800];    
A0_2 = [12000, 1400, 1400];   
A0_3 = [6000, -3000, 700];    

%真实目标
T_c = [0, 200, 0];   
r_t = 7;                 %半径
h_t = 10;          %高度


v_down = 3;              %下沉速度
r_s = 10;                %遮蔽半径
t_e_ggg = 20;            %遮蔽时长 
g = 9.8;               

%时间参数
t_start = 0;
t_end = 70;              %总时长
dt = 0.01;               %时间 步长 
t_f_ggg = t_start:dt:t_end;

fprintf('进行预计算以加速优化过程...\n');
m_t_ggg = g_m_ggg(t_f_ggg, M0, v_m, O); 
n_s_ggg = 50; 
t_v_ggg = linspace(0, 2*pi, n_s_ggg); 
t_b_ggg = [T_c(1) + r_t * cos(t_v_ggg); T_c(2) + r_t * sin(t_v_ggg); repmat(T_c(3) - h_t/2, 1, n_s_ggg)]; 
t_t_ggg = [T_c(1) + r_t * sin(t_v_ggg); T_c(2) + r_t * cos(t_v_ggg); repmat(T_c(3) + h_t/2, 1, n_s_ggg)]; 
a_p_ggg = [t_b_ggg, t_t_ggg]; 


fprintf('开始进行三无人机协同干扰方案优化 (遗传算法)...\n');

p_s_ggg = 900;      %种群   大小
n_g_ggg = 350;      %迭代   代数
n_e_ggg = 12;       %基因   数量
c_r_ggg = 0.8;      %交叉  概率
m_r_ggg = 0.25;     %变异   概率
e_c_ggg = 2;        %精英  个体  数量
t_s_ggg = 3;        %锦标赛  选择   规模


bounds.lower = [70, 0, 0, 0,  70, 0, 0, 0,  70, 0, 0, 0];
bounds.upper = [140, 15, 2*pi, 8, 140, 15, 2*pi, 8, 140, 15, 2*pi, 8];

p_p_ggg = zeros(p_s_ggg, n_e_ggg); 
for i = 1:n_e_ggg
    p_p_ggg(:, i) = bounds.lower(i) + (bounds.upper(i) - bounds.lower(i)) * rand(p_s_ggg, 1);
end

b_h_ggg = zeros(n_g_ggg, 1); 
g_f_ggg = -1; 
g_i_ggg = zeros(1, n_e_ggg); 

for gen = 1:n_g_ggg

    fitness = zeros(p_s_ggg, 1);
    for i = 1:p_s_ggg
        ind = p_p_ggg(i, :);
        
        params.p1 = struct('v_fy', ind(1), 't_deploy', ind(2), 'theta_yaw', ind(3), 't_delay', ind(4), 'A0', A0_1);
        params.p2 = struct('v_fy', ind(5), 't_deploy', ind(6), 'theta_yaw', ind(7), 't_delay', ind(8), 'A0', A0_2);
        params.p3 = struct('v_fy', ind(9), 't_deploy', ind(10), 'theta_yaw', ind(11), 't_delay', ind(12), 'A0', A0_3);

        fitness(i) = c_o_ggg(params, m_t_ggg, a_p_ggg, t_f_ggg);
    end
    
    [m_c_ggg, idx] = max(fitness);
    if m_c_ggg > g_f_ggg
        g_f_ggg = m_c_ggg;
        g_i_ggg = p_p_ggg(idx, :);
    end
    b_h_ggg(gen) = g_f_ggg;
    fprintf('第 %d 代: 当前最优遮蔽时间 = %.4f s, 全局最优 = %.4f s\n', gen, m_c_ggg, g_f_ggg);

    n_p_ggg = zeros(size(p_p_ggg)); 
    [~, s_i_ggg] = sort(fitness, 'descend'); 
    n_p_ggg(1:e_c_ggg, :) = p_p_ggg(s_i_ggg(1:e_c_ggg), :);
    
    for i = (e_c_ggg + 1):2:p_s_ggg

        parent1 = p_p_ggg(s_t_ggg(fitness, t_s_ggg), :); 
        parent2 = p_p_ggg(s_t_ggg(fitness, t_s_ggg), :);
        
        child1 = parent1; child2 = parent2;
        if rand < c_r_ggg
            alpha = rand;
            child1 = alpha * parent1 + (1 - alpha) * parent2;
            child2 = alpha * parent2 + (1 - alpha) * parent1;
        end
        
        n_p_ggg(i, :) = m_m_ggg(child1, m_r_ggg, bounds); 
        if i+1 <= p_s_ggg
           n_p_ggg(i+1, :) = m_m_ggg(child2, m_r_ggg, bounds);
        end
    end
    p_p_ggg = n_p_ggg;
end

fprintf('\n遗传算法优化完成！\n');

best_ind = g_i_ggg;
b_p_ggg.p1 = struct('v_fy', best_ind(1), 't_deploy', best_ind(2), 'theta_yaw', best_ind(3), 't_delay', best_ind(4), 'A0', A0_1);
b_p_ggg.p2 = struct('v_fy', best_ind(5), 't_deploy', best_ind(6), 'theta_yaw', best_ind(7), 't_delay', best_ind(8), 'A0', A0_2);
b_p_ggg.p3 = struct('v_fy', best_ind(9), 't_deploy', best_ind(10), 'theta_yaw', best_ind(11), 't_delay', best_ind(12), 'A0', A0_3);

if g_f_ggg > 0
    fprintf('最大总遮蔽时长: %.2f s\n\n', g_f_ggg);
    
    uavs = {'FY1', 'FY2', 'FY3'};
    ps = {b_p_ggg.p1, b_p_ggg.p2, b_p_ggg.p3};
    
    for i = 1:3
        p = ps{i};
        d_v_ggg = [p.v_fy * cos(p.theta_yaw), p.v_fy * sin(p.theta_yaw), 0];
        d_p_ggg = p.A0 + d_v_ggg * p.t_deploy;
        e_p_ggg = d_p_ggg + d_v_ggg * p.t_delay + [0, 0, -0.5 * g * p.t_delay^2];
        
        i_t_ggg = c_i_ggg(p, m_t_ggg, a_p_ggg, t_f_ggg);

        fprintf('--- %s 无人机最优参数 ---\n', uavs{i});
        fprintf('投放后飞行速度: %.2f m/s\n', p.v_fy);
        fprintf('飞行及投放时间: %.2f s\n', p.t_deploy);
        fprintf('飞行方向 (偏航角): %.2f rad (%.2f 度)\n', p.theta_yaw, rad2deg(p.theta_yaw));
        fprintf('延迟引爆时间: %.2f s\n', p.t_delay);
        fprintf('投放点: (%.2f, %.2f, %.2f)\n', d_p_ggg);
        fprintf('起爆点: (%.2f, %.2f, %.2f)\n', e_p_ggg);
        fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', i_t_ggg);
    end
else
    fprintf('未找到有效的遮蔽方案。\n');
end
toc;

figure;
plot(1:n_g_ggg, b_h_ggg, 'b-', 'LineWidth', 2);
title('遗传算法进化过程 (三无人机协同)');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大遮蔽时长 s)');
grid on;

filename = 'a4.png';
saveas(gcf, filename);



function o_t_ggg = c_o_ggg(params, m_t_ggg, a_p_ggg, t_f_ggg) 
    global v_down r_s t_e_ggg dt g;
    p = {params.p1, params.p2, params.p3};
    t_exp = zeros(1, 3);
    s_p_ggg = zeros(3, 3);
    for i = 1:3
        d_v_ggg = [p{i}.v_fy * cos(p{i}.theta_yaw), p{i}.v_fy * sin(p{i}.theta_yaw), 0];
        d_p_ggg = p{i}.A0 + d_v_ggg * p{i}.t_deploy;
        t_exp(i) = p{i}.t_deploy + p{i}.t_delay;
        s_p_ggg(i, :) = d_p_ggg + d_v_ggg*p{i}.t_delay + [0, 0, -0.5*g*p{i}.t_delay^2];
    end
    m_s_ggg = min(t_exp);
    m_e_ggg = max(t_exp) + t_e_ggg;
    start_idx = floor(m_s_ggg / dt) + 1;
    end_idx = min(floor(m_e_ggg / dt) + 1, length(t_f_ggg));
    if start_idx > end_idx || start_idx < 1
        o_t_ggg = 0;
        return;
    end
    o_s_ggg = 0;
    for i = start_idx:end_idx
        t = t_f_ggg(i);
        missile_pos = m_t_ggg(i, :);
        a_c_ggg = [];
        for j = 1:3
            if t >= t_exp(j) && t <= (t_exp(j) + t_e_ggg)
                delta_t = t - t_exp(j);
                c_p_ggg = s_p_ggg(j, :) + [0, 0, -v_down * delta_t];
                a_c_ggg = [a_c_ggg; c_p_ggg];
            end
        end
        if isempty(a_c_ggg); continue; end
        if i_f_ggg(missile_pos, a_p_ggg, a_c_ggg, r_s)
            o_s_ggg = o_s_ggg + 1;
        end
    end
    o_t_ggg = o_s_ggg * dt;
end

function o_t_ggg = c_i_ggg(g_p_ggg, m_t_ggg, a_p_ggg, t_f_ggg)
    global v_down r_s t_e_ggg dt g;
    p1 = g_p_ggg;
    d_v_ggg = [p1.v_fy * cos(p1.theta_yaw), p1.v_fy * sin(p1.theta_yaw), 0];
    d_p_ggg = p1.A0 + d_v_ggg * p1.t_deploy;
    t_exp1 = p1.t_deploy + p1.t_delay;
    s_p_ggg = d_p_ggg + d_v_ggg*p1.t_delay + [0, 0, -0.5*g*p1.t_delay^2];
    t_s_ggg_local = t_exp1;
    t_e_ggg_local = t_exp1 + t_e_ggg;
    start_idx = floor(t_s_ggg_local / dt) + 1;
    end_idx = min(floor(t_e_ggg_local / dt) + 1, length(t_f_ggg));
    if start_idx > end_idx || start_idx < 1; o_t_ggg = 0; return; end
    o_s_ggg = 0;
    for i = start_idx:end_idx
        t = t_f_ggg(i);
        missile_pos = m_t_ggg(i, :);
        c_c_ggg = s_p_ggg + [0, 0, -v_down * (t - t_exp1)];
        if i_f_ggg(missile_pos, a_p_ggg, c_c_ggg, r_s)
            o_s_ggg = o_s_ggg + 1;
        end
    end
    o_t_ggg = o_s_ggg * dt;
end

function i_f_ggg = i_f_ggg(missile_pos, t_p_ggg, s_c_ggg, r)
    n_t_ggg = size(t_p_ggg, 2);
    n_h_ggg = size(s_c_ggg, 1);
    for i = 1:n_t_ggg
        target_point = t_p_ggg(:, i)';
        i_o_ggg = false;
        for j = 1:n_h_ggg
            smoke_center = s_c_ggg(j, :);
            if c_s_ggg(missile_pos, target_point, smoke_center, r)
                i_o_ggg = true;
                break;
            end
        end
        if ~i_o_ggg
            i_f_ggg = false;
            return;
        end
    end
    i_f_ggg = true;
end

function d_i_ggg = c_s_ggg(P, Q, C, r) 
    v_r_ggg = Q - P;
    v_p_ggg = C - P;
    t = dot(v_p_ggg, v_r_ggg) / dot(v_r_ggg, v_r_ggg);
    if t < 0 || t > 1
        d_p_ggg = sum(v_p_ggg.^2);
        d_q_ggg = sum((C - Q).^2);
        d_i_ggg = (d_p_ggg <= r^2) || (d_q_ggg <= r^2);
    else
        d_s_ggg = sum(v_p_ggg.^2) - (t^2) * dot(v_r_ggg, v_r_ggg);
        d_i_ggg = (d_s_ggg <= r^2);
    end
end

function p_m_ggg = g_m_ggg(t_v_ggg, M0, v_m, O) 
    d_v_ggg = (O - M0) / norm(O - M0);
    p_m_ggg = M0 + t_v_ggg' * (v_m * d_v_ggg);
end


function s_i_ggg = s_t_ggg(fitness, t_s_ggg) 
    p_s_ggg = length(fitness);
    indices = randi(p_s_ggg, 1, t_s_ggg);
    [~, b_i_ggg] = max(fitness(indices));
    s_i_ggg = indices(b_i_ggg);
end

function i_i_ggg = m_m_ggg(individual, m_r_ggg, bounds)
    n_e_ggg = length(individual);
    i_i_ggg = individual;
    for i = 1:n_e_ggg
        if rand < m_r_ggg
            range = bounds.upper(i) - bounds.lower(i);
            m_v_ggg = (range * 0.1) * randn;
            i_i_ggg(i) = i_i_ggg(i) + m_v_ggg;
            i_i_ggg(i) = max(i_i_ggg(i), bounds.lower(i));
            i_i_ggg(i) = min(i_i_ggg(i), bounds.upper(i));
        end
    end
end