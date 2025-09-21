clear; clc;
tic; %计时

global M0 v_m O T_c r_t h_t v_down r_s t_e_ggg dt g A0 v_a; % t_smoke_effective -> t_e_ggg


M0 = [20000, 0, 2000];   %导弹   初   始    坐标 
v_m = 300;               %导弹   速度
A0 = [17800, 0, 1800];  
v_a = 120;               %无人机    投放前   速度  假设值
O = [0, 0, 0];       

%真实目标
T_c = [0, 200, 0];     
r_t = 7;               
h_t = 10;             

%干扰弹和烟雾
v_down = 3;         %烟下沉速度
r_s = 10;               %半径
t_e_ggg = 20;     %遮蔽时长 
g = 9.8;         

% 模拟时间参数
t_start = 0;
t_end = 40;              % 模拟总时长
dt = 0.01;               % 时间步长
t_f_ggg = t_start:dt:t_end; 

fprintf('进行预计算以加速优化过程...\n');

m_t_ggg = g_p_ggg(t_f_ggg, M0, v_m, O); 

n_s_ggg = 100;
t_v_ggg = linspace(0, 2*pi, n_s_ggg);
t_b_ggg = [T_c(1) + r_t * cos(t_v_ggg); ...
           T_c(2) + r_t * sin(t_v_ggg); ...
           repmat(T_c(3) - h_t/2, 1, n_s_ggg)]; 
t_p_ggg = [T_c(1) + r_t * sin(t_v_ggg); ...
           T_c(2) + r_t * cos(t_v_ggg); ...
           repmat(T_c(3) + h_t/2, 1, n_s_ggg)]; 
a_p_ggg = [t_b_ggg, t_p_ggg];



fprintf('开始进行FY1无人机干扰方案优化 (遗传算法)...\n');


p_s_ggg = 600;       % 种群   大小
n_g_ggg = 250;      % 迭代    代数
c_r_ggg = 0.8;       % 交叉    概率
m_r_ggg = 0.16;        % 变异     概率
e_c_ggg = 2;          % 精英个 体       数量
t_o_ggg = 3;        % 锦标赛        选择         规模

bounds.lower = [70, 0, 0, 0];
bounds.upper = [140, 20, 2*pi, 8];

p_p_ggg = zeros(p_s_ggg, 4);
for i = 1:4
    p_p_ggg(:, i) = bounds.lower(i) + (bounds.upper(i) - bounds.lower(i)) * rand(p_s_ggg, 1);
end

b_h_ggg = zeros(n_g_ggg, 1);
g_f_ggg = -1; 
g_i_ggg = zeros(1, 4); 

for gen = 1:n_g_ggg
    fitness = zeros(p_s_ggg, 1);
    d_i_ggg = (O - A0) / norm(O - A0); 
    
    for i = 1:p_s_ggg
        individual = p_p_ggg(i, :);
        v_fy1 = individual(1);
        t_deploy = individual(2);
        theta_yaw = individual(3);
        t_delay = individual(4);
        
        d_p_ggg = A0 + v_a * t_deploy * d_i_ggg; 
        d_v_ggg = [v_fy1 * cos(theta_yaw), v_fy1 * sin(theta_yaw), 0];
        
        fitness(i) = c_o_ggg(t_deploy, d_p_ggg, d_v_ggg, t_delay, ...
                             m_t_ggg, a_p_ggg, t_f_ggg); 
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
        parent1_idx = s_t_ggg(fitness, t_o_ggg);
        parent2_idx = s_t_ggg(fitness, t_o_ggg);
        parent1 = p_p_ggg(parent1_idx, :);
        parent2 = p_p_ggg(parent2_idx, :);
        
        child1 = parent1;
        child2 = parent2;
        if rand < c_r_ggg
            alpha = rand;
            child1 = alpha * parent1 + (1 - alpha) * parent2;
            child2 = alpha * parent2 + (1 - alpha) * parent1;
        end
        
        child1 = m_m_ggg(child1, m_r_ggg, bounds);
        child2 = m_m_ggg(child2, m_r_ggg, bounds);

        n_p_ggg(i, :) = child1;
        if i+1 <= p_s_ggg
           n_p_ggg(i+1, :) = child2;
        end
    end
    p_p_ggg = n_p_ggg;
end


fprintf('\n遗传算法优化完成！\n');

b_v_ggg = g_i_ggg(1); 
b_t_ggg = g_i_ggg(2); 
b_a_ggg = g_i_ggg(3); 
b_d_ggg = g_i_ggg(4);
m_o_ggg = g_f_ggg;   

if m_o_ggg > 0
    b_i_ggg = (O - A0) / norm(O - A0);
    b_p_ggg = A0 + v_a * b_t_ggg * b_i_ggg; 
    b_e_ggg = [b_v_ggg * cos(b_a_ggg), b_v_ggg * sin(b_a_ggg), 0];
    
    b_x_ggg = b_p_ggg + b_e_ggg * b_d_ggg + [0, 0, -0.5 * g * b_d_ggg^2];

    fprintf('最优无人机速度: %.2f m/s\n', b_v_ggg);
    fprintf('最优投放时间: %.2f s\n', b_t_ggg);
    fprintf('最优延迟起爆时间: %.2f s\n', b_d_ggg);
    fprintf('最优方向 (偏航角): %.2f rad (%.2f 度)\n', b_a_ggg, rad2deg(b_a_ggg));
    fprintf('最大遮蔽时长: %.2f s\n', m_o_ggg);
    fprintf('最优投放点: (%.2f, %.2f, %.2f)\n', b_p_ggg);
    fprintf('最优起爆点: (%.2f, %.2f, %.2f)\n', b_x_ggg);
else
    fprintf('在搜索范围内未找到有效的遮蔽方案。\n');
end
toc;

figure;
plot(1:n_g_ggg, b_h_ggg, 'b-', 'LineWidth', 2);
title('遗传算法进化过程');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大遮蔽时长 s)');
grid on;

filename = 'a2.png';
saveas(gcf, filename);

function o_t_ggg = c_o_ggg(t_d_ggg, d_p_ggg, d_v_ggg, l_d_ggg, m_t_ggg, a_p_ggg, t_f_ggg)
    global v_down r_s t_e_ggg dt g;
    t_exp = t_d_ggg + l_d_ggg;
    s_e_ggg = t_exp + t_e_ggg; 
    start_idx = floor(t_exp / dt) + 1;
    end_idx = min(floor(s_e_ggg / dt) + 1, length(t_f_ggg));
    if start_idx > end_idx
        o_t_ggg = 0; 
        return;
    end
    o_s_ggg = 0; 
    s_p_ggg = d_p_ggg + d_v_ggg*l_d_ggg + [0, 0, -0.5*g*l_d_ggg^2]; 
    for i = start_idx:end_idx
        t = t_f_ggg(i);
        missile_pos = m_t_ggg(i, :);
        e_t_ggg = t - t_exp; 
        s_o_ggg = s_p_ggg + [0, 0, -v_down * e_t_ggg]; 
        if c_i_ggg(missile_pos, a_p_ggg, s_o_ggg, r_s) 
            o_s_ggg = o_s_ggg + 1;
        end
    end
    o_t_ggg = o_s_ggg * dt;
end

function p_m_ggg = g_p_ggg(t_v_ggg, M0, v_m, O) 
    d_v_ggg = (O - M0) / norm(O - M0); 
    p_m_ggg = M0 + t_v_ggg' * (v_m * d_v_ggg); 
end

function i_f_ggg = c_i_ggg(P, q_m_ggg, C, r) 
    v_r_ggg = q_m_ggg' - P; 
    v_p_ggg = C - P;   
    t = dot(v_r_ggg, repmat(v_p_ggg, size(v_r_ggg, 1), 1), 2) ./ dot(v_r_ggg, v_r_ggg, 2);
    v_t_ggg = t > 0; 
    if ~any(v_t_ggg)
        i_f_ggg = false; 
        return;
    end
    d_s_ggg = norm(v_p_ggg)^2 - (t.^2) .* dot(v_r_ggg, v_r_ggg, 2); 
    intersect = (d_s_ggg <= r^2);
    i_f_ggg = all(intersect(v_t_ggg));
end

function s_e_ggg = s_t_ggg(fitness, t_o_ggg)
    p_s_ggg = length(fitness);
    indices = randi(p_s_ggg, 1, t_o_ggg);
    t_i_ggg = fitness(indices); 
    [~, b_t_ggg] = max(t_i_ggg); 
    s_e_ggg = indices(b_t_ggg); 
end

function individual = m_m_ggg(individual, m_r_ggg, bounds)
    for i = 1:length(individual)
        if rand < m_r_ggg
            range = bounds.upper(i) - bounds.lower(i);
            m_v_ggg = (range * 0.1) * randn;
            individual(i) = individual(i) + m_v_ggg;
            individual(i) = max(individual(i), bounds.lower(i));
            individual(i) = min(individual(i), bounds.upper(i));
        end
    end
end