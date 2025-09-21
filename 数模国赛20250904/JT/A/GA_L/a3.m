clear; clc;
tic; %计时

global M0 v_m O T_c r_t h_t v_down r_s t_e_ggg dt g A0 v_a; 

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

%模拟时间参数
t_start = 0;
t_end = 60;              %模拟总时长
dt = 0.01;               %时间步长
t_f_ggg = t_start:dt:t_end; 

fprintf('进行预计算以加速优化过程...\n');
m_t_ggg = g_m_ggg(t_f_ggg, M0, v_m, O); 
n_s_ggg = 50;
t_v_ggg = linspace(0, 2*pi, n_s_ggg); 
t_b_ggg = [T_c(1) + r_t * cos(t_v_ggg); T_c(2) + r_t * sin(t_v_ggg); repmat(T_c(3) - h_t/2, 1, n_s_ggg)];
t_p_ggg = [T_c(1) + r_t * sin(t_v_ggg); T_c(2) + r_t * cos(t_v_ggg); repmat(T_c(3) + h_t/2, 1, n_s_ggg)]; 
a_p_ggg = [t_b_ggg, t_p_ggg]; 

fprintf('开始进行三枚烟幕弹干扰方案优化 (遗传算法)...\n');

p_s_ggg = 600;      %种群   大小
n_g_ggg = 350;      %迭代   代数 
n_e_ggg = 12;       %基因   数量
c_r_ggg = 0.86;     %交叉     概率
m_r_ggg = 0.17;     %变异     概率
e_c_ggg = 2;        %精英    个体    数量
t_s_ggg = 3;        %锦标赛  选择  规模 


bounds.lower = [ 0, 70,    0,   0,    1,  70,    0,  0,    2, 70,    0,  0];
bounds.upper = [10, 140, 2*pi,  8,   25, 140, 2*pi,  8,   35, 140, 2*pi, 8];

p_p_ggg = zeros(p_s_ggg, n_e_ggg); 
for i = 1:n_e_ggg
    p_p_ggg(:, i) = bounds.lower(i) + (bounds.upper(i) - bounds.lower(i)) * rand(p_s_ggg, 1);
end

b_h_ggg = zeros(n_g_ggg, 1); 
g_f_ggg = -1;
g_i_ggg = zeros(1, n_e_ggg); 
d_i_ggg = (O - A0) / norm(O - A0);

for gen = 1:n_g_ggg
    fitness = zeros(p_s_ggg, 1);
    for i = 1:p_s_ggg
        ind = p_p_ggg(i, :);
        t_d1_ggg = ind(1);  v_f1_ggg = ind(2); t_y1_ggg = ind(3); t_l1_ggg = ind(4); % t_deploy1, v_fy1, theta_yaw1, t_delay1
        d_t2_ggg = ind(5);  v_f2_ggg = ind(6); t_y2_ggg = ind(7); t_l2_ggg = ind(8); % delta_t2, v_fy2, theta_yaw2, t_delay2
        d_t3_ggg = ind(9);  v_f3_ggg = ind(10);t_y3_ggg = ind(11);t_l3_ggg = ind(12);% delta_t3, v_fy3, theta_yaw3, t_delay3

        t_d2_ggg = t_d1_ggg + 1 + d_t2_ggg;
        t_d3_ggg = t_d2_ggg + 1 + d_t3_ggg; 
        
        if t_d3_ggg > t_end
            fitness(i) = 0;
            continue;
        end
        
        a_p1_ggg = A0 + v_a * t_d1_ggg * d_i_ggg; 
        d_v1_ggg = [v_f1_ggg * cos(t_y1_ggg), v_f1_ggg * sin(t_y1_ggg), 0]; 
        a_p2_ggg = a_p1_ggg + d_v1_ggg * (t_d2_ggg - t_d1_ggg);
        d_v2_ggg = [v_f2_ggg * cos(t_y2_ggg), v_f2_ggg * sin(t_y2_ggg), 0];
        a_p3_ggg = a_p2_ggg + d_v2_ggg * (t_d3_ggg - t_d2_ggg); 
        d_v3_ggg = [v_f3_ggg * cos(t_y3_ggg), v_f3_ggg * sin(t_y3_ggg), 0]; 

        params.grenade1 = struct('t_deploy', t_d1_ggg, 'deploy_pos', a_p1_ggg, 'deploy_v_vec', d_v1_ggg, 't_delay', t_l1_ggg, 'v_fy', v_f1_ggg, 'theta_yaw', t_y1_ggg);
        params.grenade2 = struct('t_deploy', t_d2_ggg, 'deploy_pos', a_p2_ggg, 'deploy_v_vec', d_v2_ggg, 't_delay', t_l2_ggg, 'v_fy', v_f2_ggg, 'theta_yaw', t_y2_ggg);
        params.grenade3 = struct('t_deploy', t_d3_ggg, 'deploy_pos', a_p3_ggg, 'deploy_v_vec', d_v3_ggg, 't_delay', t_l3_ggg, 'v_fy', v_f3_ggg, 'theta_yaw', t_y3_ggg);
        
        fitness(i) = c_t_ggg(params, m_t_ggg, a_p_ggg, t_f_ggg);
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
        
        n_p_ggg(i, :) = m_m_ggg(child1, m_r_ggg, bounds); % mutate -> m_m_ggg
        if i+1 <= p_s_ggg
           n_p_ggg(i+1, :) = m_m_ggg(child2, m_r_ggg, bounds);
        end
    end
    p_p_ggg = n_p_ggg;
end

fprintf('\n遗传算法优化完成！\n');

best_ind = g_i_ggg;
p_f_ggg.grenade1.t_deploy = best_ind(1);  p_f_ggg.grenade1.v_fy = best_ind(2); p_f_ggg.grenade1.theta_yaw = best_ind(3); p_f_ggg.grenade1.t_delay = best_ind(4);
d_t2_ggg = best_ind(5);  p_f_ggg.grenade2.v_fy = best_ind(6); p_f_ggg.grenade2.theta_yaw = best_ind(7); p_f_ggg.grenade2.t_delay = best_ind(8);
d_t3_ggg = best_ind(9);  p_f_ggg.grenade3.v_fy = best_ind(10);p_f_ggg.grenade3.theta_yaw = best_ind(11);p_f_ggg.grenade3.t_delay = best_ind(12);

p_f_ggg.grenade2.t_deploy = p_f_ggg.grenade1.t_deploy + 1 + d_t2_ggg;
p_f_ggg.grenade3.t_deploy = p_f_ggg.grenade2.t_deploy + 1 + d_t3_ggg;

g1 = p_f_ggg.grenade1;
g2 = p_f_ggg.grenade2;
g3 = p_f_ggg.grenade3;

g1.deploy_pos = A0 + v_a * g1.t_deploy * d_i_ggg;
g1.deploy_v_vec = [g1.v_fy * cos(g1.theta_yaw), g1.v_fy * sin(g1.theta_yaw), 0];
g2.deploy_pos = g1.deploy_pos + g1.deploy_v_vec * (g2.t_deploy - g1.t_deploy);
g2.deploy_v_vec = [g2.v_fy * cos(g2.theta_yaw), g2.v_fy * sin(g2.theta_yaw), 0];
g3.deploy_pos = g2.deploy_pos + g2.deploy_v_vec * (g3.t_deploy - g2.t_deploy);
g3.deploy_v_vec = [g3.v_fy * cos(g3.theta_yaw), g3.v_fy * sin(g3.theta_yaw), 0];

e_p1_ggg = g1.deploy_pos + g1.deploy_v_vec * g1.t_delay + [0, 0, -0.5 * g * g1.t_delay^2];
e_p2_ggg = g2.deploy_pos + g2.deploy_v_vec * g2.t_delay + [0, 0, -0.5 * g * g2.t_delay^2];
e_p3_ggg = g3.deploy_pos + g3.deploy_v_vec * g3.t_delay + [0, 0, -0.5 * g * g3.t_delay^2];

fprintf('正在计算各烟雾弹的单独贡献...\n');
o_t_1_ggg = c_i_ggg(g1, m_t_ggg, a_p_ggg, t_f_ggg);
o_t_2_ggg = c_i_ggg(g2, m_t_ggg, a_p_ggg, t_f_ggg);
o_t_3_ggg = c_i_ggg(g3, m_t_ggg, a_p_ggg, t_f_ggg);
fprintf('计算完成。\n\n');

fprintf('最大总遮蔽时长: %.2f s\n\n', g_f_ggg);
fprintf('--- 烟雾弹 1 ---\n');
fprintf('投放时间: %.2f s\n', g1.t_deploy);
fprintf('投放后无人机速度: %.2f m/s\n', g1.v_fy);
fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', g1.theta_yaw, rad2deg(g1.theta_yaw));
fprintf('延迟起爆时间: %.2f s\n', g1.t_delay);
fprintf('投放点: (%.2f, %.2f, %.2f)\n', g1.deploy_pos);
fprintf('起爆点: (%.2f, %.2f, %.2f)\n', e_p1_ggg);
fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', o_t_1_ggg);

fprintf('--- 烟雾弹 2 ---\n');
fprintf('投放时间: %.2f s\n', g2.t_deploy);
fprintf('投放后无人机速度: %.2f m/s\n', g2.v_fy);
fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', g2.theta_yaw, rad2deg(g2.theta_yaw));
fprintf('延迟起爆时间: %.2f s\n', g2.t_delay);
fprintf('投放点: (%.2f, %.2f, %.2f)\n', g2.deploy_pos);
fprintf('起爆点: (%.2f, %.2f, %.2f)\n', e_p2_ggg);
fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', o_t_2_ggg);

fprintf('--- 烟雾弹 3 ---\n');
fprintf('投放时间: %.2f s\n', g3.t_deploy);
fprintf('投放后无人机速度: %.2f m/s\n', g3.v_fy);
fprintf('投放后无人机方向 (偏航角): %.2f rad (%.2f 度)\n', g3.theta_yaw, rad2deg(g3.theta_yaw));
fprintf('延迟起爆时间: %.2f s\n', g3.t_delay);
fprintf('投放点: (%.2f, %.2f, %.2f)\n', g3.deploy_pos);
fprintf('起爆点: (%.2f, %.2f, %.2f)\n', e_p3_ggg);
fprintf('** 该弹单独有效遮蔽时长: %.2f s **\n\n', o_t_3_ggg);

toc;

figure;
plot(1:n_g_ggg, b_h_ggg, 'b-', 'LineWidth', 2);
title('遗传算法进化过程 (三枚烟幕弹)');
xlabel('代数 (Generation)');
ylabel('最优适应度 (最大遮蔽时长 s)');
grid on;

filename = 'a3.png';
saveas(gcf, filename);



function o_t_ggg = c_t_ggg(params, m_t_ggg, a_p_ggg, t_f_ggg) 
    global v_down r_s t_e_ggg dt g;
    g1 = params.grenade1; g2 = params.grenade2; g3 = params.grenade3;
    t_e1_ggg = g1.t_deploy + g1.t_delay; t_e2_ggg = g2.t_deploy + g2.t_delay; t_e3_ggg = g3.t_deploy + g3.t_delay;
    s_p1_ggg = g1.deploy_pos + g1.deploy_v_vec*g1.t_delay + [0, 0, -0.5*g*g1.t_delay^2];
    s_p2_ggg = g2.deploy_pos + g2.deploy_v_vec*g2.t_delay + [0, 0, -0.5*g*g2.t_delay^2];
    s_p3_ggg = g3.deploy_pos + g3.deploy_v_vec*g3.t_delay + [0, 0, -0.5*g*g3.t_delay^2];
    t_s1_ggg = t_e1_ggg; t_d1_ggg = t_e1_ggg + t_e_ggg;
    t_s2_ggg = t_e2_ggg; t_d2_ggg = t_e2_ggg + t_e_ggg;
    t_s3_ggg = t_e3_ggg; t_d3_ggg = t_e3_ggg + t_e_ggg;
    m_s_ggg = min([t_s1_ggg, t_s2_ggg, t_s3_ggg]);
    m_e_ggg = max([t_d1_ggg, t_d2_ggg, t_d3_ggg]);
    start_idx = floor(m_s_ggg / dt) + 1;
    end_idx = min(floor(m_e_ggg / dt) + 1, length(t_f_ggg));
    if start_idx > end_idx; o_t_ggg = 0; return; end
    o_s_ggg = 0;
    for i = start_idx:end_idx
        t = t_f_ggg(i); missile_pos = m_t_ggg(i, :);
        a_c_ggg = [];
        if t >= t_s1_ggg && t <= t_d1_ggg
            a_c_ggg = [a_c_ggg; s_p1_ggg + [0, 0, -v_down * (t - t_e1_ggg)]];
        end
        if t >= t_s2_ggg && t <= t_d2_ggg
            a_c_ggg = [a_c_ggg; s_p2_ggg + [0, 0, -v_down * (t - t_e2_ggg)]];
        end
        if t >= t_s3_ggg && t <= t_d3_ggg
            a_c_ggg = [a_c_ggg; s_p3_ggg + [0, 0, -v_down * (t - t_e3_ggg)]];
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
    g1 = g_p_ggg;
    t_e1_ggg = g1.t_deploy + g1.t_delay;
    s_p1_ggg = g1.deploy_pos + g1.deploy_v_vec*g1.t_delay + [0, 0, -0.5*g*g1.t_delay^2];
    t_s1_ggg = t_e1_ggg;
    t_d1_ggg = t_e1_ggg + t_e_ggg;
    start_idx = floor(t_s1_ggg / dt) + 1;
    end_idx = min(floor(t_d1_ggg / dt) + 1, length(t_f_ggg));
    if start_idx > end_idx; o_t_ggg = 0; return; end
    o_s_ggg = 0;
    for i = start_idx:end_idx
        t = t_f_ggg(i);
        missile_pos = m_t_ggg(i, :);
        c_c_ggg = s_p1_ggg + [0, 0, -v_down * (t - t_e1_ggg)];
        if i_f_ggg(missile_pos, a_p_ggg, c_c_ggg, r_s)
            o_s_ggg = o_s_ggg + 1;
        end
    end
    o_t_ggg = o_s_ggg * dt;
end

function i_f_ggg = i_f_ggg(missile_pos, t_p_ggg, s_c_ggg, r) 
    n_t_ggg = size(t_p_ggg, 2);
    for i = 1:n_t_ggg
        target_point = t_p_ggg(:, i)';
        i_o_ggg = false;
        for j = 1:size(s_c_ggg, 1)
            if c_s_ggg(missile_pos, target_point, s_c_ggg(j, :), r)
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
    v_r_ggg = Q - P; v_p_ggg = C - P;
    t = dot(v_p_ggg, v_r_ggg) / dot(v_r_ggg, v_r_ggg);
    if t < 0 || t > 1
        d_p_ggg = sum(v_p_ggg.^2); d_q_ggg = sum((C - Q).^2);
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

function individual = m_m_ggg(individual, m_r_ggg, bounds) 
    n_e_ggg = length(individual);
    for i = 1:n_e_ggg
        if rand < m_r_ggg
            range = bounds.upper(i) - bounds.lower(i);
            m_v_ggg = (range * 0.1) * randn;
            individual(i) = individual(i) + m_v_ggg;
            individual(i) = max(individual(i), bounds.lower(i));
            individual(i) = min(individual(i), bounds.upper(i));
        end
    end
end