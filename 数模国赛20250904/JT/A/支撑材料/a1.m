%初始位置 V
M0 = [20000, 0, 2000];   %导初坐标
v_m = 300;           %导速度
A0 = [17800, 0, 1800];      %飞机初坐标
v_a = 120;         %飞机速度
O = [0, 0, 0];          %原点

% 真实目标
T_c = [0, 200, 0];     %圆柱中心
r_t = 7;           %圆柱半径
h_t = 10;     %圆柱高度

% 干扰弹和烟雾
t1 = 1.5;         %投放时间
t_delay = 3.6;       %延时起爆时间
v_down = 3;        %烟雾云下沉速度
r_s = 10;             %烟雾有效遮蔽半径
t_s_e_g = 20;  %烟雾遮蔽时长
g = 9.8;

%模拟 时间
t_s_g = 0;
t_e_g = 40;         %总时长
dt = 0.0001;         %步长
t_vec = t_s_g:dt:t_e_g;%时间向量

%save
is_obscured = false(size(t_vec)); %存储每个步长 遮蔽方式


%计算  干扰弹  起爆点
t_deploy = t1;
t_exp = t1 + t_delay;

pptd_g = getPlanePos_g(t_deploy, A0, v_a);
%烟雾弹  平抛运动
smoke_exp_pos = [pptd_g(1) - v_a * t_delay, ...
                 pptd_g(2), ...
                 pptd_g(3) - 0.5 * g * t_delay^2];

for i = 1:length(t_vec)
    t = t_vec(i);
    
    %是否 在烟雾有效遮蔽  内（时间）
    if t >= t_exp && t <= t_exp + t_s_e_g
        missile_pos_ggg = gmp_g(t, M0, v_m, O);
        
        %烟雾球中心 移动方程
        smoke_pos = smoke_exp_pos + [0, 0, -v_down * (t - t_exp)];
        
        %真实目标  是否  完全遮蔽
        is_fully_obscured_now_ggg = true;
        
        %采样  真实目标
        num_samples = 10000;
        theta_vec = linspace(0, 2*pi, num_samples+1);
        theta_vec(end) = [];
        
        %检查 下底圆周  
        for j = 1:num_samples
            x_t = T_c(1) + r_t * cos(theta_vec(j));
            y_t = T_c(2) + r_t * sin(theta_vec(j));
            z_t = T_c(3) - h_t/2;
            target_point = [x_t, y_t, z_t];
            if ~cpsi_g(missile_pos_ggg, target_point, smoke_pos, r_s)
                is_fully_obscured_now_ggg = false;
                break;
            end
        end
        
        if ~is_fully_obscured_now_ggg
            is_obscured(i) = false;
            continue;
        end
        
        % 检查 上底圆周
        for j = 1:num_samples
            x_t = T_c(1) + r_t * cos(theta_vec(j));
            y_t = T_c(2) + r_t * sin(theta_vec(j));
            z_t = T_c(3) + h_t/2;
            target_point = [x_t, y_t, z_t];
            if ~cpsi_g(missile_pos_ggg, target_point, smoke_pos, r_s)
                is_fully_obscured_now_ggg = false;
                break;
            end
        end
        
        is_obscured(i) = is_fully_obscured_now_ggg;
    end
end

%被遮蔽    时间点位
if any(is_obscured)
    %总遮蔽  时长
    total_ggg = sum(is_obscured) * dt;
    fprintf('总遮蔽时长为：%.4f s。\n\n', total_ggg);
    
    %遮蔽时段
    fprintf('真实目标被完全遮蔽的时间段如下：\n');

    diff_obscured_g = diff([0; is_obscured(:); 0]);
    start_indices = find(diff_obscured_g == 1);
    end_indices = find(diff_obscured_g == -1) - 1;
    
    for k = 1:length(start_indices)
        start_time = t_vec(start_indices(k));
        end_time = t_vec(end_indices(k));
        fprintf('  - 从 %.4f s 到 %.4f s\n', start_time, end_time);
    end
    
else
    fprintf('真实目标未被完全遮蔽。\n');
end


%局部函数

function pos = gmp_g(t, M0, v_m, O)
    %导弹  位置
    dir_vec = (O - M0) / norm(O - M0);
    pos = M0 + v_m * t * dir_vec;
end

function pos = getPlanePos_g(t, A0, v_a)
    %飞机位置   按照之前的运动方向 
    pos = A0 + [-v_a * t, 0, 0];
end

function intersect_ggg = cpsi_g(P, Q, C, r)
    %检测是否 遮蔽    判断方式，连接线段，是否与球有交点
    vec_ray = Q - P;
    vec_PC = C - P;
    
    t = dot(vec_PC, vec_ray) / dot(vec_ray, vec_ray);
    
    if t < 0
        intersect_ggg = false;
        return;
    end
    
    dist_sq = norm(vec_PC)^2 - t^2 * norm(vec_ray)^2;
    
    if dist_sq <= r^2
        intersect_ggg = true;
    else
        intersect_ggg = false;
    end
end