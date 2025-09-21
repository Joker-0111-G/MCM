function neg_obscured_time = objectiveFunction_3smoke(x)
    % --- 1. 解析输入变量 x ---
    t_deploy = [x(1), x(5), x(9)];
    deploy_pos_xy = [x(2), x(3); x(6), x(7); x(10), x(11)];
    t_delay = [x(4), x(8), x(12)];

    % --- 2. 模拟计算总遮蔽时长 ---
    % (实现上面 "3. 目标函数" 中描述的详细循环和判断逻辑)
    total_obscured_time = calculate_total_obscuration(t_deploy, deploy_pos_xy, t_delay);
    
    % --- 3. 返回负值 ---
    neg_obscured_time = -total_obscured_time;
end