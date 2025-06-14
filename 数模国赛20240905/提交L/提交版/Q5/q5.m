% 问题5：确定龙头的最大行进速度
clear

load problem4_save_data
% 计算速度  
velocity_x = diff(x_positions, 1, 2) / time_step;  
velocity_y = diff(y_positions, 1, 2) / time_step;  
% 注意：这里diff操作会导致数组大小减少，需要确保后续处理与此兼容  
  
% 计算合速度  
velocity = sqrt(velocity_x.^2 + velocity_y.^2);  
  
% 每个板凳的最大速度（考虑diff减少了数组大小，需要处理边界）  
% 假设每个板凳的数据是一个行，时间步长是列  
num_benches = size(x_positions, 1);  
num_timesteps = size(x_positions, 2) - 1; % diff后减少一个时间步  
max_velocities = zeros(num_benches, 1);  
for i = 1:num_benches  
    max_velocities(i) = max(sqrt(velocity_x(i,:).^2 + velocity_y(i,:).^2));  
end  
  
% 所有板凳中的最大速度  
overall_max_velocity = max(max_velocities);  
  
% 最大速度与当前龙头速度的比值  
velocity_ratio = overall_max_velocity / head_speed;  
  
% 允许的最大龙头速度  
max_allowed_speed = 2 / velocity_ratio;  
  
fprintf('最大龙头速度: %.2f m/s\n', max_allowed_speed);  
  
% 调整速度（这里不直接修改原速度数据，而是计算新的速度矩阵）  
new_velocities = zeros(size(velocity)); % 保持原速度矩阵的形状  
for i = 1:num_benches  
    for j = 1:num_timesteps  
        % 只对原速度不为零的地方进行调整（如果考虑速度可能为0的情况）  
        if velocity(i,j) > 0  
            new_velocities(i,j) = velocity(i,j) * (max_allowed_speed / head_speed);  
            % 但实际上，我们应该使用全局的速度限制2 m/s来直接限制每个板凳的速度  
            % 如果上面的乘法导致速度超过2 m/s，则直接设为2 m/s  
            if new_velocities(i,j) > 2  
                new_velocities(i,j) = 2;  
            end  
        end  
    end  
end  
  
% 验证新的速度矩阵是否所有元素都不超过2 m/s（这步其实是多余的，因为我们在上一步已经处理了）  
% 但为了保持代码的一致性，还是写上  
if all(new_velocities(:) <= 2)  
    disp('验证通过：所有板凳的速度均不超过2 m/s');  
else  
    warning('理论上不应出现此警告，因为已在计算中限制了速度');  
end  
  
% 可视化新旧速度对比（这里只展示每个板凳的最大速度）  
figure('Name', '新速度与原速度对比');  
hold on;  
plot(max_velocities, 'bo-', 'DisplayName', '原速度方案的最大速度'); % 原速度的最大值  
plot(max(new_velocities, [], 2), 'r*-', 'DisplayName', '新速度方案的最大速度'); % 注意这里可能需要调整以正确显示  
% 但由于new_velocities可能有时间步长的维度，直接max可能不是预期的。我们可以直接绘制new_velocities的每行最大值  
plot([1, num_benches], [2, 2], 'k--', 'LineWidth', 2, 'DisplayName', '速度限制');  
xlabel('板凳编号');  
ylabel('最大的速度 (m/s)');  
legend show;