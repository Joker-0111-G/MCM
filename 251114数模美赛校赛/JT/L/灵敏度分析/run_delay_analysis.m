%% 救援仿真灵敏度分析：人数与延迟时间
clc; clear; close all;

fprintf('=== 开始多维灵敏度分析 (人数 & 响应延迟) ===\n');

% 默认基准参数
BASE_SPEED = 1.2;      % 默认速度 (m/s)
BASE_DELAY = 60;       % 默认延迟 (s)
BASE_N_RESPONDERS = 6; % 默认人数 (人)

%% --- 分析 1: 救援人数 (N) 对 总时间的影响 ---
% 场景：假设延迟固定为 60秒，分析增加人手带来的收益
fprintf('1. 正在运行: 人数灵敏度测试 (N=1~12)...\n');

N_range = 1:12; 
results_N = zeros(length(N_range), 1);

for i = 1:length(N_range)
    n = N_range(i);
    % 调用格式: rescue_core_v12(人数, 延迟, 速度, 画图开关)
    t_total = rescue_core_v12(n, BASE_DELAY, BASE_SPEED, false); 
    results_N(i) = t_total;
    fprintf('  N = %d, Time = %.1f s\n', n, t_total);
end

%% --- 分析 2: 响应延迟 (Start Delay) 对 总时间的影响 ---
% 场景：假设人数固定为 6人，分析如果晚到现场，对由于烟雾扩散造成的总耗时影响
fprintf('2. 正在运行: 延迟时间灵敏度测试 (Delay=0~600s)...\n');

Delay_range = 0 : 30 : 600; % 从 0秒 到 10分钟，每30秒一个测点
results_Delay = zeros(length(Delay_range), 1);

for i = 1:length(Delay_range)
    d = Delay_range(i);
    % 这里改变的是第二个参数 d
    t_total = rescue_core_v12(BASE_N_RESPONDERS, d, BASE_SPEED, false);
    results_Delay(i) = t_total;
end

%% --- 结果可视化 ---
figure('Color','w', 'Name', '灵敏度分析结果', 'Position', [100, 100, 1200, 500]);

% 子图 1: 人数 vs 总时间
subplot(1, 2, 1);
plot(N_range, results_N, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
title(sprintf('人数灵敏度 (固定延迟=%ds)', BASE_DELAY));
xlabel('救援人数 (N)');
ylabel('总任务完成时间 (秒)');
xline(6, '--r', 'Baseline (N=6)');
% 添加数据标签
text(N_range(1), results_N(1), sprintf('%.0fs', results_N(1)), 'Vert','bottom');
text(N_range(end), results_N(end), sprintf('%.0fs', results_N(end)), 'Vert','bottom');

% 子图 2: 延迟 vs 总时间
subplot(1, 2, 2);
plot(Delay_range, results_Delay, 's-', 'LineWidth', 2, 'Color', [0.85 0.33 0.1], 'MarkerFaceColor', [0.85 0.33 0.1]);
hold on;

% 绘制一条“纯线性参考线” (y = x + base_time)
% 用于对比：如果烟雾不影响速度，增加1秒延迟应该只增加1秒总时间
% 如果实际曲线比这条线陡峭，说明烟雾导致了额外的减速惩罚
base_time_no_delay = results_Delay(1); 
linear_trend = base_time_no_delay + Delay_range;
plot(Delay_range, linear_trend, '--k', 'LineWidth', 1);
legend('仿真结果 (含烟雾减速影响)', '理想线性推移 (无烟雾影响)', 'Location', 'northwest');

grid on;
title(sprintf('响应延迟灵敏度 (固定人数=%d)', BASE_N_RESPONDERS));
xlabel('开始救援前的延迟 (秒)');
ylabel('总任务完成时间 (秒)');

fprintf('=== 分析完成 ===\n');
fprintf('关键结论预览:\n');
fprintf('  - 延迟 0秒 时耗时: %.1f s\n', results_Delay(1));
fprintf('  - 延迟 600秒 时耗时: %.1f s\n', results_Delay(end));
fprintf('  - 纯延迟增量: 600s, 实际耗时增量: %.1f s\n', results_Delay(end) - results_Delay(1));
if (results_Delay(end) - results_Delay(1)) > 600
    fprintf('  -> 观察到非线性增长：烟雾扩散导致了额外的 %.1f 秒作业耗时。\n', ...
        (results_Delay(end) - results_Delay(1)) - 600);
end