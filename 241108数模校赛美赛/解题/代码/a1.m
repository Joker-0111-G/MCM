clear;clc;close all
R=[0.48 0.49 0.5 0.51 0.52 ];
date=zeros(6,5);
figure; % 创建图形窗口
% 保持图形，以便在同一个窗口中绘制所有图形
for u=1:5
q=R(u);
date(:,u)=cau(q);
end

x=1:6;

for i=1:5

plot(x', date(1:6,i));
hold on
set(gcf,'Color',[0.68 0.85 0.9]);



grid on; % 添加网格
legend('show', 'Location', 'best'); % 显示图例，并自动选择最佳位置
% 添加标题和轴标签
title('Sensitivity Analysis');
xlabel('Order Of Each Index');
ylabel('Grey Relational Degree');
end
