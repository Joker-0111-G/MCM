%应用二：灰色关联分析评价河流情况
clear;clc;
load X.mat;
%获取行数列数
r = size(X,1);
c = size(X,2);
%首先，把我们的原始指标矩阵正向化
%第二列中间型--->极大型
middle = input("请输入最佳的中间值：");
M = max(abs(X(:,2)-middle));
for i=1:r
      X(i,2) = 1-abs(X(i,2)-middle)/M;
end
%第三列极小型--->极大型
max_value = max(X(:,3)); 
X(:,3) = abs(X(:,3)-max_value);
%第四列区间型--->极大型
a = input("请输入区间的下界：");
b = input("请输入区间的下界：");
M = max(a-min(X(:,4)),max(X(:,4))-b);
for i=1:r
       if (X(i,4)<a)
            X(i,4) = 1-(a-X(i,4))/M;
       elseif (X(i,4)<=b&&X(i,4)>=a)
           X(i,4) = 1;
       else
           X(i,4) = 1-(X(i,4)-b)/M;
       end
end
disp("正向化后的矩阵为：");
disp(X);
%把正向化后的矩阵进行预处理，消除量纲的影响
avg = repmat(mean(X),r,1);
new_X = X./avg;
%将预处理后的矩阵每一行的最大值取出，当成母序列(虚构的)
Y = max(new_X,[],2);
%计算各个指标和母序列的灰色关联度
%先把new_X矩阵所有元素都减去母序列中同行的元素，并取绝对值
Y2 = repmat(Y,1,c);
new_X = abs(new_X-Y2);
a = min(min(new_X)); %全矩阵最小值
b = max(max(new_X)); %全矩阵最大值
ro = 0.5;
new_X = (a+ro*b)./(new_X+ro*b);
disp("各个指标对于母序列的灰色关联度为：");
gamma = mean(new_X)
%计算各个指标的权重
disp("各个指标的权重为：");
weight = gamma./(sum(gamma,2))
%-------------------------------------------------------------------------------------------------------
%继续TOPSIS的步骤：对正向化后的矩阵X进行标准化（原矩阵除以每一列元素平方之和的开方）
temp1 = X.*X;               %先让每每一个元素平方
temp2 = sum(temp1);         %再对每一列求和
temp3 = temp2.^0.5;         %再把结果开方
temp4 = repmat(temp3,r,1);  %把开方后的结果按行复制r行
disp("******标准化后的矩阵为：");
Z = X./temp4               %原矩阵除以每一列元素平方之和的开方
Z_max = max(Z)           %获得Z每一列中最大的元素
Z_min = min(Z)           %获得Z每一列中最小的元素
D_max = sum(weight.*(Z-repmat(Z_max,r,1)).^2,2).^0.5
D_min = sum(weight.*(Z-repmat(Z_min,r,1)).^2,2).^0.5
disp("该矩阵得分为：")
S = D_min./(D_max+D_min)
disp("矩阵归一化后得分为：");
S = S./(repmat(sum(S),r,1))
