% Dataset for DeepONet
% Data_arc:[f_j(x_i),y,G(f_j)(y)]
% G is a derivation operator.
function DON(method,save_flag,l)% l 代表高斯核函数的宽度
%% 检查形参
if nargin < 1
    error('缺少必要参数')
end
if nargin == 1
    save_flag = false;% 默认不保存
    l = 1;
end
if nargin == 2
    l = 1;
end
Gauss_flag = false;
%% 初始化
dx = 1/99; % 指定训练点的采样间隔
x = -0.5:dx:0.5; % 指定训练点(对gauss无效)
y = x; % 指定评估点
num = 49; % L_F 的上限 (从0开始)
M = 100; % 抽取 f 的个数
r_max = 1; % 重映射的最大值(对gauss无效)
r_min = 0; % 重映射的最小值(对gauss无效)
if strcmp(method,'le')
    [~,L_F] = Legendre(num);
elseif strcmp(method,'che')
    [~,L_F] = Chebyshev(num);
elseif strcmp(method,'gauss')
    Gauss_flag = true;
else
    error(['method:' method '不是le、che、gauss中的任意一种'])
end
data_F = [];
data = [];
test = [];
%% 创建训练集
if Gauss_flag == false
    for n = 1:1:M
        R = (r_max - r_min) * rand(1,length(L_F)) + r_min;
        f = R * L_F;
        g = diff(f);
        data_F = [data_F;f];
        f = matlabFunction(f);
        g = matlabFunction(g);
        temp = [kron(ones(length(y),1),f(x)),y',g(y')];
        data = [data;temp];
    end
else
    %% 高斯随机场
    x_all = [y y(end) + dx];% 由于差分会少一个元素,故预先延长一个元素
    K = kernel(l,x_all);
    R = chol(K + 1e-13 .* eye(length(x_all)),'lower'); % 加上少许摄动避免奇异(chol分解需为正定,而核函数仅为半正定)
    % 以下是另一种方法求高斯过程
    % N = randn(M,length(x));
    % N = R * N';
    % N = N';% 每 行 为一个样本
    % d_N = diff(N,1,2) ./ dx;
    % N = N(:,1:end -1);
    mu = zeros(1,size(R,1));
    sigma = R*R';
    N = mvnrnd(mu,sigma,M);
    d_N = diff(N,1,2) ./ dx;
    N = N(:,1:end-1);
    for ii = 1:1:M
        temp = [kron(ones(length(y),1),N(ii,:)),y',d_N(ii,:)'];
        data = [data;temp];
    end
end
%% 绘制训练集
plot(x,data(:,1:length(x)));
if strcmp(method,'le')
    text = ['$$u_j = \sum_{i = 0}^{' num2str(num) '}rand^j_i \cdot L_i,j = 1,\cdots,'...
        num2str(M) ';rand^j_i \sim U(' num2str(r_min) ',' num2str(r_max) ')$$'];
end
if strcmp(method,'che')
    text = ['$$u_j = \sum_{i = 0}^{' num2str(num) '}rand^j_i \cdot C_i,j = 1,\cdots,'...
        num2str(M) ';rand^j_i \sim U(' num2str(r_min) ',' num2str(r_max) ')$$'];
end
if strcmp(method,'gauss')
    text = ['$$u_k \sim GP(0,\Sigma),k=1,\cdots,' num2str(M) ',Kernel(x_i,x_j)=exp(-\frac{||x_i-x_j||^2}{2l^2}),l=' num2str(l)  '$$'];
end
title(text,Interpreter = "latex",FontSize=15);
%% 打乱训练集
data = data(randperm(size(data,1)),:);
%% 创建测试集
syms k
F = [sin(k) sec(k) 1.5^(k) k^3 k^4 exp(k)];
F_D = diff(F);
for ii = 1:1:length(F)
    f = matlabFunction(F(ii));
    f_d = matlabFunction(F_D(ii));
    temp = [kron(ones(length(y),1),f(x)),y',f_d(y')];
    test = [test;temp];
end
%% 保存数据集至指定位置
if save_flag == true
    save("D:\pythoncode\DON\data.mat","data")
    save("D:\pythoncode\DON\test.mat","test")
end
end
%% 高斯核函数
function K = kernel(l,data)
[X,Y] = meshgrid(data,data);
K = exp(-(X - Y) .^ 2 ./ (2*l^2));
end
%% 建立勒让德正交多项式
function [f,F] = Legendre(n)
syms x
if n == 0
    f = 1;
    F = 1;
end
if n == 1
    f = x;
    F = [1;x];
end
if n > 1
    F = [1;x];
    for ii = 2:1:n
        F(ii + 1) = ((2 * ii - 1) / ii) * x * F(ii) - ((ii - 1) / ii) * F(ii - 1);
        F(ii + 1) = simplify(F(ii + 1));
    end
    f = F(end);
end
end
%% 建立切比雪夫正交多项式
function [f,F] = Chebyshev(n)
syms x
if n == 0
    f = 1;
    F = 1;
end
if n == 1
    f = x;
    F = [1;x];
end
if n > 1
    F = [1;x];
    for ii = 2:1:n
        F(ii + 1) = 2 * x * F(ii) - F(ii - 1);
        F(ii + 1) = simplify(F(ii + 1));
    end
    f = F(end);
end
end