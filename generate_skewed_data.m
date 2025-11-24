% =========================================================================
%  脚本: generate_skewed_data_no_toolbox.m 
%  (无需工具箱 - 利用指数分布叠加生成 Gamma 分布)
% =========================================================================
clear; clc;
rng(777); % 幸运种子

case_name = 'case39';
num_hours = 2880; 
avg_load_factor = 0.85; 
base_error_std = 0.03; 

mpc = loadcase(case_name);
static_loads = mpc.bus(:, 3); 
total_peak_load = sum(static_loads);

% 基础预测
time_vec = (1:num_hours)';
daily_cycle = -cos(2 * pi * time_vec / 24) * 0.35; 
d_total_forecast = total_peak_load * avg_load_factor * ...
    (1 + daily_cycle + randn(num_hours,1)*0.01);
dist_factors = static_loads / sum(static_loads);
d_forecast = dist_factors * d_total_forecast'; 
[num_buses, ~] = size(d_forecast);

% --- Skewed 噪声生成 (手动构造) ---
% 我们使用 Gamma(k, theta) 分布
% 为了不用工具箱，我们将 k 设为整数。
% k=2 时的偏度约为 1.41 (高斯为0)，依然具有显著的长尾特征。
k = 2; 
theta = 1.0;

fprintf('正在生成 Gamma(k=%d, theta=%.1f) 分布数据...\n', k, theta);

% 1. 手动生成 Gamma 分布
% 原理：Gamma(k, theta) = sum of k independent Exp(1/theta)
% Exp(lambda) 可以通过 -log(U)/lambda 生成，其中 U ~ Uniform(0,1)
raw_noise = zeros(num_buses, num_hours);
for i = 1:k
    % rand 是 MATLAB 核心函数，不需要工具箱
    U = rand(num_buses, num_hours);
    % 生成指数分布并累加
    raw_noise = raw_noise + (-log(U) * theta);
end

% 2. 强行中心化 (去均值)
% Gamma(k, theta) 的理论均值是 k*theta
mean_theoretical = k * theta;
centered_noise = raw_noise - mean_theoretical;

% 3. 归一化方差
% Gamma(k, theta) 的理论标准差是 sqrt(k)*theta
std_dev_theoretical = sqrt(k) * theta;
normalized_skewed_noise = centered_noise / std_dev_theoretical;

% 4. 最终噪声
noise = normalized_skewed_noise;

% 简单验证一下偏度 (Skewness)
% skewness 函数也是统计工具箱的，如果没有工具箱，我们可以手动算一下三阶矩
% 只是为了打印看看，不影响数据生成
try
    sk = mean(skewness(noise, 0, 2));
    fprintf('数据偏度检查: %.2f (正态分布应为0, 这里的正值代表右侧长尾)\n', sk);
catch
    fprintf('无法计算偏度(无工具箱)，但数据生成已完成。\n');
end

d_real = d_forecast + (d_forecast .* base_error_std .* noise);
d_real(d_real < 0) = 0;

save('data_skewed_case39.mat', 'd_forecast', 'd_real', 'mpc', '-v7.3');
fprintf('偏斜分布 (Skewed) 数据已生成。\n');

% 画个图看看形状
figure;
histogram(noise(1,:), 50);
title('Skewed Noise Distribution (Bus 1)');
xlabel('Sigma Deviation');
ylabel('Frequency');
grid on;