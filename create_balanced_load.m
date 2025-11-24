% =========================================================================
%  脚本: 2_create_balanced_load.m
%  功能: 生成“可求解”的负荷数据 (Time x Buses)
% =========================================================================
clear; clc;

% --- 配置 ---
num_hours = 2880; 
case_name = 'case39';

% [关键调整] 降低压力，保证可行性
avg_load_factor = 0.70;   % 70% 负载 (之前是 90%+)
error_std_percent = 0.03; % 3% 误差 (之前是 5-8%)

rng(42); % 固定种子保证可复现

% --- 加载 ---
mpc = loadcase(case_name);
static_loads = mpc.bus(:, 3); 
total_peak_load = sum(static_loads);

% --- 生成总负荷曲线 ---
time_vec = (1:num_hours)';
% 日周期 + 周周期 + 随机噪声
daily_cycle = -cos(2 * pi * time_vec / 24) * 0.20; 
weekly_cycle = zeros(num_hours, 1);
day_idx = mod(floor((time_vec-1)/24), 7);
weekly_cycle(day_idx >= 5) = -0.10; % 周末低谷

% 基础预测值
d_total_forecast = total_peak_load * avg_load_factor * ...
    (1 + daily_cycle + weekly_cycle + randn(num_hours,1)*0.01);

% --- 分配到节点 ---
dist_factors = static_loads / sum(static_loads);
% 结果维度: (2880, 39) -> 每一行是一个时刻
d_forecast = d_total_forecast * dist_factors'; 

% --- 生成真实值 (加入误差) ---
noise = randn(size(d_forecast));
d_real = d_forecast .* (1 + error_std_percent .* noise);
d_real(d_real < 0) = 0;

% 保存
save('balance_synthetic_data_package.mat', 'd_forecast', 'd_real', 'mpc', '-v7.3');
fprintf('负荷数据已生成: Hours=%d, LoadFactor=%.2f, Err=%.2f\n', ...
    num_hours, avg_load_factor, error_std_percent);