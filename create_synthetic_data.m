% =========================================================================
%  脚本: prepare_synthetic_data.m (高压力场景版)
% =========================================================================

clear; clc;

%% 1. 配置参数
case_name = 'case39';
num_hours = 2880; 

% --- 关键调整 1: 提高误差 ---
% 增加到 5%，让波动更明显，容易导致越限
error_std_percent = 0.05;   

% --- 关键调整 2: 提高系统负荷 ---
% 之前是 0.7，现在提高到 0.9。
% 这会让系统运行在接近满载的状态，稍微一点波动就会造成麻烦。
avg_load_factor = 0.90;     

daily_variation = 0.25;    
weekly_variation = 0.10;    
random_noise_std = 0.02;    

rng(100); % 修改随机种子

%% 2. 加载模型
fprintf('正在加载 %s...\n', case_name);
mpc = loadcase(case_name);
static_loads = mpc.bus(:, 3); 
total_peak_load = sum(static_loads);

%% 3. 合成负荷曲线
time_vec = (1:num_hours)';
daily_cycle = -cos(2 * pi * time_vec / 24) * (daily_variation / 2);
day_of_week = mod(floor((time_vec-1)/24), 7) + 1;
weekly_cycle = zeros(num_hours, 1);
weekly_cycle(day_of_week >= 6) = -weekly_variation; 

d_total_forecast = total_peak_load * avg_load_factor * ...
    (1 + daily_cycle + weekly_cycle + randn(num_hours,1)*random_noise_std);

%% 4. 分配到节点
dist_factors = static_loads / sum(static_loads);
d_forecast = dist_factors * d_total_forecast';

%% 5. 生成真实值
fprintf('生成真实负荷 (误差 Std = %.1f%%)...\n', error_std_percent*100);
noise = randn(size(d_forecast));
errors = d_forecast .* error_std_percent .* noise;

d_real = d_forecast + errors;
d_real(d_real < 0) = 0; 

save('synthetic_data_package.mat', 'd_forecast', 'd_real', 'mpc', '-v7.3');
fprintf('高压力数据已生成。\n');