% =========================================================================
%  脚本 2: gen_v2_balanced_load.m
%  功能: 生成平衡负荷数据 (v2版本)
% =========================================================================
clear; clc;

num_hours = 2880; 
case_name = 'case39';
% [参数调整] 确保有解的“甜蜜点”
avg_load_factor = 0.72;   
error_std_percent = 0.03; 

rng(2024); % 新的随机种子

mpc = loadcase(case_name);
static_loads = mpc.bus(:, 3); 
total_peak = sum(static_loads);

% --- 生成时间序列 ---
t = (1:num_hours)';
daily = -cos(2 * pi * t / 24) * 0.20; 
weekly = zeros(num_hours, 1);
weekly(mod(floor((t-1)/24), 7) >= 5) = -0.10; % 周末

% 总负荷
d_total = total_peak * avg_load_factor * (1 + daily + weekly + randn(num_hours,1)*0.015);

% 分配到节点 (Time x Buses)
d_forecast = d_total * (static_loads / total_peak)';

% 真实值 (加入误差)
noise = randn(size(d_forecast));
d_real = d_forecast .* (1 + error_std_percent .* noise);
d_real(d_real < 0) = 0;

% [关键] 保存为 v2_balanced_load_data.mat
save('v2_balanced_load_data.mat', 'd_forecast', 'd_real', 'mpc', '-v7.3');
fprintf('步骤 2 (v2) 完成: v2_balanced_load_data.mat 已生成。\n');