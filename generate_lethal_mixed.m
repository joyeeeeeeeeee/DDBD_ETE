% =========================================================================
%  脚本: generate_realistic_mixed.m (严酷但合理的 Mixed 数据)
% =========================================================================
clear; clc;
rng(888); 

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

% --- Mixed 噪声生成 (修正版) ---
% 1. 基础底噪
noise = randn(num_buses, num_hours) * 0.6; 

% 2. 稀疏尖峰：3% 的概率
spike_prob = 0.03;
spike_mask = rand(1, num_hours) < spike_prob; 

% [核心修改] 降低幅度
% 旧代码: (rand + 2.0) * 5.0 -> 10~15 sigma (太离谱)
% 新代码: (rand + 1.5) * 2.5 -> 3.75 ~ 6.25 sigma (严酷但可解)
spike_magnitudes = (rand(1, sum(spike_mask)) + 1.5) * 2.5; 

for t_idx = find(spike_mask)
    % 定向爆破：所有节点同时增加
    noise(:, t_idx) = noise(:, t_idx) + spike_magnitudes(find(find(spike_mask)==t_idx)); 
end

d_real = d_forecast + (d_forecast .* base_error_std .* noise);
d_real(d_real < 0) = 0;

save('data_mixed_case39.mat', 'd_forecast', 'd_real', 'mpc', '-v7.3');
fprintf('合理的 Mixed 数据已生成 (Spike ~4-6 sigma)。\n');