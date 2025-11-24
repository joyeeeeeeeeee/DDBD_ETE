% =========================================================================
%  脚本: batch_generate_all_data.m
%  功能: 一次性生成 Gaussian, Student-T, Mixed 三种数据
% =========================================================================
clear; clc;

%% 1. 全局配置
case_name = 'case39';
num_hours = 2880; 
avg_load_factor = 0.8; 
base_error_std = 0.03; 
rng(123); % 固定种子

% 定义我们要生成的三种模式列表
modes_list = {'Gaussian', 'Student-T', 'Mixed'};

%% 2. 加载物理模型 (只加载一次)
mpc = loadcase(case_name);
static_loads = mpc.bus(:, 3); 
total_peak_load = sum(static_loads);

%% 3. 生成基础预测负荷 (所有模式共用同一套 Forecast，方便对比)
time_vec = (1:num_hours)';
daily_cycle = -cos(2 * pi * time_vec / 24) * 0.35; 
d_total_forecast = total_peak_load * avg_load_factor * ...
    (1 + daily_cycle + randn(num_hours,1)*0.01);

dist_factors = static_loads / sum(static_loads);
d_forecast = dist_factors * d_total_forecast'; 
[num_buses, ~] = size(d_forecast);

%% 4. 循环生成三种模式的真实值
for k = 1:length(modes_list)
    noise_mode = modes_list{k};
    fprintf('------------------------------------------------\n');
    fprintf('正在处理模式: [%s] ...\n', noise_mode);
    
    % 重置局部种子，确保不同模式下的基础随机性也是可控的
    rng(100 + k); 
    
    switch noise_mode
        case 'Gaussian'
            % 标准高斯分布
            noise = randn(num_buses, num_hours);
            
        case 'Student-T'
            % t分布 (自由度=3), 手动构造无需工具箱
            nu = 3; 
            Z = randn(num_buses, num_hours);
            V = zeros(num_buses, num_hours);
            for i = 1:nu
                V = V + randn(num_buses, num_hours).^2;
            end
            t_raw = Z ./ sqrt(V / nu);
            noise = t_raw / sqrt(nu/(nu-2));
            
        case 'Mixed'
            % 高斯 + 稀疏的大幅跳变 (Outliers)
            noise = randn(num_buses, num_hours);
            spike_mask = rand(num_buses, num_hours) < 0.05; % 5% 概率
            spikes = (rand(num_buses, num_hours) - 0.5) * 10; 
            noise(spike_mask) = noise(spike_mask) + spikes(spike_mask);
    end
    
    % 合成真实值
    d_real = d_forecast + (d_forecast .* base_error_std .* noise);
    d_real(d_real < 0) = 0;
    
    % 保存文件
    filename = sprintf('data_%s_case39.mat', lower(noise_mode));
    save(filename, 'd_forecast', 'd_real', 'mpc', '-v7.3');
    fprintf('>> 已保存: %s\n', filename);
end

fprintf('------------------------------------------------\n');
fprintf('全部完成！生成了3个文件，请在Python中加载使用。\n');