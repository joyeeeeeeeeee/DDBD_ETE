% =========================================================================
%  脚本: 3_create_learning_dataset.m
%  功能: 生成端到端学习所需的 X (特征) 和 Y (真实成本系数)
% =========================================================================
clear; clc;

load('synthetic_data_package.mat'); % 载入 d_forecast, d_real
load('generator_parameters.mat');   % 载入 c_base

num_hours = size(d_forecast, 1);
num_gens = length(c_base);

% --- 1. 生成 Y_labels (动态的真实燃料价格) ---
% 假设基础成本 c_base 随时间波动 (模拟天然气价格波动)
fuel_price_trend = 1 + cumsum(randn(num_hours, 1) * 0.01); 
fuel_price_trend = normalize(fuel_price_trend, 'range', [0.8, 1.2]);

% Y_labels: (Time, Gens). 每一列是一个发电机的实时成本
Y_labels = zeros(num_hours, num_gens);
for t = 1:num_hours
    % 加上一些随机高斯噪声，模拟不同发电机成本的独立波动
    daily_noise = 1 + randn(num_gens, 1) * 0.05;
    Y_labels(t, :) = c_base' .* fuel_price_trend(t) .* daily_noise';
end

% --- 2. 生成 X_features (机器学习特征) ---
% 特征 1: 总负荷预测
feat_load = sum(d_forecast, 2);
% 特征 2: 小时正弦
feat_sin = sin(2 * pi * (1:num_hours)' / 24);
% 特征 3: 小时余弦
feat_cos = cos(2 * pi * (1:num_hours)' / 24);
% 特征 4: 昨天同期价格 (简单的自回归特征)
feat_price_lag = [mean(c_base); mean(Y_labels(1:end-1, :), 2)]; 
% 特征 5: 燃料价格指数 (假设是已知市场信息)
feat_fuel_idx = fuel_price_trend;

X_features = [feat_load, feat_sin, feat_cos, feat_price_lag, feat_fuel_idx];

% --- 保存 ---
save('end_to_end_learning_dataset.mat', 'X_features', 'Y_labels', 'd_forecast', '-v7.3');

fprintf('机器学习数据集已生成:\n');
fprintf('  X_features: %d x %d\n', size(X_features,1), size(X_features,2));
fprintf('  Y_labels:   %d x %d\n', size(Y_labels,1), size(Y_labels,2));