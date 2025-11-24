% =========================================================================
%  脚本 3: gen_v2_learning_data.m
%  功能: 生成 ML 数据集 (v2版本)
% =========================================================================
clear; clc;

% 加载 v2 数据
if ~exist('v2_balanced_load_data.mat', 'file')
    error('请先运行脚本 2！');
end
load('v2_balanced_load_data.mat'); 
load('v2_generator_parameters.mat'); 

num_hours = size(d_forecast, 1);
num_gens = length(c_base);

fprintf('步骤 3 (v2): 生成 ML 特征...\n');

% Y: 动态成本
trend = 1 + cumsum(randn(num_hours, 1) * 0.005); 
trend = normalize(trend, 'range', [0.8, 1.5]);
Y_labels = zeros(num_hours, num_gens);
for i = 1:num_hours
    Y_labels(i, :) = c_base' .* trend(i) .* (1 + randn(1, num_gens)*0.02);
end

% X: 特征
feat_load = sum(d_forecast, 2) / max(sum(d_forecast, 2));
feat_sin = sin(2*pi*(1:num_hours)'/24);
feat_cos = cos(2*pi*(1:num_hours)'/24);
feat_price = trend;

X_features = [feat_load, feat_sin, feat_cos, feat_price];

% [关键] 保存为 v2_learning_dataset.mat
save('v2_learning_dataset.mat', 'X_features', 'Y_labels', 'd_forecast', '-v7.3');
fprintf('  -> 已保存: v2_learning_dataset.mat\n');