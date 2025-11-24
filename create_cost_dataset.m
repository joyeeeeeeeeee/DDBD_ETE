% =========================================================================
%  脚本: 为 case39 生成端到端学习的最终数据集
%  加载已合成的负荷数据，并生成动态成本数据和机器学习特征
% =========================================================================

clear; clc;

%% 1. 配置参数
% --- 输入文件 ---
load_data_file = 'synthetic_data_package.mat';

% --- 输出文件 ---
final_dataset_file = 'end_to_end_learning_dataset.mat';

% --- 生成参数 ---
num_hours = 24 * 30 * 4; % 必须与上一个脚本中的小时数一致

% 模拟燃料价格的参数 (您可以调整这些值)
fuel_price_avg = 3.5;       % 假设的年平均天然气价格 ($/MMBtu)
fuel_price_volatility = 0.5;% 价格的日度波动标准差
fuel_cost_weight = 0.7;     % 燃料成本占总边际成本的比例 (例如 70%)

%% 2. 加载之前生成的负荷数据和电网模型
fprintf('正在加载已生成的负荷数据包...\n');
if ~exist(load_data_file, 'file')
    error('错误: 未找到 %s 文件。请先运行 create_synthetic_data.m 脚本。', load_data_file);
end
load(load_data_file); % 这会加载 mpc, d_forecast, d_real 到工作区
fprintf('数据包加载成功。\n');

%% 3. 提取基础成本向量 c_base
fprintf('正在从 mpc 文件中提取基础成本向量...\n');
define_constants;
% 这部分逻辑与之前的脚本完全相同
if mpc.gencost(1, MODEL) == 2 % 检查是否为多项式
    num_cost_coeffs = mpc.gencost(:, NCOST);
    COST_START_COL = NCOST + 1;
    num_gens = size(mpc.gen, 1);
    c_base = zeros(num_gens, 1);
    for i = 1:num_gens
        n = num_cost_coeffs(i);
        if n >= 2
            b_col_index = COST_START_COL + (n - 2);
            c_base(i) = mpc.gencost(i, b_col_index);
        else
            c_base(i) = 0;
        end
    end
    fprintf('基础成本向量 c_base 提取成功。\n');
else
    error('错误: case39 文件中的成本模型不是多项式 (MODEL ~= 2)。');
end

%% 4. 生成动态的“真实”成本向量 c_t (作为机器学习的标签 Labels)
fprintf('正在生成动态的真实成本序列 c_t...\n');

% a) 生成一个模拟的日度燃料价格序列 (随机游走模型)
num_days = num_hours / 24;
daily_price_shocks = randn(num_days, 1) * fuel_price_volatility;
fuel_price_daily = fuel_price_avg + cumsum(daily_price_shocks);
fuel_price_daily(fuel_price_daily < 0.5) = 0.5; % 确保价格不为负

% b) 将日度价格扩展为小时价格
fuel_price_hourly = repelem(fuel_price_daily, 24);

% c) 创建一个价格调整因子序列 (相对于平均价格)
price_adjustment_factors = 1 + fuel_cost_weight * (fuel_price_hourly / fuel_price_avg - 1);

% d) 将调整因子应用到基础成本上，生成最终的成本时序矩阵
% [10x1] 的基础成本向量 * [1x2880] 的调整因子行向量
% 结果是一个 [10x2880] 的动态成本矩阵
c_real_t = c_base * price_adjustment_factors';
fprintf('已成功生成动态成本矩阵 c_real_t，尺寸为 %d x %d。\n', size(c_real_t,1), size(c_real_t,2));


%% 5. 构建机器学习特征矩阵 x_t (Features)
fprintf('正在构建机器学习特征矩阵 x_t...\n');

% a) 获取系统总负荷预测
d_forecast_total = sum(d_forecast, 1)'; % 转置为列向量

% b) 创建时间特征
time_vector = (1:num_hours)';
hour_of_day_sin = sin(2 * pi * time_vector / 24);
hour_of_day_cos = cos(2 * pi * time_vector / 24);
day_of_week = weekday(datetime('now') + hours(time_vector));
is_weekend = (day_of_week == 1 | day_of_week == 7);

% c) 整合所有特征
% 我们在这里创建一个 table，这在处理带名字的特征时更方便
feature_table = table( ...
    hour_of_day_sin, ...
    hour_of_day_cos, ...
    double(is_weekend), ... % 将 logical 转为 double
    d_forecast_total, ...
    fuel_price_hourly ... % 将“当天”的燃料价格作为特征
);

% 将 table 转换为纯矩阵，用于 Python/PyTorch
X_features = table2array(feature_table);
% 目标/标签是成本矩阵 c_real_t。注意需要转置以匹配特征矩阵的行
Y_labels = c_real_t'; 

fprintf('已成功构建特征矩阵 X (尺寸 %d x %d) 和标签矩阵 Y (尺寸 %d x %d)。\n', ...
    size(X_features,1), size(X_features,2), size(Y_labels,1), size(Y_labels,2));

%% 6. 保存最终的数据集
fprintf('正在保存最终的数据集...\n');
save(final_dataset_file, 'X_features', 'Y_labels', 'mpc', 'd_forecast','-v7.3');


fprintf('\n*** 所有数据生成工作已全部完成！ ***\n');
fprintf('最终的数据集已保存到: %s\n', final_dataset_file);
fprintf('您现在可以将这个 .mat 文件加载到 Python 中进行端到端学习。\n');
fprintf('文件中包含:\n');
fprintf('  - X_features: 机器学习模型的输入特征 (行:时间点, 列:特征)\n');
fprintf('  - Y_labels:   机器学习模型的目标标签 (行:时间点, 列:发电机成本)\n');
fprintf('  - mpc:        完整的 MATPOWER case39 结构体\n');
fprintf('  - d_forecast: 预测负荷矩阵 (用于在优化问题中作为确定性参数)\n');