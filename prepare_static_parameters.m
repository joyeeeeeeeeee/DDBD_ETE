% =========================================================================
%  脚本: 提取 case39 的静态参数 (PTDF, 成本系数等)
%  功能: 从原始的 case39.m 文件中提取电网拓扑和发电机参数，
%        并以 Python 兼容的 -v7.3 格式保存。
%  生成文件:
%    - network_topology.mat
%    - generator_parameters.mat
% =========================================================================

clear;
clc;

%% 1. 配置参数
case_name = 'case39';

%% 2. 加载案例并提取基础信息
fprintf('步骤 1: 加载 %s...\n', case_name);
try
    mpc = loadcase(case_name);
    disp([case_name, ' 加载成功。']);
catch ME
    error('加载 %s 失败。请确保 MATPOWER 已正确安装并添加到 MATLAB 路径中。错误信息: %s', case_name, ME.message);
end

% 加载MATPOWER的常量
define_constants;

%% 3. 提取拓扑信息并计算 PTDF 矩阵
fprintf('步骤 2: 计算 PTDF 矩阵...\n');

% 自动找到参考母线 (slack bus)
ref_bus = find(mpc.bus(:, BUS_TYPE) == REF, 1);
if isempty(ref_bus)
    ref_bus = 1; % 如果没有定义，默认选择第一个母线
end

% 使用MATPOWER内置函数计算PTDF矩阵
PTDF = makePTDF(mpc.baseMVA, mpc.bus, mpc.branch, ref_bus);

% 以 v7.3 格式保存结果
save('network_topology.mat', 'PTDF', 'mpc', '-v7.3');
disp('完成！电网拓扑信息 (PTDF) 已保存到 network_topology.mat (v7.3 格式)');
fprintf('\n');

%% 4. 提取发电机参数
fprintf('步骤 3: 提取发电机参数...\n');

gen_data = mpc.gen;
gencost_data = mpc.gencost;

% a. 保存发电机连接的母线号
gen_bus_indices = gen_data(:, GEN_BUS);

% b. 保存发电机的有功出力上下限 (单位: MW)
Pmin = gen_data(:, PMIN);
Pmax = gen_data(:, PMAX);

% c. 提取基础成本向量 c_base
disp('正在解析发电机成本函数...');

cost_model_type = gencost_data(1, MODEL);
if cost_model_type == 2 % 确认是多项式成本 (MODEL = 2)
    disp('检测到成本模型为多项式 (Polynomial)，正在提取线性系数...');
    
    num_cost_coeffs = gencost_data(:, NCOST);
    COST_START_COL = NCOST + 1;
    num_gens = size(gen_data, 1);
    c_base = zeros(num_gens, 1);

    for i = 1:num_gens
        n = num_cost_coeffs(i);
        if n >= 2 % 至少有 b*P + d
            % 线性项系数 c1 (即 b) 是倒数第二个成本参数
            b_col_index = COST_START_COL + (n - 2);
            c_base(i) = gencost_data(i, b_col_index);
        else
            c_base(i) = 0;
        end
    end
    disp('基础成本向量 c_base 已从多项式成本中成功提取。');
else
    error('错误: %s 文件中的成本模型不是多项式 (MODEL ~= 2)。请使用标准 case39 文件。', case_name);
end

% d. 以 v7.3 格式保存所有发电机参数
save('generator_parameters.mat', 'gen_bus_indices', 'Pmin', 'Pmax', 'c_base', '-v7.3');
disp('完成！发电机参数已保存到 generator_parameters.mat (v7.3 格式)');
fprintf('\n');

disp('所有静态参数提取和预处理已完成！');