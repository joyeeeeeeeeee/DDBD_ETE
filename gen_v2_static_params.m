% =========================================================================
%  脚本 1: gen_v2_static_params.m
%  功能: 提取 Case39 参数 (v2版本 - 修复维度和文件名)
% =========================================================================
clear; clc;

case_name = 'case39';
fprintf('步骤 1 (v2): 加载 %s ...\n', case_name);
try
    mpc = loadcase(case_name);
catch
    error('请检查 MATPOWER 是否安装。');
end
define_constants;

%% 1. 计算 PTDF
ref_bus = find(mpc.bus(:, BUS_TYPE) == REF, 1);
if isempty(ref_bus), ref_bus = 1; end

fprintf('  -> 计算 PTDF ...\n');
PTDF = makePTDF(mpc.baseMVA, mpc.bus, mpc.branch, ref_bus);

% [关键] 保存为 v2_network_topology.mat
save('v2_network_topology.mat', 'PTDF', 'mpc', '-v7.3');
fprintf('  -> 已保存: v2_network_topology.mat\n');

%% 2. 提取发电机参数
gen_data = mpc.gen;
gencost = mpc.gencost;

gen_bus_indices = gen_data(:, GEN_BUS);
Pmin = gen_data(:, PMIN);
Pmax = gen_data(:, PMAX);

% 提取成本 c1
num_gens = size(gen_data, 1);
c_base = zeros(num_gens, 1);
if gencost(1, MODEL) == 2
    for i = 1:num_gens
        n = gencost(i, NCOST);
        if n >= 2
            c_base(i) = gencost(i, NCOST+1 + n - 2);
        end
    end
end

% [关键] 保存为 v2_generator_parameters.mat
save('v2_generator_parameters.mat', 'gen_bus_indices', 'Pmin', 'Pmax', 'c_base', '-v7.3');
fprintf('  -> 已保存: v2_generator_parameters.mat\n');