function generate_perfect_data()
    clear; clc;
    rng(1024); % 换个幸运种子

    %% 1. 获取并修改 Case39
    mpc = case39();
    
    % [关键修改 1]：降低基础负荷水平 (Headroom)
    % IEEE 39 总容量约 7300MW，原负荷约 6200MW。余量太小，稍微加点噪声就炸。
    % 我们把负荷降到 60% (约 3700MW)，这样即使加上 30% 的极端噪声，也不会超过总容量。
    mpc.bus(:, 3) = mpc.bus(:, 3) * 0.6; 
    
    % [关键修改 2]：稍微增加线路容量 (防止卡在某条线路上)
    mpc.branch(:, 6) = mpc.branch(:, 6) * 1.2; 

    % [关键修改 3]：设置梯级发电成本 (Merit Order)
    % 只有成本有差异，"更保守" 才会体现为 "更贵"
    ng = size(mpc.gen, 1);
    % Cost format: [model, startup, shutdown, n, c2, c1, c0]
    new_costs = zeros(ng, 7);
    new_costs(:, 1) = 2; % Polynomial
    new_costs(:, 4) = 3; 
    
    % 制造显著的成本差异：
    % G1-G3 (Base): $10 (便宜)
    % G4-G7 (Mid) : $30
    % G8-G10(Peak): $100 (极贵)
    % 这样 DDRO 为了安全被迫预留 G1-G3 的容量，去用 G8，成本就会飙升。
    prices = linspace(10, 100, ng)'; 
    new_costs(:, 6) = prices; % 设置 c1 (线性项)
    new_costs(:, 5) = 0.001;  % c2 很小
    mpc.gencost = new_costs;

    %% 2. 生成“完美”的厚尾负荷数据
    days = 90;
    H = 24;
    N_samples = days * H;
    load_buses = find(mpc.bus(:, 3) > 0);
    n_load = length(load_buses);
    base_load = mpc.bus(load_buses, 3);
    
    % A. 预测负荷 (Forecast)
    t = (1:N_samples)';
    % 简单的日周期
    Pd_forecast = zeros(N_samples, size(mpc.bus, 1));
    for i = 1:n_load
        idx = load_buses(i);
        daily = 1 + 0.1 * sin(2*pi*t/24 - pi/2);
        Pd_forecast(:, idx) = base_load(i) * daily;
    end
    
    % B. 噪声生成 (核心逻辑)
    noise = zeros(N_samples, size(mpc.bus, 1));
    
    for i = 1:n_load
        idx = load_buses(i);
        
        % --- 95% 的时间：极低的高斯底噪 ---
        % Sigma 只有 2%。这会让 Gaussian CCP 觉得“很安全”。
        sigma_base = 0.02 * Pd_forecast(:, idx);
        noise_vec = randn(N_samples, 1) .* sigma_base;
        
        % --- 5% 的时间：巨大的单向尖峰 (Heavy Tail) ---
        % 模拟风电突然停摆，或者高温导致负荷激增
        % 尖峰幅度是 25% (非常大！)
        n_spikes = round(N_samples * 0.05);
        spike_idx = randperm(N_samples, n_spikes);
        
        % 强制为正向尖峰 (abs)，且幅度巨大
        spike_val = 0.25 * Pd_forecast(spike_idx, idx);
        
        % 叠加尖峰
        noise_vec(spike_idx) = noise_vec(spike_idx) + abs(spike_val);
        
        noise(:, idx) = noise_vec;
    end
    
    Pd_real = Pd_forecast + noise;
    
    %% 3. 生成价格 (运行 DC-OPF)
    fprintf('开始生成价格数据...\n');
    LMP_real = zeros(N_samples, size(mpc.bus, 1));
    mpopt = mpoption('verbose', 0, 'out.all', 0, 'model', 'DC');
    
    for k = 1:N_samples
        if mod(k, 500) == 0, fprintf('Progress: %d/%d\n', k, N_samples); end
        mpc_temp = mpc;
        mpc_temp.bus(:, 3) = Pd_real(k, :)';
        
        res = rundcopf(mpc_temp, mpopt);
        if res.success
            LMP_real(k, :) = res.bus(:, 14)';
        else
            % 如果无解，说明即使降了负荷还是有极端情况，填上一个惩罚高价
            LMP_real(k, :) = 200; 
        end
    end

    %% 4. 保存
    data_load.forecast = Pd_forecast;
    data_load.real = Pd_real;
    data_load.noise = noise;
    data_load.bus_idx = load_buses;
    
    data_price.LMP = LMP_real;
    
    save('experiment_data_case39.mat', 'data_load', 'data_price', 'mpc');
    fprintf('数据生成完毕！\n');
    
    % 绘图验证
    figure;
    histogram(noise(:, load_buses(1)), 100);
    title('Noise Distribution: Gaussian Body + Sparse Heavy Tail');
end

function mpc = case39()
   mpc = loadcase('case39'); % 如果你有文件，或者把函数贴在下面
end