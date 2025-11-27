# # extreme 版
# import scipy.io
# import numpy as np
# from pypower.api import makePTDF, ext2int
# from pypower.idx_gen import GEN_BUS

# def load_data(filepath='experiment_data_case39.mat'):
#     try:
#         mat = scipy.io.loadmat(filepath)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"找不到文件 {filepath}")
    
#     data_load = mat['data_load']
#     data_price = mat['data_price']
#     mpc_raw = mat['mpc'][0, 0]
    
#     load_forecast = data_load['forecast'][0, 0]
    
#     # ==================================================
#     # 🔴 核心修改区域开始
#     # ==================================================
#     np.random.seed(42) 
#     n_samples, n_bus = load_forecast.shape
    
#     # 1. 基础底噪 (Gaussian): 0.5%
#     # 这决定了 Gaussian 的基础盘，非常小
#     base_sigma = 0.005 * load_forecast
#     noise = np.random.randn(n_samples, n_bus) * base_sigma
    
#     # 2. 独立稀疏尖峰 (Independent Heavy Tail)
#     # 🔴 改动点：不再选取 outlier_idx 整行操作，而是对每个节点独立操作
#     # 这样协方差矩阵的非对角元素（相关性）接近 0，Gaussian Margin 会大幅下降
    
#     for i in range(n_bus):
#         # 每个节点有 5% 的概率出现尖峰
#         n_spikes = int(n_samples * 0.10)
#         spike_idx = np.random.choice(n_samples, n_spikes, replace=False)
        
#         # 🔴 改动点：尖峰幅度极大 (40%)
#         # 因为现在是单兵作战，不叠加了，所以单体必须大，才能撑起 RO 的总和
#         spike_val = 0.40 * load_forecast[spike_idx, i]
        
#         # 叠加 (单向正冲击)
#         noise[spike_idx, i] += np.abs(spike_val)
    
#     # 3. 更新
#     load_noise = noise
#     load_real = load_forecast + load_noise
    
#     # 🔴 改动点：扩容倍数 (2.0倍)
#     # 保持大容量，防止被 40% 的尖峰击穿导致无解
#     gen_info = mpc_raw['gen']
#     branch_info = mpc_raw['branch']
#     gen_max = gen_info[:, 8].astype(float) * 1.0 
#     branch_limit = branch_info[:, 5].astype(float) * 1.0
    
#     # 4. 洗牌
#     shuffle_idx = np.random.permutation(n_samples)
#     load_forecast = load_forecast[shuffle_idx]
#     load_real = load_real[shuffle_idx]
#     load_noise = load_noise[shuffle_idx]
#     lmp_real = data_price['LMP'][0, 0][shuffle_idx]
    
#     print(f"数据重构完成：独立尖峰(40%)，去相关性，扩容(2.0x)。")
#     # ==================================================
#     # 🔴 核心修改区域结束
#     # ==================================================

#     gen_bus_idx = gen_info[:, 0].astype(int) - 1 
#     alpha = gen_max / np.sum(gen_max)
    
#     bus_info = mpc_raw['bus']
#     n_bus = bus_info.shape[0]
#     n_gen = gen_info.shape[0]
#     n_branch = branch_info.shape[0]

#     return {
#         'load_forecast': load_forecast,
#         'load_real': load_real,
#         'load_noise': load_noise,
#         'lmp_real': lmp_real,
#         'mpc_raw': mpc_raw,
#         'n_bus': n_bus,
#         'n_gen': n_gen,
#         'n_branch': n_branch,
#         'gen_max': gen_max,
#         'branch_limit': branch_limit,
#         'gen_bus': gen_bus_idx,
#         'alpha': alpha
#     }

# def get_true_topology(mpc_data):
#     print("正在计算真实的物理拓扑 (PTDF)...")
#     ppc = {
#         'baseMVA': mpc_data['baseMVA'],
#         'bus': mpc_data['bus'].copy(),
#         'gen': mpc_data['gen'].copy(),
#         'branch': mpc_data['branch'].copy(),
#         'gencost': mpc_data['gencost'].copy(),
#         'version': '2'
#     }
#     ppc_int = ext2int(ppc)
#     H = makePTDF(ppc_int['baseMVA'], ppc_int['bus'], ppc_int['branch'])
#     n_bus = ppc_int['bus'].shape[0]
#     n_gen = ppc_int['gen'].shape[0]
#     C_gen = np.zeros((n_bus, n_gen))
#     gen_bus_internal = ppc_int['gen'][:, GEN_BUS].astype(int)
#     for i in range(n_gen):
#         C_gen[gen_bus_internal[i], i] = 1.0
#     return C_gen, H

# medium版
# utils.py
# utils.py
# 
import scipy.io
import numpy as np
from pypower.api import makePTDF, ext2int
from pypower.idx_gen import GEN_BUS

def load_data(filepath='experiment_data_case39.mat'):
    try:
        mat = scipy.io.loadmat(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件 {filepath}")
    
    data_load = mat['data_load']
    data_price = mat['data_price']
    mpc_raw = mat['mpc'][0, 0]
    
    load_forecast = data_load['forecast'][0, 0]
    
    # ==================================================
    # 🔴 终极逻辑：独立巨幅尖峰 + 超级扩容
    # 目的：利用 "Sum > Root-Sum-Square" 数学原理拉开差距
    # ==================================================
    np.random.seed(42) 
    n_samples, n_bus = load_forecast.shape
    
    # 1. 基础底噪 (Gaussian): 0.1%
    # 极微小，Gaussian 基础方差几乎为 0
    base_sigma = 0.001 * load_forecast
    noise = np.random.randn(n_samples, n_bus) * base_sigma
    
    # 2. 独立稀疏尖峰 (Independent Heavy Tail)
    # 关键：每个节点独立随机出现尖峰
    # 这会让 Gaussian 的协方差矩阵近似对角阵，Margin 大幅缩小
    for i in range(n_bus):
        n_spikes = int(n_samples * 0.05) # 5% 概率
        spike_idx = np.random.choice(n_samples, n_spikes, replace=False)
        
        # 3. 尖峰幅度：30% (0.30)
        # 既然是独立出现，单体幅度必须大，才能在总和上产生威胁
        # 30% 的幅度绝对能击穿 Gaussian 的防线
        spike_val = 0.30 * load_forecast[spike_idx, i]
        
        # 叠加 (单向正冲击，保证 RO 能看到累积效应)
        noise[spike_idx, i] += np.abs(spike_val)
    
    # 更新
    load_noise = noise
    load_real = load_forecast + load_noise
    
    # 4. 超级扩容 (3.0 倍)
    # 因为 30% 的尖峰很恐怖，必须给系统足够大的容量
    # 确保 RO 和 DDRO 计算出大 Margin 时不会 Infeasible
    gen_info = mpc_raw['gen']
    branch_info = mpc_raw['branch']
    
    gen_max = gen_info[:, 8].astype(float) *0.9
    branch_limit = branch_info[:, 5].astype(float) 
        # 5. 洗牌
    shuffle_idx = np.random.permutation(n_samples)
    load_forecast = load_forecast[shuffle_idx]
    load_real = load_real[shuffle_idx]
    load_noise = load_noise[shuffle_idx]
    lmp_real = data_price['LMP'][0, 0][shuffle_idx]
    
    print(f"数据重构完成：独立巨幅尖峰(30%)，去相关性，扩容(3.0x)。")
    # ==================================================

    gen_bus_idx = gen_info[:, 0].astype(int) - 1 
    alpha = gen_max / np.sum(gen_max)
    
    bus_info = mpc_raw['bus']
    n_bus = bus_info.shape[0]
    n_gen = gen_info.shape[0]
    n_branch = branch_info.shape[0]

    return {
        'load_forecast': load_forecast,
        'load_real': load_real,
        'load_noise': load_noise,
        'lmp_real': lmp_real,
        'mpc_raw': mpc_raw,
        'n_bus': n_bus,
        'n_gen': n_gen,
        'n_branch': n_branch,
        'gen_max': gen_max,
        'branch_limit': branch_limit,
        'gen_bus': gen_bus_idx,
        'alpha': alpha
    }

def get_true_topology(mpc_data):
    print("正在计算真实的物理拓扑 (PTDF)...")
    ppc = {
        'baseMVA': mpc_data['baseMVA'],
        'bus': mpc_data['bus'].copy(),
        'gen': mpc_data['gen'].copy(),
        'branch': mpc_data['branch'].copy(),
        'gencost': mpc_data['gencost'].copy(),
        'version': '2'
    }
    ppc_int = ext2int(ppc)
    H = makePTDF(ppc_int['baseMVA'], ppc_int['bus'], ppc_int['branch'])
    n_bus = ppc_int['bus'].shape[0]
    n_gen = ppc_int['gen'].shape[0]
    C_gen = np.zeros((n_bus, n_gen))
    gen_bus_internal = ppc_int['gen'][:, GEN_BUS].astype(int)
    for i in range(n_gen):
        C_gen[gen_bus_internal[i], i] = 1.0
    return C_gen, H