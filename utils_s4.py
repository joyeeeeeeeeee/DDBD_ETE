# utils_sec4.py
import numpy as np
import torch
from scipy.linalg import sqrtm
from utils import load_data, get_true_topology
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def get_generator_lmps(mpc, full_lmps):
    """
    将全网节点的 LMP 映射到发电机节点
    """
    gen_bus_idx = mpc['gen'][:, 0].astype(int) - 1 # Python index 0-based
    # full_lmps: [N_samples, N_bus]
    # return: [N_samples, N_gen]
    return full_lmps[:, gen_bus_idx]

def prepare_section4_data(load_scale=0.9):
    """
    数据预处理工厂
    load_scale: 缩放因子，让负荷稍微小一点，保证 Optimization Layer 始终有解
    """
    print(f"\n[Data Prep] 正在准备 Section 4 数据 (Load Scale = {load_scale})...")
    
    # 1. 加载原始数据
    data = load_data()
    # 缩小负荷以确保可行性 (Focus on Cost Optimization)
    data['load_forecast'] *= load_scale
    data['load_real'] *= load_scale
    
    C_gen, H = get_true_topology(data['mpc_raw'])
    
    # 2. 计算固定的物理约束参数 (Using Section 3 Logic)
    # 我们使用全部训练数据算一个“极其安全”的边界，防止训练中断
    train_noise = data['load_noise'][:int(len(data['load_noise'])*0.5)]
    
    # 简化的参数计算 (复用 Section 3 的逻辑)
    mu = np.mean(train_noise, axis=0)
    Sigma = np.cov(train_noise, rowvar=False) + 1e-6*np.eye(train_noise.shape[1])
    Sigma_sqrt = np.real(sqrtm(Sigma))
    
    # 设定一个保守的 Rho (比如 99% 可靠性)
    from scipy.stats import chi2
    rho = chi2.ppf(0.99, df=train_noise.shape[1])
    
    # 计算 Margins
    vec_ones = np.ones(train_noise.shape[1])
    omega_margin = np.sqrt(rho) * np.linalg.norm(Sigma_sqrt @ vec_ones)
    
    vec_inj = C_gen @ data['alpha']
    n_bus = H.shape[1]
    S = np.outer(vec_inj, np.ones(n_bus)) - np.eye(n_bus)
    W = H @ S
    line_margin = np.sqrt(rho) * np.linalg.norm(W @ Sigma_sqrt, axis=1)
    line_shift = W @ mu
    total_mu = np.sum(mu)
    
    robust_params = {
        'omega_margin': omega_margin,
        'line_margin': line_margin,
        'line_shift': line_shift,
        'total_mu': total_mu,
        'gen_max': data['gen_max'],
        'branch_limit': data['branch_limit'],
        'C_gen': C_gen,
        'H': H,
        'alpha': data['alpha']
    }
    
    print(f"  > 物理约束已锁定 (Rho={rho:.1f})")
    
    # 3. 构建机器学习数据集 (X: Features, Y: Generator LMPs)
    # Feature: 过去 24 小时的 [Load_Forecast_Total, Average_LMP]
    # Target: 当前时刻的 Generator LMPs
    
    gen_lmps = get_generator_lmps(data['mpc_raw'], data['lmp_real'])
    total_load_forecast = np.sum(data['load_forecast'], axis=1)
    
    # 归一化 (Normalization)
    feat_mean = np.mean(total_load_forecast)
    feat_std = np.std(total_load_forecast)
    price_mean = np.mean(gen_lmps)
    price_std = np.std(gen_lmps)
    
    X_list = []
    Y_list = []
    D_list = [] # 当天的负荷预测 (作为 Optimization 的参数)
    
    window = 24
    limit = 2000 # 限制样本数，太长训练太慢
    
    for t in range(window, min(len(total_load_forecast), limit)):
        # 特征：过去24小时的总负荷 + 过去24小时的发电机平均电价
        past_load = (total_load_forecast[t-window:t] - feat_mean) / feat_std
        past_price = (np.mean(gen_lmps[t-window:t, :], axis=1) - price_mean) / price_std
        
        feature = np.concatenate([past_load, past_price])
        target = gen_lmps[t, :] # 真实价格 (不归一化，直接算钱)
        
        X_list.append(feature)
        Y_list.append(target)
        D_list.append(data['load_forecast'][t])
        
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    # D 必须保留 numpy 给 cvxpy 用，或者 tensor 给 cvxpylayer 用
    D = torch.tensor(np.array(D_list), dtype=torch.float32)
    
    return X, Y, D, robust_params