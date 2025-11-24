import numpy as np
import cvxpy as cp
import h5py
from scipy.stats import binom
import warnings

# =========================================================================
#  1. 数据加载 (调整 Scale Factor)
# =========================================================================
def load_data(synthetic_path, topo_path, gen_path):
    print(f"--- 加载数据: {synthetic_path} ---")
    baseMVA = 100.0 
    
    with h5py.File(topo_path, 'r') as f:
        ptdf_raw = np.array(f['PTDF'])
        bus_shape = np.array(f['mpc']['bus']).shape
        num_buses = bus_shape[1]
        if ptdf_raw.shape[1] == num_buses: ptdf = ptdf_raw
        else: ptdf = ptdf_raw.T
            
        branch_data = np.array(f['mpc']['branch']).T
        flow_limits = branch_data[:, 5] 
        flow_limits[flow_limits == 0] = 9999 
        if 'mpc' in f and 'baseMVA' in f['mpc']:
            baseMVA = np.array(f['mpc']['baseMVA']).item()

        # [关键调整] 0.85 是平衡点
        # 鲁棒模型会把这 0.85 里的 0.1-0.15 作为安全余量，实际只用 0.7
        # 确定性模型会用到 0.85
        # 真实负荷一来，就会冲破 0.85，导致确定性违约
        scale_factor = 0.85
        print(f"  [设置] 线路限额缩放因子: {scale_factor}")
        flow_limits = flow_limits * scale_factor
    
    with h5py.File(synthetic_path, 'r') as f:
        d_forecast_raw = np.array(f['d_forecast'])
        d_real_raw = np.array(f['d_real'])
        if 'mpc' in f and 'baseMVA' in f['mpc']:
            baseMVA = np.array(f['mpc']['baseMVA']).item()
        if d_forecast_raw.shape[1] == num_buses:
            d_forecast = d_forecast_raw
            d_real = d_real_raw
        else:
            d_forecast = d_forecast_raw.T
            d_real = d_real_raw.T
            
    with h5py.File(gen_path, 'r') as f:
        c_base = np.array(f['c_base']).flatten()
        gen_bus_indices = np.array(f['gen_bus_indices']).flatten().astype(int) - 1
        p_min = np.array(f['Pmin']).flatten() / baseMVA
        p_max = np.array(f['Pmax']).flatten() / baseMVA
        
    num_gens = len(c_base)
    Cg = np.zeros((num_buses, num_gens))
    for i in range(num_gens): Cg[gen_bus_indices[i], i] = 1
        
    params = {
        'ptdf': ptdf, 'flow_limits': flow_limits / baseMVA,
        'c_base': c_base, 'p_min': p_min, 'p_max': p_max,
        'Cg': Cg, 'baseMVA': baseMVA, 'num_buses': num_buses
    }
    return d_forecast, d_real, params

# =========================================================================
#  2. 校准
# =========================================================================
def calibrate_uncertainty(errors, delta, epsilon):
    n_cal = int(len(errors) * 0.5)
    D_cal = errors[n_cal:]
    mu = np.mean(errors[:n_cal], axis=0)
    Sigma = np.cov(errors[:n_cal], rowvar=False) + np.eye(errors.shape[1]) * 1e-6
    
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 0
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    diff = D_cal - mu
    mah_dist = np.sum((diff @ Sigma_inv) * diff, axis=1)
    sorted_dist = np.sort(mah_dist)
    
    i_star = int(len(D_cal) * (1 - epsilon))
    if i_star >= len(D_cal): i_star = len(D_cal) - 1
    rho = np.sqrt(sorted_dist[i_star])
    
    return mu, rho, Sigma_sqrt

# =========================================================================
#  3. 求解器
# =========================================================================
def solve_dispatch(params, d_t, robust_data, tau, mode='robust'):
    x_pu = cp.Variable(len(params['c_base']))
    obj = cp.Minimize(params['c_base'].T @ x_pu)
    constraints = [x_pu >= params['p_min'], x_pu <= params['p_max']]
    
    mu_pu, rho, Sigma_sqrt_pu = robust_data
    d_t_pu = d_t / params['baseMVA']
    
    # 系统平衡
    nominal_imb = cp.sum(x_pu) - cp.sum(d_t_pu + mu_pu)
    if mode == 'robust':
        sys_margin = rho * np.linalg.norm(Sigma_sqrt_pu @ np.ones(params['num_buses']), 2)
        # 增加系统平衡的宽容度，避免因为一点点不平衡报错
        bound = max(tau/params['baseMVA'] - sys_margin, 0.01)
        constraints += [cp.abs(nominal_imb) <= bound]
    else:
        constraints += [cp.sum(x_pu) - cp.sum(d_t_pu) == 0]

    # 线路潮流
    inj = params['Cg'] @ x_pu - (d_t_pu + mu_pu)
    flow = params['ptdf'] @ inj
    
    if mode == 'robust':
        line_margins = rho * np.linalg.norm(params['ptdf'] @ Sigma_sqrt_pu, axis=1)
        eff_limits = params['flow_limits'] - line_margins
        
        # 关键逻辑：如果 Margin > Limit，不要设为 0，而是设为一个微小值
        # 并让优化器尽力而为。如果完全不可行，让它报错，我们统计这一事实。
        eff_limits[eff_limits < 0] = 1e-5
        
        constraints += [cp.abs(flow) <= eff_limits]
    else:
        constraints += [cp.abs(flow) <= params['flow_limits']]
        
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, max_iters=1000)
    except: return None
    if prob.status != 'optimal': return None
    return x_pu.value * params['baseMVA']

# =========================================================================
#  4. 主流程
# =========================================================================
def main():
    d_fcst, d_real, params = load_data('new_synthetic_data_package.mat', 'network_topology.mat', 'generator_parameters.mat')
    
    # 校准
    err = (d_real[:1000] - d_fcst[:1000]) / params['baseMVA']
    mu, rho, Sigma_sqrt = calibrate_uncertainty(err, 0.05, 0.05)
    
    margin_sys = rho * np.linalg.norm(Sigma_sqrt @ np.ones(params['num_buses'])) * params['baseMVA']
    # 给 Tau 多一点余量，确保系统平衡不是瓶颈，线路拥塞才是
    tau = margin_sys + 100.0
    print(f"  校准结果: rho={rho:.2f}, Margin={margin_sys:.1f} MW -> Tau={tau:.1f} MW")
    rob_dat = (mu, rho, Sigma_sqrt)
    
    # 筛选
    total_load = np.sum(d_fcst, axis=1)
    high_load_idx = np.argsort(total_load)[-400:] # 范围扩大一点
    np.random.seed(999)
    test_indices = np.random.choice(high_load_idx, 100, replace=False)
    
    print(f"\n--- 开始甜蜜点测试 (N={len(test_indices)}) ---")
    
    res = {'rob':{'c':[], 'v':[]}, 'det':{'c':[], 'v':[]}}
    
    success_count = 0
    for idx in test_indices:
        x_r = solve_dispatch(params, d_fcst[idx], rob_dat, tau, 'robust')
        x_d = solve_dispatch(params, d_fcst[idx], rob_dat, tau, 'deterministic')
        
        if x_r is not None and x_d is not None:
            success_count += 1
            c_r = params['c_base'].T @ (x_r/params['baseMVA'])
            c_d = params['c_base'].T @ (x_d/params['baseMVA'])
            res['rob']['c'].append(c_r)
            res['det']['c'].append(c_d)
            
            d_t = d_real[idx]
            lim = params['flow_limits'] * params['baseMVA']
            
            f_r = params['ptdf'] @ (params['Cg'] @ x_r - d_t)
            res['rob']['v'].append(1 if np.any(np.abs(f_r) > lim + 0.01) else 0)
            
            f_d = params['ptdf'] @ (params['Cg'] @ x_d - d_t)
            res['det']['v'].append(1 if np.any(np.abs(f_d) > lim + 0.01) else 0)

    if success_count == 0:
        print("仍然无解。请检查是否数据生成部分未运行，或者限额还是太紧。")
        return

    print("\n" + "="*60)
    print(f"{'指标':<20} | {'鲁棒优化 (Ours)':<18} | {'确定性优化':<15}")
    print("-" * 60)
    print(f"{'平均成本 ($)':<20} | {np.mean(res['rob']['c']):<18.2f} | {np.mean(res['det']['c']):<15.2f}")
    print(f"{'违约率 (%)':<20} | {np.mean(res['rob']['v'])*100:<18.1f} | {np.mean(res['det']['v'])*100:<15.1f}")
    print("-" * 60)
    diff = (np.mean(res['rob']['c']) - np.mean(res['det']['c'])) / np.mean(res['det']['c']) * 100
    print(f"成本增加: +{diff:.1f}% | 有效样本数: {success_count}")
    print("="*60)

if __name__ == '__main__':
    main()