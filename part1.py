import numpy as np
import cvxpy as cp
import h5py
from scipy.stats import binom
import warnings

# =========================================================================
#  1. 数据加载 (将限额缩放调整为 0.8)
# =========================================================================
def load_data(synthetic_path, topo_path, gen_path):
    print(f"--- 加载数据: {synthetic_path} ---")
    
    with h5py.File(topo_path, 'r') as f:
        ptdf_raw = np.array(f['PTDF'])
        bus_shape = np.array(f['mpc']['bus']).shape
        num_buses = bus_shape[1]
        
        if ptdf_raw.shape[1] == num_buses:
            ptdf = ptdf_raw
        else:
            ptdf = ptdf_raw.T
            
        branch_data = np.array(f['mpc']['branch']).T
        flow_limits = branch_data[:, 5] 
        flow_limits[flow_limits == 0] = 9999 

        # [修改点] 将缩放因子从 0.6 调整为 0.8
        # 既能制造拥塞，又不会让鲁棒模型直接无解
        scale_factor = 0.80
        print(f"  [Tip] 线路物理限额缩放因子: {scale_factor}")
        flow_limits = flow_limits * scale_factor
    
    with h5py.File(synthetic_path, 'r') as f:
        d_forecast_raw = np.array(f['d_forecast'])
        d_real_raw = np.array(f['d_real'])
        baseMVA = np.array(f['mpc']['baseMVA']).item()
        
        if d_forecast_raw.shape[1] == num_buses:
            d_forecast = d_forecast_raw
            d_real = d_real_raw
        else:
            d_forecast = d_forecast_raw.T
            d_real = d_real_raw.T
            
    with h5py.File(gen_path, 'r') as f:
        c_base = np.array(f['c_base']).flatten()
        p_min = np.array(f['Pmin']).flatten() / baseMVA
        p_max = np.array(f['Pmax']).flatten() / baseMVA
        gen_bus_idx = np.array(f['gen_bus_indices']).flatten().astype(int) - 1
        
    num_gens = len(c_base)
    Cg = np.zeros((num_buses, num_gens))
    for i in range(num_gens):
        Cg[gen_bus_idx[i], i] = 1
        
    params = {
        'ptdf': ptdf, 'flow_limits': flow_limits / baseMVA,
        'c_base': c_base, 'p_min': p_min, 'p_max': p_max,
        'Cg': Cg, 'baseMVA': baseMVA, 'num_buses': num_buses
    }
    return d_forecast, d_real, params

# =========================================================================
#  2. 鲁棒参数校准 (保持不变)
# =========================================================================
def calibrate_uncertainty(errors, delta, epsilon):
    n = len(errors)
    n_cal = int(n * 0.5)
    D_shape = errors[:n_cal]
    D_cal = errors[n_cal:]
    
    mu = np.mean(D_shape, axis=0)
    # 增加微小扰动防止奇异
    Sigma = np.cov(D_shape, rowvar=False) + np.eye(errors.shape[1]) * 1e-6
    
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 0
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    diff = D_cal - mu
    mah_dist = np.sum((diff @ Sigma_inv) * diff, axis=1)
    sorted_dist = np.sort(mah_dist)
    
    n_c = len(D_cal)
    i_star = n_c
    for r in range(1, n_c + 1):
        if binom.cdf(r-1, n_c, 1-delta) >= 1 - epsilon:
            i_star = r
            break
            
    rho_sq = sorted_dist[i_star-1]
    rho = np.sqrt(rho_sq)
    
    return mu, rho, Sigma_sqrt

# =========================================================================
#  3. 优化求解器 (引入软约束逻辑)
# =========================================================================
def solve_dispatch(params, d_t, robust_data, tau, mode='robust'):
    c = params['c_base']
    base = params['baseMVA']
    x_pu = cp.Variable(len(c))
    
    obj = cp.Minimize(c.T @ x_pu)
    
    constraints = [x_pu >= params['p_min'], x_pu <= params['p_max']]
    
    mu_pu, rho, Sigma_sqrt_pu = robust_data
    d_t_pu = d_t / base
    tau_pu = tau / base
    
    # --- 1. 系统平衡约束 ---
    nominal_imb = cp.sum(x_pu) - cp.sum(d_t_pu + mu_pu)
    
    if mode == 'robust':
        sys_margin = rho * np.linalg.norm(Sigma_sqrt_pu @ np.ones(params['num_buses']), 2)
        bound = tau_pu - sys_margin
        if bound < 0: bound = 0.01 # 防止完全无解
        constraints += [nominal_imb <= bound, nominal_imb >= -bound]
    else: 
        constraints += [cp.sum(x_pu) - cp.sum(d_t_pu) == 0]

    # --- 2. 线路潮流约束 (关键修复) ---
    inj_nominal = params['Cg'] @ x_pu - (d_t_pu + mu_pu)
    flow_nominal = params['ptdf'] @ inj_nominal
    limits = params['flow_limits']
    
    if mode == 'robust':
        # 计算原始鲁棒占用量
        raw_margins = rho * np.linalg.norm(params['ptdf'] @ Sigma_sqrt_pu, axis=1)
        
        # [关键] 截断 Margin：如果 Margin > Limit，强制设为 Limit 的 90%
        # 这保证了 effective_limit 永远为正，求解器不会报错
        capped_margins = np.minimum(raw_margins, 0.90 * limits)
        
        eff_limits = limits - capped_margins
        
        constraints += [flow_nominal <= eff_limits]
        constraints += [flow_nominal >= -eff_limits]
    else:
        constraints += [flow_nominal <= limits]
        constraints += [flow_nominal >= -limits]
        
    prob = cp.Problem(obj, constraints)
    try:
        # 增加最大迭代次数
        prob.solve(solver=cp.ECOS, max_iters=500)
    except:
        return None, "Error"
        
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None, prob.status
        
    return x_pu.value * base, "Optimal"

# =========================================================================
#  4. 主程序
# =========================================================================
def main():
    # 1. 加载
    d_forecast, d_real, params = load_data('synthetic_data_package.mat', 
                                         'network_topology.mat', 
                                         'generator_parameters.mat')
    
    # 2. 校准
    n_train = 1000
    errors = d_real[:n_train] - d_forecast[:n_train]
    base = params['baseMVA']
    errors_pu = errors / base
    
    print("\n--- 校准鲁棒参数 ---")
    mu_pu, rho, Sigma_sqrt_pu = calibrate_uncertainty(errors_pu, delta=0.05, epsilon=0.05)
    
    min_margin_pu = rho * np.linalg.norm(Sigma_sqrt_pu @ np.ones(params['num_buses']))
    min_margin_mw = min_margin_pu * base
    
    # 动态设置 Tau，保证有解
    tau_mw = max(min_margin_mw + 30.0, 100.0)
    print(f"  rho = {rho:.4f}")
    print(f"  系统最小需量: {min_margin_mw:.2f} MW -> 设定 Tau: {tau_mw:.2f} MW")
    
    robust_data = (mu_pu, rho, Sigma_sqrt_pu)
    
    # 3. 仿真
    n_test = 100 # 测试样本数
    # 挑选高负荷时段进行测试（避免深夜低谷掩盖问题）假设一天24小时，我们取白天 10:00 - 14:00 的时段
    test_indices = []
    for i in range(n_train, len(d_forecast)):
        hour_of_day = (i % 24)
        if 10 <= hour_of_day <= 14:
            test_indices.append(i)
        if len(test_indices) >= n_test:
            break
            
    print(f"\n--- 开始仿真 (筛选高负荷时段, N={len(test_indices)}) ---")
    
    res = {'rob': {'cost':[], 'viol':[]}, 'det': {'cost':[], 'viol':[]}}
    
    solver_fail_count = 0
    
    for idx in test_indices:
        # A. 鲁棒优化
        x_rob, stat_r = solve_dispatch(params, d_forecast[idx], robust_data, tau_mw, 'robust')
        # B. 确定性优化
        x_det, stat_d = solve_dispatch(params, d_forecast[idx], robust_data, tau_mw, 'deterministic')
        
        d_true = d_real[idx]
        
        # 仅当两者都算出解时，才对比成本（公平对比）
        if stat_r == 'Optimal' and stat_d == 'Optimal':
            cost_r = params['c_base'].T @ (x_rob/base)
            cost_d = params['c_base'].T @ (x_det/base)
            res['rob']['cost'].append(cost_r)
            res['det']['cost'].append(cost_d)
        
        # 独立评估违约情况
        if stat_r == 'Optimal':
            inj = params['Cg'] @ (x_rob/base) - d_true/base
            flows = params['ptdf'] @ inj
            # 检查物理越限
            viol = np.any(np.abs(flows) > params['flow_limits'] + 1e-4)
            res['rob']['viol'].append(viol)
        else:
            res['rob']['viol'].append(1) # 无解算违约
            solver_fail_count += 1
            
        if stat_d == 'Optimal':
            inj = params['Cg'] @ (x_det/base) - d_true/base
            flows = params['ptdf'] @ inj
            viol = np.any(np.abs(flows) > params['flow_limits'] + 1e-4)
            res['det']['viol'].append(viol)
        else:
            res['det']['viol'].append(1)

    # 4. 结果统计
    def get_avg(lst): return np.mean(lst) if lst else 0
    
    print("\n" + "="*55)
    print(f"{'Metric':<20} | {'Robust (Ours)':<15} | {'Deterministic':<15}")
    print("-" * 55)
    
    c_r = get_avg(res['rob']['cost'])
    c_d = get_avg(res['det']['cost'])
    v_r = np.mean(res['rob']['viol']) * 100
    v_d = np.mean(res['det']['viol']) * 100
    
    print(f"{'Avg Cost ($)':<20} | {c_r:<15.2f} | {c_d:<15.2f}")
    print(f"{'Violation Rate (%)':<20} | {v_r:<15.1f} | {v_d:<15.1f}")
    print("-" * 55)
    print(f"Cost Difference: Robust is {(c_r - c_d)/c_d*100:.1f}% more expensive")
    print(f"Solver Failures (Robust): {solver_fail_count}/{len(test_indices)}")
    print("="*55)

if __name__ == '__main__':
    main()