import numpy as np
import cvxpy as cp
import h5py
from scipy.stats import chi2
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    # 1. 直接设定物理限额的缩放，不区分 Opt 和 Real
    # 0.9 表示所有模型都在 90% 的容量下运行，超过 90% 就算违约
    'SCALE_FACTOR': 0.90, 
    
    'DELTA': 0.05,   
    'EPSILON': 0.01, # 99% 置信度
    'FIXED_TAU_MW': 400.0, # 给足 AGC 空间，让比赛焦点集中在线路拥塞上
    'N_TEST': 100,
    'PENALTY_COST': 1e6
}

# =========================================================================
#  2. 数据加载 (解耦优化限额与物理限额)
# =========================================================================
def load_grid_data(topo_path, gen_path):
    print(f"--- 加载电网拓扑: {topo_path} ---")
    baseMVA = 100.0
    with h5py.File(topo_path, 'r') as f:
        ptdf_raw = np.array(f['PTDF'])
        bus_shape = np.array(f['mpc']['bus']).shape
        num_buses = bus_shape[1] if len(bus_shape) > 1 else bus_shape[0]
        if ptdf_raw.shape[1] == num_buses: ptdf = ptdf_raw
        else: ptdf = ptdf_raw.T
        
        branch_data = np.array(f['mpc']['branch']).T
        raw_limits = branch_data[:, 5] 
        raw_limits[raw_limits == 0] = 9999 
        
        if 'mpc' in f and 'baseMVA' in f['mpc']:
            baseMVA = np.array(f['mpc']['baseMVA']).item()
         # [修正逻辑]
        # 1. 考核标准: 使用物理铭牌上的真实容量 (raw_limits)
        limits_mw_real = raw_limits
        
        # 2. 优化约束: 使用缩水后的容量 (raw_limits * Scale)
        # 目的是欺骗优化器，让它以为路很窄，从而被迫预留出 buffer
        limits_mw_opt = raw_limits * CONFIG['SCALE_FACTOR']    
        
        
    with h5py.File(gen_path, 'r') as f:
        c_base = np.array(f['c_base']).flatten()
        gen_bus_indices = np.array(f['gen_bus_indices']).flatten().astype(int) - 1
        p_min = np.array(f['Pmin']).flatten() / baseMVA
        p_max = np.array(f['Pmax']).flatten() / baseMVA
        
    num_gens = len(c_base)
    Cg = np.zeros((num_buses, num_gens))
    for i in range(num_gens): Cg[gen_bus_indices[i], i] = 1
        
    params = {
        'ptdf': ptdf, 
        'limits_mw_real': limits_mw_real, # 真值
        'limits_pu_opt':  limits_mw_opt / baseMVA, # 优化用(标幺值)
        'c_base': c_base, 'p_min': p_min, 'p_max': p_max,
        'Cg': Cg, 'baseMVA': baseMVA, 'num_buses': num_buses
    }
    return params

def load_scenario_data(mat_path, num_buses):
    if not os.path.exists(mat_path): return None, None
    with h5py.File(mat_path, 'r') as f:
        d_fcst = np.array(f['d_forecast'])
        d_real = np.array(f['d_real'])
        if d_fcst.shape[1] != num_buses:
            d_fcst = d_fcst.T
            d_real = d_real.T
    return d_fcst, d_real

# =========================================================================
#  3. 校准 (Ours 使用 99.5% 分位数)
# =========================================================================
def calibrate_data_driven(errors, target_prob):
    mu = np.mean(errors, axis=0)
    Sigma = np.cov(errors, rowvar=False) + np.eye(errors.shape[1]) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    diff = errors - mu
    mah_dist = np.sum((diff @ Sigma_inv) * diff, axis=1)
    sorted_dist = np.sort(mah_dist)
    
    # 确保索引不越界
    idx = int(np.ceil(len(errors) * target_prob))
    if idx >= len(errors): idx = len(errors) - 1
    
    rho = np.sqrt(sorted_dist[idx])
    return mu, rho, Sigma_sqrt

def calibrate_parametric_gaussian(errors, target_prob):
    # 剔除 5% 离群点
    norms = np.linalg.norm(errors, axis=1)
    threshold = np.percentile(norms, 95)
    clean_errors = errors[norms <= threshold]
    
    mu = np.mean(clean_errors, axis=0)
    Sigma = np.cov(clean_errors, rowvar=False) + np.eye(errors.shape[1]) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    rho_sq = chi2.ppf(target_prob, df=errors.shape[1])
    rho = np.sqrt(rho_sq)
    return mu, rho, Sigma_sqrt

# =========================================================================
#  4. 求解器
# =========================================================================
def solve_opf(params, d_t, robust_data, tau, mode='robust'):
    x_pu = cp.Variable(len(params['c_base']))
    slack_sys = cp.Variable(nonneg=True)
    slack_lines = cp.Variable(params['ptdf'].shape[0], nonneg=True)

    # 目标函数
    cost_gen = params['c_base'].T @ x_pu
    cost_penalty = CONFIG['PENALTY_COST'] * (slack_sys + cp.sum(slack_lines))
    obj = cp.Minimize(cost_gen + cost_penalty)
    
    constraints = [x_pu >= params['p_min'], x_pu <= params['p_max']]
    
    mu_pu, rho, Sigma_sqrt_pu = robust_data
    d_t_pu = d_t / params['baseMVA']
    
    # 1. 系统平衡
    nominal_imb = cp.sum(x_pu) - cp.sum(d_t_pu + mu_pu)
    tau_pu = tau / params['baseMVA']
    
    if mode == 'robust':
        sys_margin = rho * np.linalg.norm(Sigma_sqrt_pu @ np.ones(params['num_buses']), 2)
                # [安全] 确保 Margin 不超过 Tau 的 90%
        capped_sys_margin = min(sys_margin, 0.9 * (tau / params['baseMVA']))
        bound_pu = (tau / params['baseMVA']) - capped_sys_margin
        constraints += [cp.abs(nominal_imb) <= bound_pu + slack_sys]
    else:
        constraints += [cp.sum(x_pu) - cp.sum(d_t_pu) == 0 + slack_sys]

    # 2. 线路潮流 (使用 limits_pu_opt 即缩水后的限额)
    inj = params['Cg'] @ x_pu - (d_t_pu + mu_pu)
    flow = params['ptdf'] @ inj
    limits_pu = params['limits_pu_opt'] # 优化时用严苛标准
    
    if mode == 'robust':
        raw_margins = rho * np.linalg.norm(params['ptdf'] @ Sigma_sqrt_pu, axis=1)
        
        # [关键修复] 裕度截断 (Margin Capping)
        # 强制鲁棒裕度最大只能占用线路容量的 80%
        # 这样保证了 eff_limits 至少有 20% 的空间供电力流动，避免优化器"弃疗"
        capped_margins = np.minimum(raw_margins, 0.8 * limits_pu)
        
        eff_limits = limits_pu - capped_margins
        constraints += [cp.abs(flow) <= eff_limits + slack_lines]
    else:
        constraints += [cp.abs(flow) <= limits_pu + slack_lines]        
    prob = cp.Problem(obj, constraints)
    try: prob.solve(solver=cp.ECOS)
    except: return None, "Error"
    if prob.status not in ['optimal', 'optimal_inaccurate']: return None, prob.status
    return x_pu.value * params['baseMVA'], "Optimal"

# =========================================================================
#  5. 主程序 (修正 KeyError 版)
# =========================================================================
def run_experiment():
    params = load_grid_data('network_topology.mat', 'generator_parameters.mat')
    
    datasets = [
        ('data_gaussian_case39.mat', 'Gaussian'),
        ('data_student-t_case39.mat', 'Student-T'),
        ('data_mixed_case39.mat', 'Mixed')
    ]
    
    print("\n" + "="*95)
    print(f"{'Dataset':<18} | {'Model':<15} | {'Avg Cost($)':<12} | {'Viol Rate(%)':<12} | {'Feasible(%)':<12}")
    print("="*95)
    
    for mat_name, label in datasets:
        d_fcst, d_real = load_scenario_data(mat_name, params['num_buses'])
        if d_fcst is None: continue
        
        # 校准
        n_cal = 1000
        errs = (d_real[:n_cal] - d_fcst[:n_cal]) / params['baseMVA']
        
        rb_ours = calibrate_data_driven(errs, 1 - CONFIG['EPSILON'])
        rb_gauss = calibrate_parametric_gaussian(errs, 1 - CONFIG['EPSILON'])
        
        print(f"[{label}] Rho: Ours={rb_ours[1]:.2f} vs Param={rb_gauss[1]:.2f}")
        
        tau = CONFIG['FIXED_TAU_MW']
        
        # 筛选高误差样本
        error_norms = np.linalg.norm(d_real - d_fcst, axis=1)
        test_pool_idx = np.arange(n_cal, len(d_fcst))
        worst_indices_global = test_pool_idx[np.argsort(error_norms[test_pool_idx])[-200:]]
        
        np.random.seed(42)
        test_idx = np.random.choice(worst_indices_global, CONFIG['N_TEST'], replace=False)
        
        res = {'Ours': {'c':[], 'v':[], 'f':0}, 'Param': {'c':[], 'v':[], 'f':0}, 'Det': {'c':[], 'v':[], 'f':0}}
        
        for idx in test_idx:
            d_t_f = d_fcst[idx]
            d_t_r = d_real[idx]
            
            out_ours = solve_opf(params, d_t_f, rb_ours, tau, 'robust')
            out_para = solve_opf(params, d_t_f, rb_gauss, tau, 'robust')
            out_det  = solve_opf(params, d_t_f, rb_ours, tau, 'deterministic')
            
            # --- [修改] 修正了 record 函数中的 Key ---
            def record(key, out):
                x, status = out
                if x is None: return
                res[key]['f'] += 1
                res[key]['c'].append(params['c_base'].T @ (x / params['baseMVA']))
                
                # 1. 线路检查: 必须使用 'limits_mw_real' (物理真值)
                inj = params['Cg'] @ (x / params['baseMVA']) - d_t_r / params['baseMVA']
                f_real = params['ptdf'] @ inj * params['baseMVA']
                
                # [Fixed] 这里原来是 params['limits_mw']，现在改为 params['limits_mw_real']
                viol = np.any(np.abs(f_real) > params['limits_mw_real'] * 1.001)
                
                # 2. 系统平衡检查
                if np.abs(np.sum(x) - np.sum(d_t_r)) > tau: viol = True
                
                res[key]['v'].append(1 if viol else 0)
            # ----------------------------------------

            record('Ours', out_ours)
            record('Param', out_para)
            record('Det', out_det)
            
        def p(name, d):
            c = np.mean(d['c']) if d['c'] else 0
            v = np.mean(d['v']) * 100 if d['v'] else 0
            f = d['f'] / CONFIG['N_TEST'] * 100
            print(f"{label:<18} | {name:<15} | {c:<12.2f} | {v:<12.1f} | {f:<12.1f}")
            
        p("Ours", res['Ours'])
        p("Param", res['Param'])
        p("Determ", res['Det'])
        print("-" * 95)
        
if __name__ == '__main__':
    run_experiment()