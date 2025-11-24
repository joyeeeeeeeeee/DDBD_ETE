import numpy as np
import cvxpy as cp
import h5py
from scipy.stats import chi2
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    # [调整] 稍微降低优化时的线路容量上限，留出更多物理余量 (从0.95降到0.93)
    # 这能直接显著降低违约率
    'OPT_SCALE_FACTOR': 0.93, 
    
    # 考核时限额 (100%)
    'CHECK_REAL_LIMITS': True,

    'DELTA': 0.05,   
    
    # [调整] 稍微放宽一点 EPSILON，让 Rho 不要因为个别极值变得过大
    # 这样配合更准的 Sigma 形状，效果更好
    'EPSILON': 0.05, 
    
    'FIXED_TAU_MW': 500.0, 
    'N_TEST': 100,
    'PENALTY_COST': 1e6
}

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

        limits_mw_real = raw_limits
        limits_mw_opt = raw_limits * CONFIG['OPT_SCALE_FACTOR']
    
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
        'limits_mw_real': limits_mw_real, 
        'limits_pu_opt':  limits_mw_opt / baseMVA, 
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
#  [核心修改] 校准函数
# =========================================================================
def calibrate_data_driven(errors, target_prob):
    # 1. 均值计算：依然可以使用剔除极端值后的均值，防止中心偏移过大
    norms = np.linalg.norm(errors, axis=1)
    threshold_95 = np.percentile(norms, 95)
    clean_errors = errors[norms <= threshold_95]
    mu = np.mean(clean_errors, axis=0)
    
    # 2. [重要修改] 协方差计算 (Sigma)
    # 之前只用 clean_errors 计算 Sigma，导致丢掉了 Spike 的方向信息。
    # 现在改用 全量 errors 计算 Sigma，或者至少包含大部分数据。
    # 并且加入 Shrinkage (收缩估计)，防止过拟合。
    
    # 使用全量数据计算原始协方差，捕捉尖峰方向
    Sigma_raw = np.cov(errors, rowvar=False)
    
    # 收缩估计 (Shrinkage): 
    # 95% 原始形状 + 5% 单位圆。这能保证在所有方向都有最基本的防御，
    # 同时让椭球主要轴向对准数据的方差方向。
    mean_eig = np.trace(Sigma_raw) / Sigma_raw.shape[0]
    Sigma = 0.95 * Sigma_raw + 0.05 * mean_eig * np.eye(Sigma_raw.shape[0])
    
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    # 3. 尺寸校准 (Rho)
    diff = errors - mu
    mah_dist = np.sum((diff @ Sigma_inv) * diff, axis=1)
    sorted_dist = np.sort(mah_dist)
    
    # 使用 target_prob 对应的百分位
    idx = int(np.ceil(len(errors) * target_prob))
    if idx >= len(errors): idx = len(errors) - 1
    rho = np.sqrt(sorted_dist[idx])
    
    # [调整] 放宽 Capping，但不要让它无限大
    # 因为现在的 Sigma 包含了尖峰，算出来的 Mahalanobis 距离实际上会变小
    # (因为分母 Sigma 变大了)，所以 Rho 可能不需要强行截断。
    # 但为了安全，保留一个合理的上限。
    if rho > 15.0:
        rho = 15.0
        
    return mu, rho, Sigma_sqrt

def calibrate_parametric_gaussian(errors, target_prob):
    # Parametric 保持不变，作为基准
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
#  求解器
# =========================================================================
def solve_opf(params, d_t, robust_data, tau, mode='robust'):
    x_pu = cp.Variable(len(params['c_base']))
    slack_lines = cp.Variable(params['ptdf'].shape[0], nonneg=True)

    cost_gen = params['c_base'].T @ x_pu
    cost_penalty = CONFIG['PENALTY_COST'] * cp.sum(slack_lines)
    obj = cp.Minimize(cost_gen + cost_penalty)
    
    constraints = [x_pu >= params['p_min'], x_pu <= params['p_max']]
    
    mu_pu, rho, Sigma_sqrt_pu = robust_data
    d_t_pu = d_t / params['baseMVA']
    
    # 系统平衡
    nominal_imb = cp.sum(x_pu) - cp.sum(d_t_pu + mu_pu)
    tau_pu = tau / params['baseMVA']
    
    if mode == 'robust':
        sys_margin = rho * np.linalg.norm(Sigma_sqrt_pu @ np.ones(params['num_buses']), 2)
        bound_pu = max(tau_pu - sys_margin, 0)
        constraints += [cp.abs(nominal_imb) <= bound_pu]
    else:
        constraints += [cp.sum(x_pu) - cp.sum(d_t_pu) == 0]

    # 线路潮流
    inj = params['Cg'] @ x_pu - (d_t_pu + mu_pu)
    flow = params['ptdf'] @ inj
    limits_pu = params['limits_pu_opt'] 
    
    if mode == 'robust':
        raw_margins = rho * np.linalg.norm(params['ptdf'] @ Sigma_sqrt_pu, axis=1)
        
        # [调整] 稍微放宽截断逻辑
        # 之前是 0.6，现在改为 0.7，允许鲁棒余量占据更多线路容量
        # 这会迫使调度结果更加保守（Cost增加），但能显著降低违约率
        capped_margins = np.minimum(raw_margins, 0.7 * limits_pu)
        
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
#  主程序
# =========================================================================
def run_experiment():
    params = load_grid_data('network_topology.mat', 'generator_parameters.mat')
    
    datasets = [
        ('data_gaussian_case39.mat', 'Gaussian'),
        ('data_student-t_case39.mat', 'Student-T'),
        ('data_mixed_case39.mat', 'Mixed'),
        ('data_skewed_case39.mat', 'Skewed (Asym)')
    ]
    print("\n" + "="*95)
    print(f"{'Dataset':<18} | {'Model':<15} | {'Avg Cost($)':<12} | {'Viol Rate(%)':<12} | {'Rho Value':<12}")
    print("="*95)
    
    for mat_name, label in datasets:
        d_fcst, d_real = load_scenario_data(mat_name, params['num_buses'])
        if d_fcst is None: continue
        
        n_cal = 1000
        errs = (d_real[:n_cal] - d_fcst[:n_cal]) / params['baseMVA']
        
        # 校准
        rb_ours = calibrate_data_driven(errs, 1 - CONFIG['EPSILON'])
        rb_gauss = calibrate_parametric_gaussian(errs, 1 - CONFIG['EPSILON'])
        
        tau = CONFIG['FIXED_TAU_MW']
        
        # 筛选误差最大的点进行测试
        error_norms = np.linalg.norm(d_real - d_fcst, axis=1)
        test_pool_idx = np.arange(n_cal, len(d_fcst))
        worst_indices = test_pool_idx[np.argsort(error_norms[test_pool_idx])[-100:]]
        
        res = {'Ours': {'c':[], 'v':[]}, 'Param': {'c':[], 'v':[]}, 'Det': {'c':[], 'v':[]}}
        
        for idx in worst_indices:
            d_t_f = d_fcst[idx]
            d_t_r = d_real[idx]
            
            out_ours = solve_opf(params, d_t_f, rb_ours, tau, 'robust')
            out_para = solve_opf(params, d_t_f, rb_gauss, tau, 'robust')
            out_det  = solve_opf(params, d_t_f, rb_ours, tau, 'deterministic')
            
            def record(key, out):
                x, status = out
                if x is None: return
                res[key]['c'].append(params['c_base'].T @ (x / params['baseMVA']))
                
                # 考核：用真实 Limits
                inj = params['Cg'] @ (x / params['baseMVA']) - d_t_r / params['baseMVA']
                f_real = params['ptdf'] @ inj * params['baseMVA']
                
                # 判断违约
                viol = np.any(np.abs(f_real) > params['limits_mw_real'] * 1.0001)
                res[key]['v'].append(1 if viol else 0)

            record('Ours', out_ours)
            record('Param', out_para)
            record('Det', out_det)
            
        def p(name, d, rho_val):
            c = np.mean(d['c']) if d['c'] else 0
            v = np.mean(d['v']) * 100 if d['v'] else 0
            print(f"{label:<18} | {name:<15} | {c:<12.2f} | {v:<12.1f} | {rho_val:<12.2f}")
            
        p("Ours", res['Ours'], rb_ours[1])
        p("Param", res['Param'], rb_gauss[1])
        p("Determ", res['Det'], 0.0)
        print("-" * 95)

if __name__ == '__main__':
    run_experiment()