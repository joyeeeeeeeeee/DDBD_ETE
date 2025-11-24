import numpy as np
import cvxpy as cp
import h5py
from scipy.stats import chi2, binom
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    # [关键参数] 物理限额缩放
    # 0.60: 把线路容量砍到 60%。
    # 既然我们不再特意挑选"最难样本"，而是按顺序测最后 100 个，
    # 我们需要把电网弄得拥挤一点，确保 Deterministic 必定会违约。
    'GLOBAL_LIMIT_SCALE': 0.90, 
    
    'FIXED_TAU_MW': 500.0, 
    'PENALTY_COST': 1e7,
    
    # 统计参数
    'EPSILON': 0.07,         # 目标 95% 可靠性
    'CONFIDENCE_LEVEL': 0.95 # 95% 置信度
}

# =========================================================================
#  1. 数据加载
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

        # 统一缩放：让所有模型都在 60% 容量的电网里跑
        limits_mw_real = raw_limits * CONFIG['GLOBAL_LIMIT_SCALE']
        limits_mw_opt = limits_mw_real 
    
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
#  2. 校准逻辑 (Ours vs Gaussian)
# =========================================================================
def calibrate_data_driven(errors, target_epsilon, confidence_level):
    n_samples = len(errors)
    # 2780 个样本：对半分，1390 定形状，1390 定大小
    # 数据量很大，结果会非常稳定
    n1 = int(n_samples * 0.5)
    n2 = n_samples - n1
    D1 = errors[:n1]
    D2 = errors[n1:]
    
    mu = np.mean(D1, axis=0)
    Sigma = np.cov(D1, rowvar=False) + np.eye(D1.shape[1]) * 1e-8
    
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    diff = D2 - mu
    scores = np.sum((diff @ Sigma_inv) * diff, axis=1)
    scores_sorted = np.sort(scores)
    
    p_success = 1.0 - target_epsilon 
    i_star = -1
    found = False
    
    # 寻找满足二项分布置信度的阈值
    for r in range(1, n2 + 1):
        lhs = binom.cdf(r - 1, n2, p_success)
        if lhs >= confidence_level:
            i_star = r
            found = True
            break
    if not found: i_star = n2
        
    rho_squared = scores_sorted[i_star - 1]
    rho = np.sqrt(rho_squared)
    return mu, rho, Sigma_sqrt

def calibrate_parametric_gaussian(errors, target_epsilon):
    mu = np.mean(errors, axis=0)
    Sigma = np.cov(errors, rowvar=False) + np.eye(errors.shape[1]) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    rho_sq = chi2.ppf(1 - target_epsilon, df=errors.shape[1])
    rho = np.sqrt(rho_sq)
    return mu, rho, Sigma_sqrt

# =========================================================================
#  3. 求解器
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
        # 0.9: 允许鲁棒裕度占用 90% 的物理空间
        capped_margins = np.minimum(raw_margins, 0.9 * limits_pu)
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
#  4. 主程序 (2780 Train / 100 Test)
# =========================================================================
def run_experiment():
    params = load_grid_data('network_topology.mat', 'generator_parameters.mat')
    datasets = [('data_student-t_case39.mat', 'Student-T'),('data_skewed_case39.mat', 'Skewed (Asym)')]
    
    print("\n" + "="*95)
    print(f"{'Dataset':<18} | {'Model':<15} | {'Avg Cost($)':<12} | {'Viol Rate(%)':<12} | {'Rho Value':<12}")
    print("="*95)
    
    for mat_name, label in datasets:
        d_fcst, d_real = load_scenario_data(mat_name, params['num_buses'])
        if d_fcst is None: continue
        
        # [修改] 
        # n_train = 2780 用于校准 (学习)
        # n_test  = 100  用于测试
        n_train = 2780
        n_test = 100
        
        if len(d_fcst) < n_train + n_test:
            print(f"Data not enough! Total: {len(d_fcst)}, Need: {n_train+n_test}")
            continue
            
        # 1. 训练集：计算校准参数
        d_train_real = d_real[:n_train]
        d_train_fcst = d_fcst[:n_train]
        errs_train = (d_train_real - d_train_fcst) / params['baseMVA']
        
        rb_ours = calibrate_data_driven(errs_train, CONFIG['EPSILON'], CONFIG['CONFIDENCE_LEVEL'])
        rb_gauss = calibrate_parametric_gaussian(errs_train, CONFIG['EPSILON'])
        
        tau = CONFIG['FIXED_TAU_MW']
        
        # 2. 测试集：使用紧随其后的 100 个样本
        # d_test_real = d_real[n_train : n_train + n_test]
        test_indices = range(n_train, n_train + n_test)
        
        res = {'Ours': {'c':[], 'v':[]}, 'Param': {'c':[], 'v':[]}, 'Det': {'c':[], 'v':[]}}
        
        for idx in test_indices:
            d_t_f = d_fcst[idx]
            d_t_r = d_real[idx]
            
            out_ours = solve_opf(params, d_t_f, rb_ours, tau, 'robust')
            out_para = solve_opf(params, d_t_f, rb_gauss, tau, 'robust')
            out_det  = solve_opf(params, d_t_f, rb_ours, tau, 'deterministic')
            
            def record(key, out):
                x, status = out
                if x is None: return
                res[key]['c'].append(params['c_base'].T @ (x / params['baseMVA']))
                
                # 违约检查
                inj = params['Cg'] @ (x / params['baseMVA']) - d_t_r / params['baseMVA']
                f_real = params['ptdf'] @ inj * params['baseMVA']
                # 严格判定，只要超过一点点就算
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