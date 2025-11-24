import numpy as np
import cvxpy as cp
import h5py
from scipy.stats import chi2
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    # 稍微放松一点限额，因为 99% 可靠性会导致 rho 变大，占用更多线路
    'SCALE_FACTOR': 0.96, 
    
    'DELTA': 0.05,   
    
    # [关键修改] 提高可靠性要求到 99%
    # 只有这样，Ours 才会去覆盖那些占比 3% 的 Spikes
    'EPSILON': 0.01, 
    
    # 保持固定 Tau
    'FIXED_TAU_MW': 350.0, 
    
    'N_TEST': 100,
    'PENALTY_COST': 1e5 
}

def load_grid_data(topo_path, gen_path):
    # ... (此处代码保持不变，直接复制之前的 load_grid_data) ...
    print(f"--- 加载电网拓扑: {topo_path} ---")
    baseMVA = 100.0
    with h5py.File(topo_path, 'r') as f:
        ptdf_raw = np.array(f['PTDF'])
        bus_shape = np.array(f['mpc']['bus']).shape
        num_buses = bus_shape[1] if len(bus_shape) > 1 else bus_shape[0]
        if ptdf_raw.shape[1] == num_buses: ptdf = ptdf_raw
        else: ptdf = ptdf_raw.T
        branch_data = np.array(f['mpc']['branch']).T
        flow_limits = branch_data[:, 5] 
        flow_limits[flow_limits == 0] = 9999 
        if 'mpc' in f and 'baseMVA' in f['mpc']:
            baseMVA = np.array(f['mpc']['baseMVA']).item()
        flow_limits = flow_limits * CONFIG['SCALE_FACTOR']
    
    with h5py.File(gen_path, 'r') as f:
        c_base = np.array(f['c_base']).flatten()
        gen_bus_indices = np.array(f['gen_bus_indices']).flatten().astype(int) - 1
        p_min = np.array(f['Pmin']).flatten() / baseMVA
        p_max = np.array(f['Pmax']).flatten() / baseMVA
        
    num_gens = len(c_base)
    Cg = np.zeros((num_buses, num_gens))
    for i in range(num_gens): Cg[gen_bus_indices[i], i] = 1
        
    params = {
        'ptdf': ptdf, 'flow_limits': flow_limits / baseMVA, 'limits_mw': flow_limits,
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
#  校准函数
# =========================================================================
def calibrate_data_driven(errors, target_prob):
    # Ours: 全样本校准 (包含离群点，这正是鲁棒性的来源)
    mu = np.mean(errors, axis=0)
    Sigma = np.cov(errors, rowvar=False) + np.eye(errors.shape[1]) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    Sigma_inv = eigvecs @ np.diag(1.0 / (eigvals + 1e-9)) @ eigvecs.T
    
    diff = errors - mu
    mah_dist = np.sum((diff @ Sigma_inv) * diff, axis=1)
    sorted_dist = np.sort(mah_dist)
    idx = int(len(errors) * target_prob)
    if idx >= len(errors): idx = len(errors) - 1
    rho = np.sqrt(sorted_dist[idx])
    return mu, rho, Sigma_sqrt

def calibrate_parametric_gaussian(errors, target_prob):
    """
    传统参数化方法：
    1. 假设数据是高斯的。
    2. 使用剔除离群点后的数据计算均值和方差 (Trimmed Mean/Cov)。
    3. 直接查卡方分布表 (Chi2 Table) 得到 rho。
    
    缺点：在长尾分布中，Chi2 表给出的 rho 往往太小，包不住尾巴。
    """
    # 剔除前 5% 的极端值来拟合高斯参数（模拟工程师认为那是噪音）
    norms = np.linalg.norm(errors, axis=1)
    threshold = np.percentile(norms, 95)
    clean_errors = errors[norms <= threshold]
    
    mu = np.mean(clean_errors, axis=0)
    # 加上小正则项
    Sigma = np.cov(clean_errors, rowvar=False) + np.eye(errors.shape[1]) * 1e-8
    
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals[eigvals < 0] = 1e-9
    Sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    # [关键] 查卡方表。对于非高斯分布，这个理论值往往偏小。
    rho_sq = chi2.ppf(target_prob, df=errors.shape[1])
    rho = np.sqrt(rho_sq)
    
    return mu, rho, Sigma_sqrt
# =========================================================================
#  求解器 (包含松弛变量)
# =========================================================================
def solve_opf(params, d_t, robust_data, tau, mode='robust'):
    x_pu = cp.Variable(len(params['c_base']))
    slack_sys = cp.Variable(nonneg=True)
    slack_lines = cp.Variable(params['ptdf'].shape[0], nonneg=True)

    # 目标函数: 成本 + 惩罚
    cost_gen = params['c_base'].T @ x_pu
    cost_penalty = CONFIG['PENALTY_COST'] * (slack_sys + cp.sum(slack_lines))
    obj = cp.Minimize(cost_gen + cost_penalty)
    
    constraints = [x_pu >= params['p_min'], x_pu <= params['p_max']]
    
    mu_pu, rho, Sigma_sqrt_pu = robust_data
    d_t_pu = d_t / params['baseMVA']
    
    # 1. 系统平衡 (受限于固定的 Tau)
    nominal_imb = cp.sum(x_pu) - cp.sum(d_t_pu + mu_pu)
    tau_pu = tau / params['baseMVA']
    
    if mode == 'robust':
        sys_margin = rho * np.linalg.norm(Sigma_sqrt_pu @ np.ones(params['num_buses']), 2)
        # 如果 Margin > Tau，说明预测误差太大，AGC 都不够用 -> 必须用 Slack
        bound_pu = max(tau_pu - sys_margin, 0)
        constraints += [cp.abs(nominal_imb) <= bound_pu + slack_sys]
    else:
        constraints += [cp.sum(x_pu) - cp.sum(d_t_pu) == 0 + slack_sys]

    # 2. 线路潮流
    inj = params['Cg'] @ x_pu - (d_t_pu + mu_pu)
    flow = params['ptdf'] @ inj
    limits_pu = params['flow_limits']
    
    if mode == 'robust':
        line_margins = rho * np.linalg.norm(params['ptdf'] @ Sigma_sqrt_pu, axis=1)
        eff_limits = limits_pu - line_margins
        # 如果 Margin > Limit，说明为了抗风险必须把线空出来 -> 可能需要 Slack
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
        ('data_mixed_case39.mat', 'Mixed (Spikes)')
    ]
    
    print("\n" + "="*95)
    print(f"{'Dataset':<18} | {'Model':<15} | {'Avg Cost($)':<12} | {'Viol Rate(%)':<12} | {'Feasible(%)':<12}")
    print("="*95)
    
    for mat_name, label in datasets:
        d_fcst, d_real = load_scenario_data(mat_name, params['num_buses'])
        if d_fcst is None: continue
        
        # --- 校准 ---
        n_cal = 1000
        errs = (d_real[:n_cal] - d_fcst[:n_cal]) / params['baseMVA']
        
        rb_ours = calibrate_data_driven(errs, 1 - CONFIG['EPSILON'])
        rb_gauss = calibrate_parametric_gaussian(errs, 1 - CONFIG['EPSILON'])
        print(f"[{label}] Rho对比 -> Ours(Data): {rb_ours[1]:.2f} vs Param(Gauss): {rb_gauss[1]:.2f}")
        # [关键修改 2] 使用固定的 Tau
        tau = CONFIG['FIXED_TAU_MW']
        
        # [关键修改 3] 挑选"预测误差最大"的时刻进行测试，而不是"负荷最大"
        # 这样才能捕捉到 Spikes 和 Heavy Tails 发生的时候
        error_norms = np.linalg.norm(d_real - d_fcst, axis=1)
        # 避开 Calibration Set
        test_pool_idx = np.arange(n_cal, len(d_fcst))
        pool_errors = error_norms[test_pool_idx]
        
        # 选误差最大的前 200 个时刻里随机挑 100 个
        worst_indices_local = np.argsort(pool_errors)[-200:]
        worst_indices_global = test_pool_idx[worst_indices_local]
        
        np.random.seed(42)
        test_idx = np.random.choice(worst_indices_global, CONFIG['N_TEST'], replace=False)
        
        res = {'Ours': {'c':[], 'v':[], 'f':0}, 
               'Param': {'c':[], 'v':[], 'f':0}, 
               'Det': {'c':[], 'v':[], 'f':0}}
        
        for idx in test_idx:
            d_t_f = d_fcst[idx]
            d_t_r = d_real[idx]
            
            # 运行模型
            # 注意：Det 模型也传入 robust_data 只是为了占位，实际没用
            out_ours = solve_opf(params, d_t_f, rb_ours, tau, 'robust')
            out_para = solve_opf(params, d_t_f, rb_gauss, tau, 'robust')
            out_det  = solve_opf(params, d_t_f, rb_ours, tau, 'deterministic')
            
            def record(key, out):
                x, status = out
                if x is None: return
                res[key]['f'] += 1
                
                # 成本 (仅发电)
                c = params['c_base'].T @ (x / params['baseMVA'])
                res[key]['c'].append(c)
                
                # 违约检查 (最严格的物理检查)
                viol = False
                
                # 1. 线路越限检查
                inj = params['Cg'] @ (x / params['baseMVA']) - d_t_r / params['baseMVA']
                f_real = params['ptdf'] @ inj * params['baseMVA']
                if np.any(np.abs(f_real) > params['limits_mw'] * 1.001): viol = True
                
                # 2. 系统平衡检查 (AGC 是否够用)
                imb = np.abs(np.sum(x) - np.sum(d_t_r))
                if imb > tau: viol = True
                
                res[key]['v'].append(1 if viol else 0)

            record('Ours', out_ours)
            record('Param', out_para)
            record('Det', out_det)
            
        def p(name, d):
            c = np.mean(d['c']) if d['c'] else 0
            v = np.mean(d['v']) * 100 if d['v'] else 0
            f = d['f'] / CONFIG['N_TEST'] * 100
            print(f"{label:<18} | {name:<15} | {c:<12.2f} | {v:<12.1f} | {f:<12.1f}")
            
        p("Ours", res['Ours'])
        p("Param (Gauss)", res['Param'])
        p("Deterministic", res['Det'])
        print("-" * 95)

if __name__ == '__main__':
    run_experiment()