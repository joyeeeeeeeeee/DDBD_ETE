# exp_section3.py (Final Corrected Version)
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
from scipy.stats import chi2
from utils import load_data, get_true_topology
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_scores(train_noise, mu, Sigma, rho_gauss, rho_ddro):
    """
    画图证明：你的数据分布 vs 理论卡方分布
    """
    Sigma_inv = np.linalg.inv(Sigma + 1e-6*np.eye(Sigma.shape[0]))
    
    # 计算所有训练数据的马氏距离平方
    scores = []
    for i in range(train_noise.shape[0]):
        diff = train_noise[i] - mu
        scores.append(diff.T @ Sigma_inv @ diff)
    scores = np.array(scores)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, stat="density", bins=50, color="skyblue", label="Empirical Data Distribution", alpha=0.6)
    
    x = np.linspace(0, max(scores), 100)
    df = train_noise.shape[1]
    plt.plot(x, chi2.pdf(x, df), 'r-', lw=2, label=f"Theoretical Chi2 (Gaussian Assumption)")
    
    plt.axvline(rho_gauss, color='r', linestyle='--', label=f"Gaussian 95% Bound ({rho_gauss:.1f})")
    plt.axvline(rho_ddro, color='b', linestyle='--', label=f"DDRO 95% Bound ({rho_ddro:.1f})")
    
    plt.title("Why Gaussian CCP Fails: Empirical vs Theoretical Tails")
    plt.xlabel("Non-conformity Score (Mahalanobis Distance^2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_robust_parameters(train_noise, cal_noise, H, C_gen, alpha, epsilon=0.05):
    """
    严格基于 Split Conformal Prediction 计算鲁棒参数
    并引入 Affine Recourse 对线路 Margin 的修正
    """
    print(f"\n[Calibration] 正在执行保形预测校准 (Target Violation < {epsilon*100}%)...")
    
    # 1. 形状估计 (Shape Estimation) - 使用训练集
    mu = np.mean(train_noise, axis=0)
    Sigma = np.cov(train_noise, rowvar=False)
    # 加微小扰动防止奇异
    Sigma += 1e-6 * np.eye(Sigma.shape[0])
    
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_sqrt = np.real(sqrtm(Sigma))
    
    # 2. 分数计算 (Score Computation) - 使用校准集
    n_cal = cal_noise.shape[0]
    scores = []
    
    for i in range(n_cal):
        diff = cal_noise[i] - mu
        score = diff.T @ Sigma_inv @ diff
        scores.append(score)
    scores = np.array(scores)
    
    # 3. 半径校准 (Radius Calibration)
    rho_ddro = np.quantile(scores, 1 - epsilon)
    
    # --- 诊断对比 ---
    df = train_noise.shape[1] 
    rho_gauss = chi2.ppf(1 - epsilon, df)
    
    print(f"  > 自由度 (Dimension): {df}")
    print(f"  > 理论 Gaussian 半径 (Chi2): {rho_gauss:.2f}")
    print(f"  > 数据驱动 DDRO 半径 (Quantile): {rho_ddro:.2f}")
    
    if rho_ddro > rho_gauss * 1.1:
        print("  > [结论] 检测到显著的厚尾特征！DDRO 半径比 Gaussian 大，安全性更高。")
    elif rho_ddro < rho_gauss * 0.9:
        print("  > [结论] 数据分布比正态分布更集中（薄尾）。")
    else:
        print("  > [结论] 数据非常接近正态分布，两种方法结果相似。")

    # 4. 映射到约束 (Map to Constraints)
    
    # A. 功率平衡 (Power Balance)
    # Margin = sqrt(rho) * || Sigma^1/2 * 1 ||
    vec_ones = np.ones(train_noise.shape[1])
    term_bal = Sigma_sqrt @ vec_ones
    omega_margin = np.sqrt(rho_ddro) * np.linalg.norm(term_bal)
    
    # B. 线路潮流 (Line Flow) - 关键修正！
    # 必须考虑发电机响应带来的反向潮流
    # Effective Injection Sensitivity S = (vec_inj * 1^T - I)
    vec_inj = C_gen @ alpha # [n_bus, ]
    S = np.outer(vec_inj, np.ones(H.shape[1])) - np.eye(H.shape[1])
    
    # Effective Line Sensitivity W = H * S
    W = H @ S
    
    # Margin = sqrt(rho) * || W * Sigma^1/2 || (按行求范数)
    term_line = W @ Sigma_sqrt
    line_margin = np.sqrt(rho_ddro) * np.linalg.norm(term_line, axis=1)
    
    return omega_margin, line_margin, mu, rho_ddro

def solve_affine_dispatch(d_forecast, C_gen, H, gen_max, branch_limit, alpha, 
                          omega_margin, line_margin, prices): 
    """
    仿射策略调度求解器 (带经济目标)
    """
    n_gen = len(gen_max)
    x = cp.Variable(n_gen)
    
    # 仿射备用
    reserve_req = alpha * omega_margin 
    
    constraints = [
        # 基准平衡
        cp.sum(x) == np.sum(d_forecast),
        
        # 仿射容量约束 (预留备用)
        x <= gen_max - reserve_req,
        x >= reserve_req, 
        
        # 线路鲁棒约束
        H @ (C_gen @ x - d_forecast) <= branch_limit - line_margin,
        H @ (C_gen @ x - d_forecast) >= -branch_limit + line_margin
    ]
    
    # 使用梯级电价最小化成本
    cost = prices @ x
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    try:
        prob.solve(solver=cp.ECOS)
    except:
        return None, False, 0
        
    if prob.status != 'optimal':
        return None, False, 0
    
    return x.value, True, prob.value

def run_experiment():
    print("=== 开始第三章实验：Deep Diagnostics Version (With Affine Line Correction) ===")
    data = load_data()
    C_gen, H = get_true_topology(data['mpc_raw'])
    
    # 1. 数据划分
    N = data['load_noise'].shape[0]
    n_train = int(N * 0.5)
    n_cal = int(N * 0.2)
    
    train_noise = data['load_noise'][:n_train]
    cal_noise = data['load_noise'][n_train : n_train + n_cal]
    test_forecast = data['load_forecast'][n_train + n_cal:]
    test_real = data['load_real'][n_train + n_cal:]
    
    # 2. 梯级电价
    gen_prices = np.linspace(10, 100, data['n_gen'])
    
    # --- 参数计算与诊断 ---
    print("\n[Margin Diagnostics]")
    
    # 准备 Affine 修正所需的 S 矩阵 (用于 Gaussian 和 RO 的计算，保持公平)
    vec_inj = C_gen @ data['alpha']
    S = np.outer(vec_inj, np.ones(H.shape[1])) - np.eye(H.shape[1])
    W = H @ S # 有效灵敏度矩阵
    
    # A. Gaussian
    df = train_noise.shape[1]
    rho_gauss = chi2.ppf(0.95, df)
    
    Sig_g = np.cov(train_noise, rowvar=False) + 1e-6*np.eye(df)
    Sig_sqrt_g = np.real(sqrtm(Sig_g))
    
    omega_gauss = np.sqrt(rho_gauss) * np.linalg.norm(Sig_sqrt_g @ np.ones(df))
    # Gaussian Line Margin 也要用 W 矩阵
    line_gauss = np.sqrt(rho_gauss) * np.linalg.norm(W @ Sig_sqrt_g, axis=1)
    
    print(f"  > Gaussian Margin : {omega_gauss:.2f} MW")

    # B. DDRO
    # 这里的函数已经内置了 W 矩阵的逻辑
    omega_ddro, line_ddro, _, rho_ddro_val = calculate_robust_parameters(
        train_noise, cal_noise, H, C_gen, data['alpha'], epsilon=0.05
    )
    print(f"  > DDRO Margin     : {omega_ddro:.2f} MW")

    # C. Traditional RO
    train_imbalance = np.abs(np.sum(train_noise, axis=1))
    omega_ro = np.max(train_imbalance)
    
    # RO Line Margin 也要用 W 矩阵 (W * noise)
    # 这样才是物理上公平的对比
    train_flow_err = np.abs(train_noise @ W.T) # [Samples, n_lines]
    line_ro = np.max(train_flow_err, axis=0)
    
    print(f"  > Trad. RO Margin : {omega_ro:.2f} MW")
    
    # 逻辑检查
    if omega_ro < omega_gauss:
        print(f"{omega_gauss:.2f} {omega_ro:.2f}   !!! 警告：RO 小于 Gaussian，说明训练集里没有尖峰！请检查 utils.py 数据生成逻辑。")
    else:
        print("  >>> 正常：RO 大于 Gaussian，说明捕捉到了历史极值。")

    # --- 评估循环 ---
    methods = {
        'Deterministic': (0.0, np.zeros_like(line_ddro)),
        'Gaussian CCP':  (omega_gauss, line_gauss),
        'Ours (CP-DDRO)': (omega_ddro, line_ddro),
        'Traditional RO': (omega_ro, line_ro)
    }
    
    print(f"\n{'Method':<15} | {'Viol%':<8} | {'Infeas':<6} | {'Cost($)':<10} | {'GenVio':<6} | {'LineVio':<6}")
    print("-" * 75)
    
    for name, (m_omega, m_line) in methods.items():
        violations = 0
        infeasible = 0
        total_cost = 0
        valid_samples = 0
        
        reason_gen = 0
        reason_line = 0
        
        for t in range(len(test_forecast)):
            x_star, feasible, cost_val = solve_affine_dispatch(
                test_forecast[t], C_gen, H, data['gen_max'], data['branch_limit'],
                data['alpha'], m_omega, m_line, gen_prices
            )
            
            if not feasible:
                infeasible += 1
                violations += 1
                continue
            
            total_cost += cost_val
            valid_samples += 1
            
            # 实时校验
            real_noise_t = test_real[t] - test_forecast[t]
            omega_real = np.sum(real_noise_t)
            p_real = x_star + data['alpha'] * omega_real
            
            # 诊断具体原因
            is_gen_vio = np.any(p_real > data['gen_max'] + 1e-3) or np.any(p_real < -1e-3)
            
            inj_real = C_gen @ p_real - test_real[t]
            flow_real = H @ inj_real
            is_line_vio = np.any(np.abs(flow_real) > data['branch_limit'] + 1e-3)
            
            if is_gen_vio: reason_gen += 1
            if is_line_vio: reason_line += 1
            
            if is_gen_vio or is_line_vio:
                violations += 1
        
        rate = violations / len(test_forecast) * 100
        avg_cost = total_cost / valid_samples if valid_samples > 0 else 0.0
        
        print(f"{name:<15} | {rate:<8.2f} | {infeasible:<6} | {avg_cost:<10.0f} | {reason_gen:<6} | {reason_line:<6}")

if __name__ == '__main__':
    run_experiment()