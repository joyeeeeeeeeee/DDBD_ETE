# exp_final_compare.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 解决报错

import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from cvxpylayers.torch import CvxpyLayer
from utils_s4 import prepare_section4_data

# ==========================================
# 1. 复用之前的组件
# ==========================================
def create_opt_layer(n_gen, robust_params):
    # (与之前完全相同)
    x = cp.Variable(n_gen)
    c_robust = cp.Parameter(n_gen) 
    d_f = cp.Parameter(robust_params['H'].shape[1]) 
    
    alpha = robust_params['alpha']
    total_mu = robust_params['total_mu']
    omega = robust_params['omega_margin']
    gen_max = robust_params['gen_max']
    H = robust_params['H']
    C_gen = robust_params['C_gen']
    line_shift = robust_params['line_shift']
    branch_limit = robust_params['branch_limit']
    line_margin = robust_params['line_margin']
    
    constraints = [
        cp.sum(x) == cp.sum(d_f) + total_mu,
        x + alpha * total_mu + alpha * omega <= gen_max,
        x + alpha * total_mu - alpha * omega >= 0,
        H @ (C_gen @ x - d_f) + line_shift <= branch_limit - line_margin,
        H @ (C_gen @ x - d_f) + line_shift >= -branch_limit + line_margin
    ]
    epsilon = 1e-4
    objective = cp.Minimize(c_robust @ x + epsilon * cp.sum_squares(x))
    problem = cp.Problem(objective, constraints)
    return CvxpyLayer(problem, parameters=[c_robust, d_f], variables=[x])

class PricePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

# 简单的非可微校准 (用于 MSE 模型推理阶段)
def calibrate_standard(scores, alpha=0.1):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    sorted_scores, _ = torch.sort(scores, dim=0)
    return sorted_scores[k-1, :]

# 可微校准 (用于 E2E 训练)
def differentiable_conformal_layer(h_cal, y_cal, h_pred, alpha=0.1):
    scores = y_cal - h_cal 
    n_cal = scores.shape[0]
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    k = min(max(k, 1), n_cal)
    sorted_scores, _ = torch.sort(scores, dim=0)
    q = sorted_scores[k-1, :] 
    c_robust = h_pred + q.unsqueeze(0)
    return c_robust, q

# ==========================================
# 2. 对比实验主逻辑
# ==========================================
def run_full_comparison():
    # 准备数据
    X, Y, D, robust_params = prepare_section4_data(load_scale=0.85)
    
    n_samples = X.shape[0]
    n_train = int(n_samples * 0.7)
    
    X_train, Y_train, D_train = X[:n_train], Y[:n_train], D[:n_train]
    X_test, Y_test, D_test = X[n_train:], Y[n_train:], D[n_train:]
    
    n_gen = Y.shape[1]
    opt_layer = create_opt_layer(n_gen, robust_params)
    
    print(f"Data Loaded: Train={n_train}, Test={n_samples-n_train}")

    # ----------------------------------------
    # STEP 1: 计算 Perfect Information Lower Bound
    # ----------------------------------------
    print("\n[Baseline] Calculating Perfect Information Cost...")
    total_perfect_cost = 0
    valid_count = 0
    with torch.no_grad():
        for i in range(len(Y_test)):
            try:
                # 传入真实价格和负荷
                x_star, = opt_layer(Y_test[i:i+1], D_test[i:i+1])
                total_perfect_cost += torch.sum(x_star * Y_test[i:i+1]).item()
                valid_count += 1
            except: pass
    avg_perfect_cost = total_perfect_cost / valid_count
    print(f"  > Avg Perfect Cost: ${avg_perfect_cost:,.2f}")

    # ----------------------------------------
    # STEP 2: 训练 Traditional MSE Model
    # ----------------------------------------
    print("\n[Model 1] Training Traditional MSE Predictor...")
    model_mse = PricePredictor(X.shape[1], n_gen)
    opt_mse = optim.Adam(model_mse.parameters(), lr=1e-3)
    
    for epoch in range(30):
        model_mse.train()
        indices = torch.randperm(n_train)
        epoch_loss = 0
        for i in range(0, n_train, 64):
            idx = indices[i:i+64]
            pred = model_mse(X_train[idx])
            loss = nn.MSELoss()(pred, Y_train[idx])
            
            opt_mse.zero_grad()
            loss.backward()
            opt_mse.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0: print(f"  Epoch {epoch}: MSE Loss = {epoch_loss/n_train:.4f}")

    # ----------------------------------------
    # STEP 3: 训练 Ours (End-to-End)
    # ----------------------------------------
    print("\n[Model 2] Training End-to-End Regret Minimizer...")
    model_e2e = PricePredictor(X.shape[1], n_gen)
    opt_e2e = optim.Adam(model_e2e.parameters(), lr=5e-4) # 学习率稍小
    
    for epoch in range(30):
        model_e2e.train()
        indices = torch.randperm(n_train)
        epoch_regret = 0
        
        for i in range(0, n_train, 64):
            idx = indices[i:i+64]
            if len(idx) < 10: continue
            
            # Split Batch
            split = len(idx) // 2
            idx_cal, idx_opt = idx[:split], idx[split:]
            
            # Forward
            h_cal = model_e2e(X_train[idx_cal])
            h_opt = model_e2e(X_train[idx_opt])
            
            # Diff Layer
            c_robust, _ = differentiable_conformal_layer(
                h_cal, Y_train[idx_cal], h_opt, alpha=0.1
            )
            
            try:
                # Opt Layer
                x_sol, = opt_layer(c_robust, D_train[idx_opt])
                # Regret Loss
                loss = torch.sum(x_sol * Y_train[idx_opt]) / len(idx_opt)
                
                opt_e2e.zero_grad()
                loss.backward()
                opt_e2e.step()
                epoch_regret += loss.item()
            except: continue
            
        if epoch % 10 == 0: print(f"  Epoch {epoch}: Regret = ${epoch_regret:.2f}")

    # ----------------------------------------
    # STEP 4: 最终对比评估 (Final Evaluation)
    # ----------------------------------------
    print("\n=== Final Evaluation on Test Set ===")
    
    # 校准 MSE 模型 (Standard Conformal)
    with torch.no_grad():
        h_cal_mse = model_mse(X_train)
        scores_mse = Y_train - h_cal_mse
        q_mse = calibrate_standard(scores_mse, alpha=0.1)
        print(f"  > MSE Model Calibrated Q: {q_mse.mean():.2f}")
    
    # 校准 E2E 模型 (虽然训练时用了Batch，测试时用全量训练集校准更稳)
    with torch.no_grad():
        h_cal_e2e = model_e2e(X_train)
        scores_e2e = Y_train - h_cal_e2e
        q_e2e = calibrate_standard(scores_e2e, alpha=0.1)
        print(f"  > E2E Model Calibrated Q: {q_e2e.mean():.2f}")

    # 测试循环
    mse_costs = []
    e2e_costs = []
    
    with torch.no_grad():
        for i in range(len(Y_test)):
            d_in = D_test[i:i+1]
            y_real = Y_test[i:i+1]
            
            # MSE Strategy
            h_mse = model_mse(X_test[i:i+1])
            c_mse = h_mse + q_mse # Apply Q
            try:
                x_mse, = opt_layer(c_mse, d_in)
                cost = torch.sum(x_mse * y_real).item()
                mse_costs.append(cost)
            except: pass
            
            # E2E Strategy
            h_e2e = model_e2e(X_test[i:i+1])
            c_e2e = h_e2e + q_e2e # Apply Q
            try:
                x_e2e, = opt_layer(c_e2e, d_in)
                cost = torch.sum(x_e2e * y_real).item()
                e2e_costs.append(cost)
            except: pass

    avg_mse_cost = np.mean(mse_costs)
    avg_e2e_cost = np.mean(e2e_costs)
    
    regret_mse = avg_mse_cost - avg_perfect_cost
    regret_e2e = avg_e2e_cost - avg_perfect_cost
    reduction = (regret_mse - regret_e2e) / regret_mse * 100
    
    print("\n" + "="*50)
    print(f"RESULT SUMMARY (Avg Cost per Hour)")
    print("="*50)
    print(f"1. Perfect Information : ${avg_perfect_cost:,.2f}")
    print(f"2. Traditional MSE     : ${avg_mse_cost:,.2f} (Regret: ${regret_mse:,.2f})")
    print(f"3. Ours (End-to-End)   : ${avg_e2e_cost:,.2f} (Regret: ${regret_e2e:,.2f})")
    print("-" * 50)
    print(f"*** Regret Reduction   : {reduction:.2f}% ***")
    print("="*50)

if __name__ == '__main__':
    run_full_comparison()