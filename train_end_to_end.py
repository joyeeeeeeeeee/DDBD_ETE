import torch
import torch.optim as optim
import numpy as np
from data_utils import load_mat_data
from constraint_model import calibrate_load_uncertainty, build_robust_constraints
from models import PricePredictorLSTM, DifferentiableOPFLayer, differentiable_conformal_calibration

# --- Configuration ---
BATCH_SIZE = 32
HIDDEN_DIM = 64
EPOCHS = 50
LR = 1e-3
# 如果数据已经变得比较“温和”（balance），这里的备用可以设得合理一些，比如 500 或 1000
TAU_MW = 1000.0 

def main():
    # 1. Load Data
    data = load_mat_data('./') 
    
    # 2. Constraint Robustness (Offline Phase)
    print("\n--- Phase 1: Calibrating Load Uncertainty ---")
    mu, Sigma_sqrt, rho = calibrate_load_uncertainty(
        data['d_forecast'], data['d_real']
    )
    
    # 计算原始鲁棒参数
    margin_sys, margin_lines, tau_eff = build_robust_constraints(
        data, mu, Sigma_sqrt, rho, tau=TAU_MW/data['baseMVA']
    )
    
    # ---【关键修复】处理不可行问题 (即使数据变好了，保留这个以防万一) ---
    
    # 1. 修复系统平衡约束 (Tau)
    if tau_eff < 0.01:
        print(f"WARNING: Calculated Tau_eff is low/negative ({tau_eff:.4f}). Forcing to 0.1 p.u.")
        tau_eff = 0.1 

    # 2. 修复线路约束 (F_eff)
    F_eff_raw = data['F_limits'] - margin_lines
    
    # 检查有多少条线路“爆”了
    neg_idx = np.where(F_eff_raw <= 0.01)[0] # 给一点余量 0.01
    if len(neg_idx) > 0:
        print(f"WARNING: {len(neg_idx)} lines have tight margins.")
        # 强制给 1MW (0.01 p.u.) 的物理通道，保证数学上有解
        F_eff_raw = np.maximum(F_eff_raw, 0.01)
        
    # --------------------------------
    
    # Prepare margin dictionary for Layer
    margins = {
        'sys': margin_sys,
        'F_eff': torch.tensor(F_eff_raw, dtype=torch.float32),
        'tau_eff': torch.tensor(tau_eff, dtype=torch.float32)
    }
    
    # 3. Setup Models
    num_gens = data['C_gen'].shape[1]
    feature_dim = data['X'].shape[1]
    
    predictor = PricePredictorLSTM(input_dim=feature_dim, hidden_dim=HIDDEN_DIM, num_gens=num_gens)
    opf_layer = DifferentiableOPFLayer(data, mu)
    
    optimizer = optim.Adam(predictor.parameters(), lr=LR)
    
    # 4. Training Loop
    X_train = data['X']
    Y_train = data['Y'] 
    D_fcst_train = torch.tensor(data['d_forecast'], dtype=torch.float32)
    
    print("\n--- Phase 2: End-to-End Training ---")
    for epoch in range(EPOCHS):
        predictor.train()
        total_regret = 0
        valid_batches = 0 
        
        indices = torch.randperm(X_train.shape[0])
        
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            if len(batch_idx) < BATCH_SIZE: continue
            
            x_batch = X_train[batch_idx].unsqueeze(1) 
            y_real_batch = Y_train[batch_idx]
            d_fcst_batch = D_fcst_train[batch_idx]
            
            h_lo, h_hi = predictor(x_batch)
            q = differentiable_conformal_calibration(h_lo, h_hi, y_real_batch, alpha=0.1)
            c_robust = h_hi + q
            
            try:
                # 求解鲁棒调度
                x_star = opf_layer(c_robust, d_fcst_batch, margins)
                if torch.sum(x_star) == 0 and torch.max(x_star) == 0:
                    print(f"Skipping batch {i}: NaN detected in inputs.")
                    optimizer.zero_grad() # 清空可能存在的坏梯度
                    continue
                # 计算 Regret
                with torch.no_grad():
                    # 理想情况下的最优解
                    x_perfect = opf_layer(y_real_batch, d_fcst_batch, margins)
                
                realized_cost = torch.sum(x_star * y_real_batch, dim=1)
                ideal_cost = torch.sum(x_perfect * y_real_batch, dim=1)
                
                regret = torch.mean(realized_cost - ideal_cost)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.zero_grad()
                regret.backward()
                optimizer.step()
                
                total_regret += regret.item()
                valid_batches += 1
                
            except Exception as e:
                # 打印第一个错误方便调试，之后的跳过
                if valid_batches == 0 and i == 0:
                    print(f"First batch failed: {e}")
                continue
        
        if valid_batches > 0:
            avg_regret = total_regret / valid_batches
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Regret: ${avg_regret:.4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS}: No valid batches. (Try increasing TAU_MW or regenerating easier data)")

if __name__ == '__main__':
    main()