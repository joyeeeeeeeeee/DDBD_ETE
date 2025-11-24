import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

# 全局双精度，防止数值误差导致梯度计算失败
torch.set_default_dtype(torch.float64)

def test_differentiability():
    print("--- 开始梯度检查 ---")
    
    # 1. 模拟极简数据 (2个节点，2个发电机)
    num_gens = 2
    num_buses = 2
    p_min = np.array([0.0, 0.0])
    p_max = np.array([100.0, 100.0])
    
    # 简单的 PTDF 和 Cg
    # Gen1 在 Bus1, Gen2 在 Bus2
    Cg_np = np.eye(2) 
    # Line1 连接 Bus1->Bus2
    PTDF_np = np.array([[0.5, -0.5]]) 
    F_limits = np.array([50.0])

    # 2. 构建可微层 (带松弛变量)
    x_var = cp.Variable(num_gens)
    s_bal = cp.Variable(1, nonneg=True) # 松弛
    
    # 参数
    c_param = cp.Parameter(num_gens)
    rhs_bal = cp.Parameter(1)
    
    # 简单约束: 供需平衡 + 松弛
    # sum(x) == load
    net_inj = np.ones(num_buses) @ Cg_np @ x_var
    constraints = [
        x_var >= p_min, x_var <= p_max,
        net_inj <= rhs_bal + s_bal,
        net_inj >= rhs_bal - s_bal
    ]
    
    # 目标: min c*x + penalty*s
    objective = cp.Minimize(c_param @ x_var + 1000.0 * cp.sum(s_bal))
    problem = cp.Problem(objective, constraints)
    
    print("-> 初始化 CvxpyLayer...")
    cvx_layer = CvxpyLayer(problem, parameters=[c_param, rhs_bal], variables=[x_var, s_bal])
    
    # 3. 模拟神经网络输出 (Requires Grad!)
    # 假设神经网络预测价格 c = [10.0, 20.0]
    c_pred = torch.tensor([[10.0, 20.0]], requires_grad=True)
    
    # 模拟负荷需求 load = 50
    rhs_val = torch.tensor([[50.0]])
    
    print("-> 前向传播 (Solving OPF)...")
    try:
        # 求解
        x_star, s_val = cvx_layer(c_pred, rhs_val, solver_args={'solver': cp.SCS, 'eps': 1e-4})
        print(f"   最优调度 x*: {x_star.detach().numpy()}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return

    # 4. 模拟 Loss 并反向传播
    # 假设真实成本是 0 (为了让模型尽量降低成本)
    # Loss = sum(c_real * x_star) - 这里简化为 minimize x_star 本身
    loss = x_star.sum()
    
    print("-> 反向传播 (Backward)...")
    loss.backward()
    
    # 5. 检查梯度
    grad = c_pred.grad
    print(f"   输入价格的梯度: {grad}")
    
    if grad is not None and torch.norm(grad) > 1e-6:
        print("\n✅ Gradient Check PASSED! OPF 层是可微的。")
        print("   解释: 改变预测价格 (c_pred) 会导致调度 (x*) 变化，导数被成功计算。")
    else:
        print("\n❌ Gradient Check FAILED. 梯度丢失。")

if __name__ == '__main__':
    test_differentiability()