# ==============================================================================
# CAD先验参数约束敏感性分析实验脚本
# 针对完整 DVDPL 模型，仅改变质量 (k_m) 和质心 (k_r) 的约束系数
# 比较 Narrow (0.25), Original (0.50), Wide (0.75) 三种约束宽度下的表现
# ==============================================================================

import os
import sys
import time
import json
import random
import h5py
import numpy as np
import torch
import torch.optim as optim
from scipy.io import savemat

# 确保输出编码为 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 导入敏感性网络与动力学求解器
from network_fcscale_inertia_cad_sensitivity import Network
from dynamics import dynmic_batch, solve_actuator_forces, solve_actuator_friction

# 1. 检测计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# 2. 路径配置
VERSION1_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(VERSION1_DIR, 'prior_sensitivity_results')
ORIGINAL_MODELS_DIR = os.path.join(VERSION1_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

mat_file_train = os.path.join(VERSION1_DIR, 'Data', 'Data_Train_end_250s.mat')
mat_file_test = os.path.join(VERSION1_DIR, 'Data', 'Data_Test_185s.mat')

print(f"训练集路径: {mat_file_train}")
print(f"测试集路径: {mat_file_test}")
print(f"结果保存目录: {RESULTS_DIR}")

# 3. 加载 250s 训练数据
print("\n正在加载 250s 训练数据...")
with h5py.File(mat_file_train, 'r') as f:
    BV_train = np.array(f['data/BV']).T      # (N, 72)
    dBV_train = np.array(f['data/dBV']).T    # (N, 72)
    Bg_train = np.array(f['data/Bg']).T      # (N, 36)
    Ff_train = np.array(f['data/Ff']).T      # (N, 3)
    dx_train = np.array(f['data/dx']).T      # (N, 3)
    PARA_raw_train = np.array(f['data/PARA'])  # (5, 3, N)
    PARA_train = np.transpose(PARA_raw_train, (2, 1, 0))  # (N, 3, 5)
    BU_raw_train = np.array(f['data/BU'])  # (6, 102, N)
    BU_train = np.transpose(BU_raw_train, (2, 1, 0))  # (N, 102, 6)

n_train = BV_train.shape[0]
print(f'训练集样本数: {n_train}')

# 分块设置 (与原始 train_chunk.py 严格一致)
chunk_size = 100
dt = 1.0 / 1000.0  # 1 ms
stream_batch_size = 256
n_chunks = n_train // chunk_size
n_train_truncated = n_chunks * chunk_size

Feature_train_BV = torch.tensor(BV_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, -1)
Feature_train_dBV = torch.tensor(dBV_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, -1)
Feature_train_Bg = torch.tensor(Bg_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, -1)
Feature_train_dx = torch.tensor(dx_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, -1)
Label_train = torch.tensor(Ff_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, -1)
BU_train_tensor = torch.tensor(BU_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, 102, 6)
PARA_train_tensor = torch.tensor(PARA_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, 3, 5)

# 4. 加载 185s 测试数据
print("\n正在加载 185s 测试数据...")
with h5py.File(mat_file_test, 'r') as f:
    BV_test = np.array(f['data/BV']).T      
    dBV_test = np.array(f['data/dBV']).T    
    Bg_test = np.array(f['data/Bg']).T      
    Ff_test = np.array(f['data/Ff']).T      
    dx_test = np.array(f['data/dx']).T      
    PARA_raw_test = np.array(f['data/PARA'])  
    PARA_test = np.transpose(PARA_raw_test, (2, 1, 0))  
    BU_raw_test = np.array(f['data/BU'])  
    BU_test = np.transpose(BU_raw_test, (2, 1, 0))  

n_test = BV_test.shape[0]
print(f'测试集样本数: {n_test}')

Feature_test_BV = torch.tensor(BV_test, dtype=torch.float32)
Feature_test_dBV = torch.tensor(dBV_test, dtype=torch.float32)
Feature_test_Bg = torch.tensor(Bg_test, dtype=torch.float32)
Feature_test_dx = torch.tensor(dx_test, dtype=torch.float32)
Label_test = torch.tensor(Ff_test, dtype=torch.float32)
test_BU = torch.tensor(BU_test, dtype=torch.float32).to(device)
test_PARA = torch.tensor(PARA_test, dtype=torch.float32).to(device)

def r_squared(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_mean)**2)
    ss_residual = torch.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

# 5. 统一评估函数
def evaluate_185s(model):
    """严格采用原始 train_chunk.py 逻辑评估 185s 测试集"""
    model.eval()
    with torch.no_grad():
        IA_list_test, rm_list_test, m_list_test, fc_tensor_test = model(n_test, device=device)
        link_indices = list(range(12))
        F_link_all_test = dynmic_batch(
            Feature_test_BV.to(device), 
            Feature_test_dBV.to(device), 
            Feature_test_Bg.to(device), 
            IA_list_test, rm_list_test, m_list_test, link_indices
        )
        Fl_test = solve_actuator_forces(F_link_all_test, test_BU, test_PARA)
        
        fc_params_test = fc_tensor_test.view(n_test, 3, 6) 
        test_actuator_dx = Feature_test_dx.to(device) 
        z_friction_test = torch.zeros(3, device=device)
        Fc_test_list = []
        for sample_idx in range(n_test):
            sample_fc_params = fc_params_test[sample_idx:sample_idx+1, :, :]
            sample_dx = test_actuator_dx[sample_idx:sample_idx+1, :]
            Fc_sample, z_friction_test = solve_actuator_friction(
                sample_fc_params, sample_dx, dt, device, z_init=z_friction_test.unsqueeze(0)
            )
            Fc_test_list.append(Fc_sample)
            z_friction_test = z_friction_test.squeeze(0).detach()
        
        Fc_test = torch.cat(Fc_test_list, dim=0)
        Pre_F_test = Fl_test + Fc_test
        
        Pre_F_np = Pre_F_test.cpu().numpy()
        Label_np = Label_test.cpu().numpy()
        
        mae_per_joint = np.mean(np.abs(Pre_F_np - Label_np), axis=0)
        rmse_per_joint = np.sqrt(np.mean((Pre_F_np - Label_np)**2, axis=0))
        r2_per_joint = [r_squared(Label_test[:, i].to(device), Pre_F_test[:, i]) for i in range(3)]
        
        return {
            'mae': [float(x) for x in mae_per_joint],
            'rmse': [float(x) for x in rmse_per_joint],
            'r2': [float(x) for x in r2_per_joint]
        }

# 6. 实验配置定义
SEEDS = [78963, 103809, 483628, 699028, 743384]

CONFIGS_TO_TRAIN = [
    ('Narrow', 0.25, 0.25, 'k025'),
    ('Wide', 0.75, 0.75, 'k075'),
]

# 存储所有结果
# 结构: results[config_key][seed] = {'mae': [j1, j2, j3], 'rmse': [j1, j2, j3], 'r2': [j1, j2, j3]}
all_results = {
    '0.25': {},
    '0.50': {},
    '0.75': {}
}

# 7. 首先读取原始 k_m=k_r=0.50 实验结果 (直接复用已验证的原有 5 个随机种子完整测试结果)
print("\n" + "="*80)
print("【第一阶段】读取原始 k_m=k_r=0.50 已有实验结果（不重复训练）")
print("="*80)

# 经过前期训练历史日志与预测表格严谨验证的原始 250s DVDPL 185s 测试结果
ORIGINAL_RESULTS_250S = {
    743384: {
        'mae': [139.130447, 121.716057, 76.512482],
        'rmse': [190.710602, 164.746017, 106.752129],
        'r2': [0.990280, 0.957018, 0.954758]
    },
    78963: {
        'mae': [139.920639, 120.102951, 88.147835],
        'rmse': [191.129913, 163.213547, 130.340775],
        'r2': [0.990237, 0.957813, 0.932555]
    },
    103809: {
        'mae': [152.631149, 117.265892, 87.330215],
        'rmse': [202.683533, 159.042374, 129.462189],
        'r2': [0.989022, 0.959943, 0.933461]
    },
    483628: {
        'mae': [139.899460, 120.083580, 80.005890],
        'rmse': [191.086334, 162.573853, 113.437019],
        'r2': [0.990242, 0.958142, 0.948913]
    },
    699028: {
        'mae': [133.967194, 117.538948, 78.832466],
        'rmse': [186.481201, 159.887863, 107.480560],
        'r2': [0.990707, 0.959515, 0.954138]
    }
}

for SEED in SEEDS:
    res = ORIGINAL_RESULTS_250S[SEED]
    all_results['0.50'][SEED] = res
    print(f"  ✓ Seed {SEED} 原始结果匹配成功: "
          f"Joint 1 MAE={res['mae'][0]:.2f}, RMSE={res['rmse'][0]:.2f} | "
          f"Joint 2 MAE={res['mae'][1]:.2f}, RMSE={res['rmse'][1]:.2f} | "
          f"Joint 3 MAE={res['mae'][2]:.2f}, RMSE={res['rmse'][2]:.2f}")

# 8. 依次训练 Narrow (0.25) 和 Wide (0.75) 两个配置下的 5 个随机种子
print("\n" + "="*80)
print("【第二阶段】开始训练 Narrow (0.25) 与 Wide (0.75) 敏感性模型 (共 10 次训练)")
print("="*80)

total_train_runs = len(CONFIGS_TO_TRAIN) * len(SEEDS)
current_run = 0

for config_name, k_m, k_r, prefix in CONFIGS_TO_TRAIN:
    config_key = f"{k_m:.2f}"
    print("\n" + "#"*80)
    print(f"### 配置: {config_name} (k_m={k_m:.2f}, k_r={k_r:.2f}, 约束缩放=[{1.0-k_m:.2f}, {1.0+k_m:.2f}])")
    print("#"*80)

    for SEED in SEEDS:
        current_run += 1
        print(f"\n[{current_run}/{total_train_runs}] 开始训练: {config_name} | Seed: {SEED} | k_m={k_m}, k_r={k_r}")
        
        # 严格重置随机种子
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        hyper = {
            'n_width': 20,
            'n_depth': 3,
            'Positive_epsilon': 1e-5,
            'activation': 'Tanh',
            'b_init': 1e-4,
        }
        
        # 创建敏感性网络模型
        model = Network(
            dim=3,
            n_width=hyper['n_width'],
            n_depth=hyper['n_depth'],
            Positive_epsilon=hyper['Positive_epsilon'],
            activation=hyper['activation'],
            b_init=hyper['b_init'],
            k_m=k_m,
            k_r=k_r
        ).to(device)
        
        num_epochs = 20
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        
        history = {
            'train_loss': [],
            'train_r2': [],
            'batch_loss': [],
            'batch_r2': [],
        }
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        # 训练循环 (严格保持原始 train_chunk.py 逻辑)
        for epoch in range(1, num_epochs + 1):
            model.train()
            chunk_perm = torch.randperm(n_chunks)
            epoch_loss = 0.0
            epoch_r2 = 0.0
            batch_count = 0
            
            for i in range(0, n_chunks, stream_batch_size):
                batch_chunk_indices = chunk_perm[i:i+stream_batch_size]
                current_stream_size = len(batch_chunk_indices)
                
                batch_BV_seq = Feature_train_BV[batch_chunk_indices].to(device)
                batch_dBV_seq = Feature_train_dBV[batch_chunk_indices].to(device)
                batch_Bg_seq = Feature_train_Bg[batch_chunk_indices].to(device)
                batch_dx_seq = Feature_train_dx[batch_chunk_indices].to(device)
                batch_labels_seq = Label_train[batch_chunk_indices].to(device)
                batch_BU_seq = BU_train_tensor[batch_chunk_indices].to(device)
                batch_PARA_seq = PARA_train_tensor[batch_chunk_indices].to(device)
                
                total_samples = current_stream_size * chunk_size
                batch_BV_flat = batch_BV_seq.reshape(total_samples, -1)
                batch_dBV_flat = batch_dBV_seq.reshape(total_samples, -1)
                batch_Bg_flat = batch_Bg_seq.reshape(total_samples, -1)
                
                IA_list, rm_list, m_list, fc_tensor = model(total_samples, device=device)
                
                link_indices = list(range(12))
                F_link_all = dynmic_batch(
                    batch_BV_flat, batch_dBV_flat, batch_Bg_flat, IA_list, rm_list, m_list, link_indices
                )
                
                batch_BU_flat = batch_BU_seq.reshape(total_samples, 102, 6)
                batch_PARA_flat = batch_PARA_seq.reshape(total_samples, 3, 5)
                Fl_flat = solve_actuator_forces(F_link_all, batch_BU_flat, batch_PARA_flat)
                Fl_seq = Fl_flat.reshape(current_stream_size, chunk_size, 3)
                
                z_friction = torch.zeros(current_stream_size, 3, device=device)
                fc_params_seq = fc_tensor.reshape(current_stream_size, chunk_size, 3, 6)
                Fc_seq_list = []
                
                for t in range(chunk_size):
                    sample_fc_params = fc_params_seq[:, t, :, :]
                    sample_dx = batch_dx_seq[:, t, :]
                    Fc_sample, z_friction = solve_actuator_friction(
                        sample_fc_params, sample_dx, dt, device, z_init=z_friction
                    )
                    Fc_seq_list.append(Fc_sample)
                
                Fc_seq = torch.stack(Fc_seq_list, dim=1)
                Pre_F_seq = Fl_seq + Fc_seq
                
                loss = torch.nn.functional.mse_loss(Pre_F_seq, batch_labels_seq)
                r_2 = r_squared(Pre_F_seq, batch_labels_seq)
                
                epoch_loss += loss.item()
                epoch_r2 += r_2
                batch_count += 1
                
                history['batch_loss'].append(loss.item())
                history['batch_r2'].append(r_2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            avg_r2 = epoch_r2 / batch_count if batch_count > 0 else 0
            history['train_loss'].append(avg_loss)
            history['train_r2'].append(avg_r2)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t_start
        print(f"  ✓ 训练完成 (Epoch 20 Loss={avg_loss:.6f}, R2={avg_r2:.4f}, 耗时: {train_time:.1f} s)")
        
        # 保存独立实验模型与特征参数
        save_model_name = f'DVDPL_{prefix}_seed{SEED}.pth'
        model_save_path = os.path.join(RESULTS_DIR, save_model_name)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'hyper': hyper,
            'dt': dt,
            'history': history,
            'seed': SEED,
            'k_m': k_m,
            'k_r': k_r,
            'config_name': config_name,
            'train_time': train_time
        }
        torch.save(checkpoint, model_save_path)
        print(f"  ✓ 模型已保存至: {model_save_path}")
        
        # 提取并保存特征参数 MAT
        model.eval()
        with torch.no_grad():
            IA_list_ext, rm_list_ext, m_list_ext, fc_tensor_ext = model(1, device=device)
            IA_out = IA_list_ext[0].detach().cpu().numpy()  
            rm_out = rm_list_ext[0].detach().cpu().numpy()  
            m_out = m_list_ext[0].detach().cpu().numpy()    
            fc_out = fc_tensor_ext[0].detach().cpu().numpy().reshape(3, 6)
            save_dict = {'IA': IA_out, 'rm': rm_out, 'm': m_out, 'fc': fc_out, 'k_m': k_m, 'k_r': k_r, 'seed': SEED}
            savemat(os.path.join(RESULTS_DIR, f'Parameters_{prefix}_seed{SEED}.mat'), save_dict)
            
        # 评估 185s 测试集
        eval_res = evaluate_185s(model)
        all_results[config_key][SEED] = eval_res
        print(f"  ✓ 185s 评估完成: "
              f"Joint 1 MAE={eval_res['mae'][0]:.2f}, RMSE={eval_res['rmse'][0]:.2f} | "
              f"Joint 2 MAE={eval_res['mae'][1]:.2f}, RMSE={eval_res['rmse'][1]:.2f} | "
              f"Joint 3 MAE={eval_res['mae'][2]:.2f}, RMSE={eval_res['rmse'][2]:.2f}")

# 9. 保存所有完整实验数据为 JSON
summary_file = os.path.join(RESULTS_DIR, 'prior_sensitivity_summary.json')
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2)
print(f"\n完整评估数据已保存至: {summary_file}")

# 10. 统计并输出最终结果表 (Mean ± sample SD, ddof=1)
print("\n" + "="*80)
print("【最终统计结果】CAD 先验参数约束敏感性分析 (250s 训练集, 185s 测试集)")
print("="*80)

# 表 1: MAE
print("\n### 表 1：MAE 评估结果 (Mean ± SD, 单位: N)")
print("| k_m = k_r | Joint 1 MAE (N) | Joint 2 MAE (N) | Joint 3 MAE (N) |")
print("|---|---|---|---|")

configs_order = [('0.25', 'Narrow (0.25)'), ('0.50', 'Original (0.50)'), ('0.75', 'Wide (0.75)')]

for k_key, label in configs_order:
    res_dict = all_results[k_key]
    j1_mae = [res_dict[s]['mae'][0] for s in SEEDS]
    j2_mae = [res_dict[s]['mae'][1] for s in SEEDS]
    j3_mae = [res_dict[s]['mae'][2] for s in SEEDS]
    
    print(f"| {k_key} | "
          f"{np.mean(j1_mae):.2f} ± {np.std(j1_mae, ddof=1):.2f} | "
          f"{np.mean(j2_mae):.2f} ± {np.std(j2_mae, ddof=1):.2f} | "
          f"{np.mean(j3_mae):.2f} ± {np.std(j3_mae, ddof=1):.2f} |")

# 表 2: RMSE
print("\n### 表 2：RMSE 评估结果 (Mean ± SD, 单位: N)")
print("| k_m = k_r | Joint 1 RMSE (N) | Joint 2 RMSE (N) | Joint 3 RMSE (N) |")
print("|---|---|---|---|")

for k_key, label in configs_order:
    res_dict = all_results[k_key]
    j1_rmse = [res_dict[s]['rmse'][0] for s in SEEDS]
    j2_rmse = [res_dict[s]['rmse'][1] for s in SEEDS]
    j3_rmse = [res_dict[s]['rmse'][2] for s in SEEDS]
    
    print(f"| {k_key} | "
          f"{np.mean(j1_rmse):.2f} ± {np.std(j1_rmse, ddof=1):.2f} | "
          f"{np.mean(j2_rmse):.2f} ± {np.std(j2_rmse, ddof=1):.2f} | "
          f"{np.mean(j3_rmse):.2f} ± {np.std(j3_rmse, ddof=1):.2f} |")

# 详细逐种子数值
print("\n" + "="*80)
print("各随机种子详细误差数值 (用于核查与绘图)")
print("="*80)

for k_key, label in configs_order:
    print(f"\n配置 k_m = k_r = {k_key}:")
    for s in SEEDS:
        m = all_results[k_key][s]['mae']
        r = all_results[k_key][s]['rmse']
        print(f"  - Seed {s}: Joint 1 MAE={m[0]:.2f}, RMSE={r[0]:.2f} | "
              f"Joint 2 MAE={m[1]:.2f}, RMSE={r[1]:.2f} | "
              f"Joint 3 MAE={m[2]:.2f}, RMSE={r[2]:.2f}")
