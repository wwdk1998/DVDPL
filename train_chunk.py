# 数据分块，拟合LUGRE动态摩擦模型
import torch
import torch.optim as optim
import h5py
import numpy as np
import random
import os
import json
from scipy.io import savemat
from network_fcscale_learn import Network
from dynamics import dynmic_batch, solve_actuator_forces, solve_actuator_friction

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')

# 选择 '250s' 或 '750s'
TRAIN_MODEL_TYPE = '750s'

# 数据读取 - 训练集
if TRAIN_MODEL_TYPE == '750s':
    mat_file_train = r'Data\Data_Train_750s.mat'
elif TRAIN_MODEL_TYPE == '250s':
    mat_file_train = r'Data\Data_Train_end_250s.mat'
else:
    raise ValueError("未知的模型类型")

print(f"[{TRAIN_MODEL_TYPE} 模型] 训练集：{mat_file_train}")

with h5py.File(mat_file_train, 'r') as f:
    BV_train = np.array(f['data/BV']).T      # 转置：(72, N) -> (N, 72)
    dBV_train = np.array(f['data/dBV']).T    # 转置：(72, N) -> (N, 72)
    Bg_train = np.array(f['data/Bg']).T      # 转置：(36, N) -> (N, 36)
    Ff_train = np.array(f['data/Ff']).T      # 转置：(3, N) -> (N, 3)
    dx_train = np.array(f['data/dx']).T      # 转置：(3, N) -> (N, 3) - 3个执行器的速度
    PARA_raw_train = np.array(f['data/PARA'])  # (5, 3, N)
    PARA_train = np.transpose(PARA_raw_train, (2, 1, 0))  # (5, 3, N) -> (N, 3, 5)
    BU_raw_train = np.array(f['data/BU'])  # (6, 102, N)
    BU_train = np.transpose(BU_raw_train, (2, 1, 0))  # (6, 102, N) -> (N, 102, 6)

# 数据集大小
n_train = BV_train.shape[0]
print(f'训练集样本数：{n_train}')

# 训练集特征
Feature_train_BV = torch.tensor(BV_train, dtype=torch.float32)
Feature_train_dBV = torch.tensor(dBV_train, dtype=torch.float32)
Feature_train_Bg = torch.tensor(Bg_train, dtype=torch.float32)
Feature_train_dx = torch.tensor(dx_train, dtype=torch.float32)

# 标签
Label_train = torch.tensor(Ff_train, dtype=torch.float32)

# ==================== 序列分块设置 ====================
chunk_size = 100  # 每块100个样本
dt = 1.0 / 1000.0  # 采样周期
stream_batch_size = 256 # 并行处理的块数量（相当于Batch Size）

# 将训练数据重塑为 (n_chunks, chunk_size, features)
n_chunks = n_train // chunk_size
n_train_truncated = n_chunks * chunk_size

print(f'数据分块：{n_chunks}个块，每块{chunk_size}个样本，并行流Batch={stream_batch_size}')

# 截断数据以适应块大小
Feature_train_BV = Feature_train_BV[:n_train_truncated].view(n_chunks, chunk_size, -1)
Feature_train_dBV = Feature_train_dBV[:n_train_truncated].view(n_chunks, chunk_size, -1)
Feature_train_Bg = Feature_train_Bg[:n_train_truncated].view(n_chunks, chunk_size, -1)
Feature_train_dx = Feature_train_dx[:n_train_truncated].view(n_chunks, chunk_size, -1)
Label_train = Label_train[:n_train_truncated].view(n_chunks, chunk_size, -1)
BU_train_tensor = torch.tensor(BU_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, 102, 6)
PARA_train_tensor = torch.tensor(PARA_train[:n_train_truncated], dtype=torch.float32).view(n_chunks, chunk_size, 3, 5)

def r_squared(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_mean)**2)
    ss_residual = torch.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

# ==================== 多随机种子循环训练 ====================
SEEDS_TO_TRAIN = [743384, 78963, 103809, 483628, 699028]

for SEED in SEEDS_TO_TRAIN:
    print('\n' + '='*80)
    print(f'开始训练，当前随机种子: {SEED}')
    print('='*80 + '\n')
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # 以下设置可能降低性能，但保证完全可重复
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 网络超参数
    hyper = {
        'n_width': 20,
        'n_depth': 3,
        'Positive_epsilon': 1e-5,
        'activation': 'Tanh',
        'b_init': 1e-4,
    }
    
    # 创建模型
    model = Network(
        dim=3,
        n_width=hyper['n_width'],
        n_depth=hyper['n_depth'],
        Positive_epsilon=hyper['Positive_epsilon'],
        activation=hyper['activation'],
        b_init=hyper['b_init']
    ).to(device)
    
    # 模型信息
    print(f'model_device={next(model.parameters()).device}')
    
    # 训练参数
    num_epochs = 20
    learning_rate = 0.001
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 余弦退火衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = 20,  # 周期长度（总epoch数）
        eta_min=1e-5       # 最小学习率
    )
    
    # ==================== 训练历史记录 ====================
    history = {
        'train_loss': [],  # 每个epoch的平均训练loss
        'train_r2': [],    # 每个epoch的平均训练R2
        'batch_loss': [],  # 每个batch的loss（详细记录）
        'batch_r2': [],    # 每个batch的R2（详细记录）
    }
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch={epoch}/{num_epochs}')
        
        # 显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  learning_rate={current_lr:.6f}')
        
        model.train()
        
        # 打乱块的顺序（相当于打乱并行流的来源）
        chunk_perm = torch.randperm(n_chunks)
        
        epoch_loss = 0.0
        epoch_r2 = 0.0
        batch_count = 0
        
        # 按 stream_batch_size 遍历所有块
        for i in range(0, n_chunks, stream_batch_size):
            # 获取当前并行的块索引
            batch_chunk_indices = chunk_perm[i:i+stream_batch_size]   # 一次多块
            current_stream_size = len(batch_chunk_indices)
            
            # 提取当前Batch的数据: [current_stream_size, chunk_size, features]
            batch_BV_seq = Feature_train_BV[batch_chunk_indices].to(device)
            batch_dBV_seq = Feature_train_dBV[batch_chunk_indices].to(device)
            batch_Bg_seq = Feature_train_Bg[batch_chunk_indices].to(device)
            batch_dx_seq = Feature_train_dx[batch_chunk_indices].to(device)
            batch_labels_seq = Label_train[batch_chunk_indices].to(device)
            batch_BU_seq = BU_train_tensor[batch_chunk_indices].to(device)
            batch_PARA_seq = PARA_train_tensor[batch_chunk_indices].to(device)
            
            # 展平: [current_stream_size * chunk_size, features]
            total_samples = current_stream_size * chunk_size
            
            batch_BV_flat = batch_BV_seq.reshape(total_samples, -1)
            batch_dBV_flat = batch_dBV_seq.reshape(total_samples, -1)
            batch_Bg_flat = batch_Bg_seq.reshape(total_samples, -1)
             
            # 模型前向传播
            IA_list, rm_list, m_list, fc_tensor = model(total_samples, device=device)
            
            # 计算所有12个连杆的动力学
            link_indices = list(range(12))
            F_link_all = dynmic_batch(
                batch_BV_flat, batch_dBV_flat, batch_Bg_flat, IA_list, rm_list, m_list, link_indices
            )
            
            # 求解执行器力
            batch_BU_flat = batch_BU_seq.reshape(total_samples, 102, 6)
            batch_PARA_flat = batch_PARA_seq.reshape(total_samples, 3, 5)
            Fl_flat = solve_actuator_forces(F_link_all, batch_BU_flat, batch_PARA_flat) # [total_samples, 3]
            
            # 重塑回序列格式: [current_stream_size, chunk_size, 3]
            Fl_seq = Fl_flat.reshape(current_stream_size, chunk_size, 3)
    
            # 初始化摩擦状态 z: [current_stream_size, 3]
            z_friction = torch.zeros(current_stream_size, 3, device=device)
            
            # 摩擦参数重塑:[total_samples, 18] -> [current_stream_size, chunk_size, 3, 6]
            fc_params_seq = fc_tensor.reshape(current_stream_size, chunk_size, 3, 6)
            
            Fc_seq_list = []
            
            # 按时间步遍历 chunk_size
            for t in range(chunk_size):
                sample_fc_params = fc_params_seq[:, t, :, :] # [current_stream_size, 3, 6]
                sample_dx = batch_dx_seq[:, t, :]            # [current_stream_size, 3]
                
                # 计算摩擦力并更新z状态
                Fc_sample, z_friction = solve_actuator_friction(
                    sample_fc_params, sample_dx, dt, device, z_init=z_friction
                )
                
                Fc_seq_list.append(Fc_sample)
      
            # 堆叠结果
            Fc_seq = torch.stack(Fc_seq_list, dim=1)
               
            # 总预测力
            Pre_F_seq = Fl_seq  + Fc_seq
            
            # 计算损失 
            loss = torch.nn.functional.mse_loss(Pre_F_seq, batch_labels_seq)
            r_2 = r_squared(Pre_F_seq, batch_labels_seq)
    
            epoch_loss += loss.item()
            epoch_r2 += r_2
            batch_count += 1
            
            history['batch_loss'].append(loss.item())
            history['batch_r2'].append(r_2)
            
            if batch_count % 5 == 0:
                print(f'  Batch {batch_count}, loss={loss.item():.6f}, r2={r_2:.4f}')
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        avg_r2 = epoch_r2 / batch_count if batch_count > 0 else 0
        
        history['train_loss'].append(avg_loss)
        history['train_r2'].append(avg_r2)
        
        print(f'Epoch {epoch} completed - avg_loss={avg_loss:.6f}, avg_r2={avg_r2:.4f}, total_batches={batch_count}')
    
    print(f'\n当前种子 {SEED} 训练完成！')
    
    # ==================== 保存模型 ====================
    print('\n正在保存模型...')
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hyper': hyper,
        'dt': dt,
        'history': history,
        'seed': SEED,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    model_save_path = os.path.join(save_dir, f'VDC_DL_model_{TRAIN_MODEL_TYPE}_{SEED}.pth')
    torch.save(checkpoint, model_save_path)
    print(f'✓ 模型已保存为 {model_save_path}')
    
    history_serializable = {
        'train_loss': history['train_loss'],
        'train_r2': history['train_r2'],
        'batch_loss': history['batch_loss'],
        'batch_r2': history['batch_r2'],
        'seed': SEED,
        'num_epochs': num_epochs,
    }
    history_save_path = os.path.join(save_dir, f'VDC_DL_train_history_{TRAIN_MODEL_TYPE}_{SEED}.json')
    with open(history_save_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f'✓ 训练历史已保存为 json 文件: {history_save_path}')
    
    print('正在提取并保存特征参数...')
    model.eval()
    with torch.no_grad():
        IA_list_ext, rm_list_ext, m_list_ext, fc_tensor_ext = model(1, device=device)
        IA_out = IA_list_ext[0].detach().cpu().numpy()  
        rm_out = rm_list_ext[0].detach().cpu().numpy()  
        m_out = m_list_ext[0].detach().cpu().numpy()    
        fc_out = fc_tensor_ext[0].detach().cpu().numpy()
        fc_reshaped_out = fc_out.reshape(3, 6)
        save_dict = {'IA': IA_out, 'rm': rm_out, 'm': m_out, 'fc': fc_reshaped_out}
        param_mat_out = os.path.join(save_dir, f'Parameters_{TRAIN_MODEL_TYPE}_{SEED}.mat')
        savemat(param_mat_out, save_dict)
        print(f'✓ 模型特征参数已保存为 MAT 文件: {param_mat_out}\n')
    
    # ==================== 测试评估 ====================
    print('\n' + '='*80)
    print(f'开始使用刚训练好的模型对 185s 工况进行快速预测评估 (SEED={SEED})...')
    print('='*80 + '\n')
    model.eval()
    
    mat_file_test = r'Data\Data_Test_185s.mat'
    if not os.path.exists(mat_file_test):
        print(f"未找到预测数据 {mat_file_test}，跳过临时判断。")
    else:
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
        Feature_test_BV = torch.tensor(BV_test, dtype=torch.float32)
        Feature_test_dBV = torch.tensor(dBV_test, dtype=torch.float32)
        Feature_test_Bg = torch.tensor(Bg_test, dtype=torch.float32)
        Feature_test_dx = torch.tensor(dx_test, dtype=torch.float32)
        Label_test = torch.tensor(Ff_test, dtype=torch.float32)
    
        with torch.no_grad():
            IA_list_test, rm_list_test, m_list_test, fc_tensor_test = model(n_test, device=device)
            link_indices = list(range(12))
            F_link_all_test = dynmic_batch(
                Feature_test_BV.to(device), 
                Feature_test_dBV.to(device), 
                Feature_test_Bg.to(device), 
                IA_list_test, rm_list_test, m_list_test, link_indices
            )
            test_BU = torch.tensor(BU_test, dtype=torch.float32).to(device)
            test_PARA = torch.tensor(PARA_test, dtype=torch.float32).to(device)
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
            
            def r_squared_temp(y_true, y_pred):
                y_mean = torch.mean(y_true)
                ss_total = torch.sum((y_true - y_mean)**2)
                ss_residual = torch.sum((y_true - y_pred)**2)
                return (1 - (ss_residual / ss_total)).item()
                
            mae_per_joint = np.mean(np.abs(Pre_F_np - Label_np), axis=0)
            rmse_per_joint = np.sqrt(np.mean((Pre_F_np - Label_np)**2, axis=0))
            r2_per_joint = [r_squared_temp(Label_test[:, i].to(device), Pre_F_test[:, i]) for i in range(3)]
            
            actuator_names = ['执行器1', '执行器2', '执行器3']
            print("\n【临时185s验证结果】")
            for i in range(3):
                print(f"  {actuator_names[i]}: MAE={mae_per_joint[i]:.6f}, RMSE={rmse_per_joint[i]:.6f}, R2={r2_per_joint[i]:.6f}")
            print("="*60)
