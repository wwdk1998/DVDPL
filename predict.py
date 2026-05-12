# 独立预测模块 - 加载训练好的模型进行预测
import torch
import h5py
import sys
import os
import numpy as np
import pandas as pd
from network_fcscale_learn import Network
from dynamics import dynmic_batch, solve_actuator_forces, solve_actuator_friction
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 选择预测的模型和测试工况
PREDICTIONS_TO_RUN = [
    # (模型类型, 工况类型)
    # ('250s', '185s'),
    ('750s', '185s'),
    # ('250s', '96s'), 
    # ('750s', '96s'), 
]

def r_squared(y_true, y_pred):
    y_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_mean)**2)
    ss_residual = torch.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

def run_prediction(model_type, test_type, seed):
    print(f"\n[{model_type} 模型 - {test_type} 工况 - seed={seed}] 预测开始...")
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    
    # 1. 确定测试数据路径和输出文件名
    if test_type == '185s':
        mat_file_test = r'E:\WQ\PHD\SCI_Write\VDC_DL\Code\Matlab\Data_Test_185s.mat'
        if model_type == '250s':
            out_excel = f'VDCDL_Test1_185s_{seed}.xlsx'
        elif model_type == '750s':
            out_excel = f'VDCDL_Test2_185s_{seed}.xlsx'
        else:
            out_excel = f'VDCDL_Test_{model_type}_{test_type}_{seed}.xlsx'
    elif test_type == '96s':
        mat_file_test = r'E:\WQ\PHD\SCI_Write\VDC_DL\Code\Matlab\Data_Test_draw8_96s.mat'
        if model_type == '250s':
            out_excel = f'VDCDL_Test3_96s_{seed}.xlsx'
        elif model_type == '750s':
            out_excel = f'VDCDL_Test4_96s_{seed}.xlsx'
        else:
            out_excel = f'VDCDL_Test3_96s_{model_type}_{seed}.xlsx'
    else:
        raise ValueError("不支持的测试工况")

    model_path = f'models/VDC_DL_model_{model_type}_{seed}.pth'

    if not os.path.exists(model_path):
        print(f"找不到模型文件 {model_path}，请先训练！")
        return

    # ==================== 加载模型 ====================
    print(f'正在加载模型 {model_path}...')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 提取超参数和配置
    hyper = checkpoint['hyper']
    dt = checkpoint['dt']
    
    # 重建模型结构
    model = Network(
        dim=3,
        n_width=hyper['n_width'],
        n_depth=hyper['n_depth'],
        Positive_epsilon=hyper['Positive_epsilon'],
        activation=hyper['activation'],
        b_init=hyper['b_init']
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ==================== 加载测试数据 ====================
    print(f'正在加载测试数据 {mat_file_test}...')
    if not os.path.exists(mat_file_test):
        print(f"找不到测试集文件 {mat_file_test}，请检查路径！")
        return
        
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
    
    # 转换为张量
    Feature_test_BV = torch.tensor(BV_test, dtype=torch.float32)
    Feature_test_dBV = torch.tensor(dBV_test, dtype=torch.float32)
    Feature_test_Bg = torch.tensor(Bg_test, dtype=torch.float32)
    Feature_test_dx = torch.tensor(dx_test, dtype=torch.float32)
    Label_test = torch.tensor(Ff_test, dtype=torch.float32)

    # ==================== 预测 ====================
    print('开始计算...')
    with torch.no_grad():
        # 测试集前向传播
        IA_list_test, rm_list_test, m_list_test, fc_tensor_test = model(n_test, device=device)
        
        # 计算所有12个连杆的动力学
        link_indices = list(range(12))
        F_link_all_test = dynmic_batch(
            Feature_test_BV.to(device), 
            Feature_test_dBV.to(device), 
            Feature_test_Bg.to(device), 
            IA_list_test, rm_list_test, m_list_test, link_indices
        )
        
        # 反向迭代求解执行器力
        test_BU = torch.tensor(BU_test, dtype=torch.float32).to(device)
        test_PARA = torch.tensor(PARA_test, dtype=torch.float32).to(device)
        Fl_test = solve_actuator_forces(F_link_all_test, test_BU, test_PARA)
        
        # 计算测试集摩擦力
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
        
        # ==================== 保存预测结果表格 ====================
        Pre_F_np = Pre_F_test.cpu().numpy()
        Label_np = Label_test.cpu().numpy()
        
        joint_names = ['F2', 'F4', 'F5']
        
        df_list = []
        for i in range(3):
            df_temp = pd.DataFrame({
                f'{joint_names[i]}': Pre_F_np[:, i],
            })
            df_list.append(df_temp)
        
        df_results = pd.concat(df_list, axis=1)
        
        # 保存为 Excel 文件
        save_dir = r'E:\WQ\PHD\SCI_Write\VDC_DL\Code\Matlab\Plot'
        os.makedirs(save_dir, exist_ok=True)
        excel_path = os.path.join(save_dir, out_excel)
        df_results.to_excel(excel_path, index=False, engine='openpyxl')
        print(f'预测结果已保存为 Excel 文件: {excel_path}')

        # 评估指标
        mae_per_joint = np.mean(np.abs(Pre_F_np - Label_np), axis=0)
        rmse_per_joint = np.sqrt(np.mean((Pre_F_np - Label_np)**2, axis=0))
        r2_per_joint = []
        for i in range(3):
            r2_i = r_squared(Label_test[:, i].to(device), Pre_F_test[:, i])
            r2_per_joint.append(r2_i)
            
        print("="*60)
        actuator_names = ['执行器2', '执行器4', '执行器5']
        for i in range(3):
            print(f"  {actuator_names[i]}: MAE={mae_per_joint[i]:.6f}, RMSE={rmse_per_joint[i]:.6f}, R2={r2_per_joint[i]:.6f}")
        print("="*60)

        # ==================== 可视化 ====================
        # plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        # plt.rcParams['axes.unicode_minus'] = False
        
        # for i in range(3):
        #     fig, ax = plt.subplots(figsize=(14, 7))
        #     ax.plot(Pre_F_np[:, i], label='VDC_DL 预测', linewidth=2, alpha=0.9, color='blue')
        #     ax.plot(Label_np[:, i], label='真实值', linewidth=1.5, alpha=0.7, color='red', linestyle='--')
        #     ax.legend(fontsize=11, loc='upper right')
        #     ax.set_title(f"预测结果展示 - {out_excel} - {actuator_names[i]}\nMAE={mae_per_joint[i]:.4f}, RMSE={rmse_per_joint[i]:.4f}, R2={r2_per_joint[i]:.4f}")
        #     ax.set_xlabel('样本索引')
        #     ax.set_ylabel(f'{actuator_names[i]} 输出力 (N)')
        #     ax.grid(True, alpha=0.3)
        #     fig.tight_layout()

if __name__ == '__main__':
    # 预设的5个随机数种子
    SEEDS = [743384, 78963, 103809, 483628, 699028]
    
    for seed in SEEDS:
        for model_type, test_type in PREDICTIONS_TO_RUN:
            run_prediction(model_type, test_type, seed=seed)
    
    print('\n所有种子和工况的预测任务已完成。')
    # plt.show() # 如需查看所有图表，取消注释此行






