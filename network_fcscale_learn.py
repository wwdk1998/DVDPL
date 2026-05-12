import torch
import torch.nn as nn
from net_layer import NetLayer
"""使用CAD参数作为先验，质量和质心位置为CAD值倍，同时有连杆质量大小约束。"""
"""使用网络学习摩擦参数的量级"""

class Network(nn.Module):
    """惯性参数预测网络"""
    
    def __init__(self, dim, n_width=128, n_depth=1, b_init=0.1, 
                 Positive_epsilon=1e-5, activation='ReLU'):
        """
        Args:
            dim: 矩阵维度（惯性张量维度）
            n_width: 网络宽度
            n_depth: 隐藏层深度
            b_init: 偏置初始化值
            Positive_epsilon: 确保正定性的小值（未使用，保留兼容性）
            activation: 激活函数类型
        """
        super(Network, self).__init__()
        
        self.dim = dim
        self.n_width = n_width
        self.n_hidden = n_depth
        self.epsilon = Positive_epsilon
        
        # 单位矩阵
        self.register_buffer('eye_matrix', torch.eye(dim))
        
        # 加载CAD先验参数
        self._load_cad_parameters()
        
        # 惯性张量虚拟参数范围：
        self.theta_L_scale = 50.0
        # RPY欧拉角范围 [rad]
        self.theta_R_scale = 3.1415926  # π
        
        # 创建共享特征提取层
        self.layers = nn.ModuleList()
        self.layers.append(NetLayer(1, n_width, activation))
        for _ in range(1, n_depth):
            self.layers.append(NetLayer(n_width, n_width, activation))
        
        # 为12个连杆创建条件编码
        self.num_links = 12
        self.link_embedding_dim = 8
        self.link_embedding = nn.Embedding(self.num_links, self.link_embedding_dim)
        
        # 为12个连杆创建独立的输出层（输入包含共享特征+link_embedding）
        conditional_input_size = n_width + self.link_embedding_dim
        self.net_theta_m_list = nn.ModuleList()  # 质量虚拟参数 theta_m (1维)
        self.net_theta_L_list = nn.ModuleList()  # 惯性虚拟向量 theta_L (3维)
        self.net_theta_R_list = nn.ModuleList()  # RPY欧拉角 theta_R (3维)
        self.net_rm_list = nn.ModuleList()       # 质心位置 rm (3维)
        
        for _ in range(self.num_links):
            self.net_theta_m_list.append(NetLayer(conditional_input_size, 1, 'Tanh'))  # Tanh激活，forward中映射到[0,1]
            self.net_theta_L_list.append(NetLayer(conditional_input_size, 3, 'Tanh'))
            self.net_theta_R_list.append(NetLayer(conditional_input_size, 3, 'Tanh'))
            self.net_rm_list.append(NetLayer(conditional_input_size, dim, 'Tanh'))  # Tanh激活，forward中映射到[0,1]
        
        # 创建摩擦网络输出层（直接从共享特征输出，使用 n_depth 控制深度）
        self.net_fc_list = nn.ModuleList()
        # 增加中间层便于学习摩擦参数的非线性映射
        for _ in range(n_depth):
            self.net_fc_list.append(NetLayer(n_width, n_width, 'Tanh')) 
        fc_out = NetLayer(n_width, 18, 'Tanh')
        self.net_fc_list.append(fc_out)
        
        # 摩擦参数的尺度因子（可学习参数，使用log形式保证正值）
        # 顺序: [alpha_0, alpha_1, alpha_2, v_s, sigma_0, sigma_1]
        scale_vals_init = [500.0, 500.0, 500.0, 0.005, 1.0e7, 1.0e3] * 3
        # 初始化为log值，实际使用时通过exp还原
        log_scale_init = torch.log(torch.tensor(scale_vals_init, dtype=torch.float32))
        self.log_fc_scale = nn.Parameter(log_scale_init.view(1, 3, 6))

        # 初始化参数
        self._init_layers(b_init)
        
    def _init_layers(self, b_init):
        """初始化层参数"""
        # 初始化共享层的偏置
        for layer in self.layers:
            nn.init.constant_(layer.bias, b_init)
        
        # 初始化12个连杆的输出层偏置
        for i in range(self.num_links):
            nn.init.constant_(self.net_theta_m_list[i].bias, b_init)
            nn.init.constant_(self.net_theta_L_list[i].bias, b_init)
            nn.init.constant_(self.net_theta_R_list[i].bias, b_init)
            nn.init.constant_(self.net_rm_list[i].bias, b_init)
        
        # 初始化link embedding
        nn.init.normal_(self.link_embedding.weight, mean=0.0, std=0.1)
    
    def _load_cad_parameters(self):
        """定义CAD先验参数（12个连杆）"""
        # 质量 [kg] - 顺序：B211, B212, B221, B222, B411, B412, B421, B422, B511, B512, B521, B522
        m_cad = [
            130.06,  # 0: B211 (Main1)
            197.57,  # 1: B212 (Main2)
            27.24,   # 2: B221 (Cyl)
            6.66,    # 3: B222 (Rod)
            41.31,   # 4: B411 (Main1)
            14.85,   # 5: B412 (Main2)
            28.94,   # 6: B421 (Cyl)
            5.95,    # 7: B422 (Rod)
            29.10,   # 8: B511 (Main1)
            103.58,  # 9: B512 (Main2)
            11.22,   # 10: B521 (Cyl)
            2.61     # 11: B522 (Rod)
        ]
        
        # 质心位置 [m] - [x, y, z]
        rm_cad = [
            [0.0558, 0.0628, -0.0002],   # 0: B211
            [0.0125, 0.2022, -0.0003],   # 1: B212
            [0.2163, 0.0454, 0.0010],    # 2: B221
            [0.2989, 0.0000, 0.0000],    # 3: B222
            [0.2973, 0.0175, -0.0152],   # 4: B411
            [0.0835, -0.0113, -0.0191],  # 5: B412
            [0.2050, 0.0543, -0.0016],   # 6: B421
            [0.2528, 0.0000, 0.0000],    # 7: B422
            [0.1895, 0.0022, -0.0271],   # 8: B511
            [0.0001, 0.3167, -0.0683],   # 9: B512
            [0.1510, 0.0402, 0.0051],    # 10: B521
            [0.1691, 0.0000, -0.0007]    # 11: B522
        ]
        
        # 注册为buffer（不参与训练但会随模型移动到GPU）
        self.register_buffer('m_cad', torch.tensor(m_cad, dtype=torch.float32))  # [12]
        self.register_buffer('rm_cad', torch.tensor(rm_cad, dtype=torch.float32))  # [12, 3]
        
        # 设置约束范围
        self.cad_scale_min = 0.5
        self.cad_scale_max = 1.5
    
    def forward(self, batch_size, device='cuda'):
        """
        Args:
            batch_size: 批次大小
        Returns:
            IA_tensor: [batch_size, 12, 3, 3] 
            rm_tensor: [batch_size, 12, 3]   
            m_tensor:  [batch_size, 12]
            fc_tensor: [batch_size, 18] 摩擦参数（3个液压缸×6个参数）
        """
        # 共享特征提取（连杆参数用）
        u = torch.ones(batch_size, 1, device=device)
        y_shared = u
        for layer in self.layers:
            y_shared = layer(y_shared)

        # 初始化输出列表
        rm_list = []
        delta_m_list = []
        
        # ===== 第一步：计算质量增量和质心位置 =====
        for link_id in range(self.num_links):
            link_id_tensor = torch.tensor([link_id] * batch_size, device=device)
            link_emb = self.link_embedding(link_id_tensor)
            y_conditional = torch.cat([y_shared, link_emb], dim=1)
            
            # 1. 质量增量
            tanh_output = self.net_theta_m_list[link_id](y_conditional).squeeze(1)
            delta_m = (tanh_output + 1.0) / 2.0
            delta_m_list.append(delta_m)
            
            # 2. 质心位置
            rm_tanh = self.net_rm_list[link_id](y_conditional)
            rm_weight = (rm_tanh + 1.0) / 2.0
            rm_cad_val = self.rm_cad[link_id].unsqueeze(0).expand(batch_size, -1)
            scale_factor_rm = self.cad_scale_min + (self.cad_scale_max - self.cad_scale_min) * rm_weight
            rm = rm_cad_val * scale_factor_rm
            rm_list.append(rm)
        
        # ===== 第二步：质量约束和归一化 =====
        delta_m_tensor = torch.stack(delta_m_list, dim=1)  # [batch, 12]
        
        # 新的质量计算逻辑：基于CAD参数上下界的插值
        m_tensor = torch.zeros_like(delta_m_tensor)
        eps = self.epsilon  # 质量增量的最小间隔
        
        for group_indices in [[0,1,2,3], [4,5,6,7], [8,9,10,11]]:
            # 提取每组的索引：Main1, Main2, Cyl, Rod
            idx_main1, idx_main2, idx_cyl, idx_rod = group_indices
            
            # 提取delta值
            delta_rod = delta_m_tensor[:, idx_rod]
            delta_cyl = delta_m_tensor[:, idx_cyl]
            delta_main1 = delta_m_tensor[:, idx_main1]
            delta_main2 = delta_m_tensor[:, idx_main2]
            
            # 提取CAD质量
            m_cad_rod = self.m_cad[idx_rod]
            m_cad_cyl = self.m_cad[idx_cyl]
            m_cad_main1 = self.m_cad[idx_main1]
            m_cad_main2 = self.m_cad[idx_main2]
            
            # 计算质量（满足 Main1>Cyl>Rod, Main2>Rod）
            # m_rod = (self.cad_scale_min + delta_rod * (self.cad_scale_max - self.cad_scale_min)) * m_cad_rod
            # m_cyl = m_rod + eps + delta_cyl * (self.cad_scale_max * m_cad_cyl - torch.max(m_rod, self.cad_scale_min * m_cad_cyl))
            # m_main1 = m_cyl + eps + delta_main1 * (self.cad_scale_max * m_cad_main1 - torch.max(m_cyl, self.cad_scale_min * m_cad_main1))
            # m_main2 = m_rod + eps + delta_main2 * (self.cad_scale_max * m_cad_main2 - torch.max(m_rod, self.cad_scale_min * m_cad_main2))


            m_rod = (self.cad_scale_min + delta_rod * (self.cad_scale_max - self.cad_scale_min)) * m_cad_rod
            lower_cyl = torch.max(m_rod + eps, self.cad_scale_min * m_cad_cyl)
            m_cyl = lower_cyl + delta_cyl * (self.cad_scale_max * m_cad_cyl - lower_cyl)
            lower_main1 = torch.max(m_cyl + eps, self.cad_scale_min * m_cad_main1)
            m_main1 = lower_main1 + delta_main1 * (self.cad_scale_max * m_cad_main1 - lower_main1)
            lower_main2 = torch.max(m_rod + eps, self.cad_scale_min * m_cad_main2)
            m_main2 = lower_main2 + delta_main2 * (self.cad_scale_max * m_cad_main2 - lower_main2)


            # 赋值回张量
            m_tensor[:, idx_rod] = m_rod
            m_tensor[:, idx_cyl] = m_cyl
            m_tensor[:, idx_main1] = m_main1
            m_tensor[:, idx_main2] = m_main2
        
        
        # ===== 第三步：计算惯性张量 =====
        IA_list = []
        eye_3x3 = self.eye_matrix.unsqueeze(0).expand(batch_size, 3, 3)
        for link_id in range(self.num_links):
            link_id_tensor = torch.tensor([link_id] * batch_size, device=device)
            link_emb = self.link_embedding(link_id_tensor)
            y_conditional = torch.cat([y_shared, link_emb], dim=1)
            
            # 1. 惯性虚拟向量 theta_L
            theta_L = self.net_theta_L_list[link_id](y_conditional)
            theta_L = self.theta_L_scale * theta_L
            theta_L1, theta_L2, theta_L3 = theta_L[:, 0], theta_L[:, 1], theta_L[:, 2]
            
            # 2. 主惯性矩 I_p
            I_x = theta_L2 ** 2 + theta_L3 ** 2
            I_y = theta_L3 ** 2 + theta_L1 ** 2
            I_z = theta_L1 ** 2 + theta_L2 ** 2
            I_p = torch.zeros(batch_size, 3, 3, device=device)
            I_p[:, 0, 0] = I_x
            I_p[:, 1, 1] = I_y
            I_p[:, 2, 2] = I_z
            
            # 3. RPY欧拉角
            theta_R = self.net_theta_R_list[link_id](y_conditional) * self.theta_R_scale
            phi_x, phi_y, phi_z = theta_R[:, 0], theta_R[:, 1], theta_R[:, 2]
            
            # 4. 旋转矩阵
            cos_x, sin_x = torch.cos(phi_x), torch.sin(phi_x)
            R_x = torch.zeros(batch_size, 3, 3, device=device)
            R_x[:, 0, 0] = 1.0
            R_x[:, 1, 1] = cos_x
            R_x[:, 1, 2] = -sin_x
            R_x[:, 2, 1] = sin_x
            R_x[:, 2, 2] = cos_x
            
            cos_y, sin_y = torch.cos(phi_y), torch.sin(phi_y)
            R_y = torch.zeros(batch_size, 3, 3, device=device)
            R_y[:, 0, 0] = cos_y
            R_y[:, 0, 2] = sin_y
            R_y[:, 1, 1] = 1.0
            R_y[:, 2, 0] = -sin_y
            R_y[:, 2, 2] = cos_y
            
            cos_z, sin_z = torch.cos(phi_z), torch.sin(phi_z)
            R_z = torch.zeros(batch_size, 3, 3, device=device)
            R_z[:, 0, 0] = cos_z
            R_z[:, 0, 1] = -sin_z
            R_z[:, 1, 0] = sin_z
            R_z[:, 1, 1] = cos_z
            R_z[:, 2, 2] = 1.0
            
            R_I = torch.bmm(R_z, torch.bmm(R_y, R_x))
            
            # 5. 旋转惯性张量
            I_rotated = torch.bmm(R_I, torch.bmm(I_p, R_I.transpose(1, 2)))
            
            # 6. 平行移轴定理
            m = m_tensor[:, link_id]
            rm = rm_list[link_id]
            rm_norm_sq = torch.sum(rm ** 2, dim=1)
            rm_outer = torch.bmm(rm.unsqueeze(2), rm.unsqueeze(1))
            parallel_axis_term = m.view(batch_size, 1, 1) * (rm_norm_sq.view(batch_size, 1, 1) * eye_3x3 - rm_outer)
            
            IA = I_rotated + parallel_axis_term
            IA_list.append(IA)
        
        # ===== 摩擦参数计算 =====
        fc_feat = y_shared
        for i in range(len(self.net_fc_list) - 1):
            fc_feat = self.net_fc_list[i](fc_feat)
        fc_raw = self.net_fc_list[-1](fc_feat)  # [batch_size, 18]
        
        fc_reshaped = fc_raw.view(batch_size, 3, 6)
        fc_scale = torch.exp(self.log_fc_scale)
        fc_exp = torch.exp(fc_reshaped * 10.0)
        fc_scaled = fc_exp * fc_scale
        fc_tensor = fc_scaled.view(batch_size, 18)

        # 转换为张量
        IA_tensor = torch.stack(IA_list, dim=1)  # [batch_size, 12, 3, 3]
        rm_tensor = torch.stack(rm_list, dim=1)  # [batch_size, 12, 3]
        
        return IA_tensor, rm_tensor, m_tensor, fc_tensor

