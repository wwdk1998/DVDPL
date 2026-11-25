import torch
import torch.nn as nn
from net_layer import NetLayer
"""Fixed消融实验网络：刚体参数固定为CAD值，仅学习LuGre摩擦参数。"""
"""按照Full模型完全一致的顺序创建网络模块，保证随机权重初始化完全对齐；刚体模块显式冻结且不参与forward。"""

class Network(nn.Module):
    """固定刚体CAD参数，仅预测摩擦参数的网络"""
    
    def __init__(self, dim, n_width=128, n_depth=1, b_init=0.1, 
                 Positive_epsilon=1e-5, activation='ReLU'):
        """
        Args:
            dim: 矩阵维度（惯性张量维度，3）
            n_width: 网络宽度
            n_depth: 隐藏层深度
            b_init: 偏置初始化值
            Positive_epsilon: 确保正定性的小值（保留接口兼容性）
            activation: 激活函数类型
        """
        super(Network, self).__init__()
        
        self.dim = dim
        self.n_width = n_width
        self.n_hidden = n_depth
        self.epsilon = Positive_epsilon
        
        # 1. 单位矩阵
        self.register_buffer('eye_matrix', torch.eye(dim))
        
        # 2. 加载CAD先验参数（同时预计算固定IA_cad）
        self._load_cad_parameters()
        
        # 3. RPY欧拉角范围 [rad]
        self.theta_R_scale = 3.1415926  # π
        
        # 4. 创建共享特征提取层（与 Full 模型完全一致）
        self.layers = nn.ModuleList()
        self.layers.append(NetLayer(1, n_width, activation))
        for _ in range(1, n_depth):
            self.layers.append(NetLayer(n_width, n_width, activation))
        
        # 5. 为12个连杆创建条件编码（保持与 Full 模型完全一致的随机数消耗顺序）
        self.num_links = 12
        self.link_embedding_dim = 8
        self.link_embedding = nn.Embedding(self.num_links, self.link_embedding_dim)
        
        # 6. 为12个连杆创建独立的输出层（保持与 Full 模型完全一致的随机数消耗顺序）
        conditional_input_size = n_width + self.link_embedding_dim
        self.net_theta_m_list = nn.ModuleList()
        self.net_theta_L_list = nn.ModuleList()
        self.net_theta_R_list = nn.ModuleList()
        self.net_rm_list = nn.ModuleList()
        
        for _ in range(self.num_links):
            self.net_theta_m_list.append(NetLayer(conditional_input_size, 1, 'Tanh'))
            self.net_theta_L_list.append(NetLayer(conditional_input_size, 3, 'Tanh'))
            self.net_theta_R_list.append(NetLayer(conditional_input_size, 3, 'Tanh'))
            self.net_rm_list.append(NetLayer(conditional_input_size, dim, 'Tanh'))
            
        # 7. 创建摩擦网络输出层（与 Full 模型完全一致）
        self.net_fc_list = nn.ModuleList()
        for _ in range(n_depth):
            self.net_fc_list.append(NetLayer(n_width, n_width, 'Tanh')) 
        fc_out = NetLayer(n_width, 18, 'Tanh')
        self.net_fc_list.append(fc_out)
        
        # 8. 摩擦参数的尺度因子（与 Full 模型完全一致）
        # 顺序: [alpha_0, alpha_1, alpha_2, v_s, sigma_0, sigma_1]
        scale_vals_init = [500.0, 500.0, 500.0, 0.005, 1.0e7, 1.0e3] * 3
        log_scale_init = torch.log(torch.tensor(scale_vals_init, dtype=torch.float32))
        self.log_fc_scale = nn.Parameter(log_scale_init.view(1, 3, 6))

        # 9. 初始化参数（与 Full 模型完全一致）
        self._init_layers(b_init)
        
        # 10. 冻结刚体参数相关模块（不参与训练与优化）
        for p in self.link_embedding.parameters():
            p.requires_grad = False
        for net_list in [self.net_theta_m_list, self.net_theta_L_list, self.net_theta_R_list, self.net_rm_list]:
            for layer in net_list:
                for p in layer.parameters():
                    p.requires_grad = False
        
    def _init_layers(self, b_init):
        """初始化层参数（调用顺序与 Full 模型完全一致）"""
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
        """定义CAD先验参数（12个连杆），严格保持CAD固定值"""
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

        # 惯性张量 [kg*m^2] - 质心处惯性张量 (12个连杆)
        I_cad = [
            # 0: B211
            [[2.623, 0.008, 0.018],
             [0.008, 16.607, -0.005],
             [0.018, -0.005, 16.983]],
            # 1: B212
            [[11.474, -0.435, -0.015],
             [-0.435, 4.322, -0.012],
             [-0.015, -0.012, 13.979]],
            # 2: B221
            [[0.188, -0.150, 0.004],
             [-0.150, 2.006, 0.005],
             [0.004, 0.005, 2.155]],
            # 3: B222
            [[0.003, 0.0, 0.0],
             [0.0, 0.859, 0.0],
             [0.0, 0.0, 0.859]],
            # 4: B411
            [[0.786, 0.0001, -0.167],
             [0.0001, 6.213, -0.026],
             [-0.167, -0.026, 5.893]],
            # 5: B412
            [[0.103, 0.004, -0.017],
             [0.004, 0.245, 0.007],
             [-0.017, 0.007, 0.187]],
            # 6: B421
            [[0.223, -0.279, 0.010],
             [-0.279, 1.730, 0.007],
             [0.010, 0.007, 1.914]],
            # 7: B422
            [[0.002, 0.0, 0.0],
             [0.0, 0.552, 0.0],
             [0.0, 0.0, 0.553]],
            # 8: B511
            [[0.285, 0.042, -0.090],
             [0.042, 1.584, -0.007],
             [-0.090, -0.007, 1.489]],
            # 9: B512
            [[15.055, -0.016, 0.0001],
             [-0.016, 1.116, -1.977],
             [0.0001, -1.977, 14.466]],
            # 10: B521
            [[0.050, 0.072, 0.012],
             [0.072, 0.324, 0.004],
             [0.012, 0.004, 0.348]],
            # 11: B522
            [[0.001, 0.0, -0.001],
             [0.0, 0.105, 0.0],
             [-0.001, 0.0, 0.105]]
        ]
        
        # 注册为buffer
        self.register_buffer('m_cad', torch.tensor(m_cad, dtype=torch.float32))  # [12]
        self.register_buffer('rm_cad', torch.tensor(rm_cad, dtype=torch.float32))  # [12, 3]
        self.register_buffer('I_cad', torch.tensor(I_cad, dtype=torch.float32))  # [12, 3, 3]
        
        # 预先根据平行移轴定理计算精确的固定连杆参考点惯性张量 IA_cad
        # IA = I_cad + m * (||rm||^2 * E - rm * rm^T)
        IA_cad_list = []
        eye_3x3 = self.eye_matrix
        for i in range(12):
            m = self.m_cad[i]
            rm = self.rm_cad[i]
            I_c = self.I_cad[i]
            rm_norm_sq = torch.sum(rm ** 2)
            rm_outer = torch.outer(rm, rm)
            parallel_axis_term = m * (rm_norm_sq * eye_3x3 - rm_outer)
            IA = I_c + parallel_axis_term
            IA_cad_list.append(IA)
            
        self.register_buffer('IA_cad', torch.stack(IA_cad_list, dim=0))  # [12, 3, 3]
    
    def forward(self, batch_size, device='cuda'):
        """
        Args:
            batch_size: 批次大小
        Returns:
            IA_tensor: [batch_size, 12, 3, 3] (固定CAD真值，不含梯度)
            rm_tensor: [batch_size, 12, 3]    (固定CAD真值，不含梯度)
            m_tensor:  [batch_size, 12]       (固定CAD真值，不含梯度)
            fc_tensor: [batch_size, 18] 摩擦参数（3个液压缸×6个参数，可学习）
        """
        # 共享特征提取（摩擦网络使用）
        u = torch.ones(batch_size, 1, device=device)
        y_shared = u
        for layer in self.layers:
            y_shared = layer(y_shared)

        # 刚体参数直接取 CAD 固定值并扩展维度（完全不进入计算图，无梯度）
        m_tensor = self.m_cad.unsqueeze(0).expand(batch_size, -1)
        rm_tensor = self.rm_cad.unsqueeze(0).expand(batch_size, -1, -1)
        IA_tensor = self.IA_cad.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 摩擦参数计算（与 Full 模型完全一致）
        fc_feat = y_shared
        for i in range(len(self.net_fc_list) - 1):
            fc_feat = self.net_fc_list[i](fc_feat)
        fc_raw = self.net_fc_list[-1](fc_feat)  # [batch_size, 18]
        
        fc_reshaped = fc_raw.view(batch_size, 3, 6)
        fc_scale = torch.exp(self.log_fc_scale)
        fc_exp = torch.exp(fc_reshaped * 10.0)
        fc_scaled = fc_exp * fc_scale
        fc_tensor = fc_scaled.view(batch_size, 18)
        
        return IA_tensor, rm_tensor, m_tensor, fc_tensor
