import math
import torch
import torch.nn as nn
from net_layer import NetLayer


class Network(nn.Module):
    """NoCADPrior 消融网络：不使用 CAD 参数、CAD 派生边界或 CAD 派生增量。"""

    def __init__(self, dim, n_width=128, n_depth=1, b_init=0.1,
                 Positive_epsilon=1e-5, activation='ReLU'):
        super(Network, self).__init__()

        self.dim = dim
        self.n_width = n_width
        self.n_hidden = n_depth
        self.epsilon = Positive_epsilon
        self.register_buffer('eye_matrix', torch.eye(dim))

        self.num_links = 12
        self.link_embedding_dim = 8
        self.theta_R_scale = 1.0  # RPY 输出直接以 rad 表示，不施加边界
        # 通用质量尺度，仅改善参数数值尺度；不是 CAD 先验或训练边界。
        self.mass_scale = 50.0

        # 共享特征提取层
        self.layers = nn.ModuleList()
        self.layers.append(NetLayer(1, n_width, activation))
        for _ in range(1, n_depth):
            self.layers.append(NetLayer(n_width, n_width, activation))

        self.link_embedding = nn.Embedding(self.num_links, self.link_embedding_dim)
        conditional_input_size = n_width + self.link_embedding_dim

        # 所有物理参数头均为无界线性输出；正质量通过带通用尺度的平方参数化保证。
        self.net_theta_m_list = nn.ModuleList()
        self.net_theta_L_list = nn.ModuleList()
        self.net_theta_R_list = nn.ModuleList()
        self.net_rm_list = nn.ModuleList()

        for _ in range(self.num_links):
            self.net_theta_m_list.append(NetLayer(conditional_input_size, 1, 'Linear'))
            self.net_theta_L_list.append(NetLayer(conditional_input_size, 3, 'Linear'))
            self.net_theta_R_list.append(NetLayer(conditional_input_size, 3, 'Linear'))
            self.net_rm_list.append(NetLayer(conditional_input_size, dim, 'Linear'))

        # LuGre 摩擦分支保持与原网络一致
        self.net_fc_list = nn.ModuleList()
        for _ in range(n_depth):
            self.net_fc_list.append(NetLayer(n_width, n_width, 'Tanh'))
        self.net_fc_list.append(NetLayer(n_width, 18, 'Tanh'))

        scale_vals_init = [500.0, 500.0, 500.0, 0.005, 1.0e7, 1.0e3] * 3
        log_scale_init = torch.log(torch.tensor(scale_vals_init, dtype=torch.float32))
        self.log_fc_scale = nn.Parameter(log_scale_init.view(1, 3, 6))

        self._init_layers(b_init)

    def _init_layers(self, b_init):
        """随机初始化物理参数；以下范围只用于生成初始值，不是训练期间的参数边界。"""
        for layer in self.layers:
            nn.init.constant_(layer.bias, b_init)

        nn.init.normal_(self.link_embedding.weight, mean=0.0, std=0.1)

        # 通用随机初值分布，不读取或使用任何 CAD 参数。
        # 这些范围仅用于生成初始值，训练期间不限制参数取值。
        # 质量：log-uniform(2, 200) kg；用通用 mass_scale 归一化虚拟参数尺度。
        init_mass = torch.exp(
            torch.empty(self.num_links).uniform_(math.log(2.0), math.log(200.0))
        )
        init_theta_m = torch.sqrt(
            torch.clamp((init_mass - self.epsilon) / self.mass_scale, min=1e-12)
        )

        # 质心：采用较温和的通用局部坐标初值，训练时仍可无界移动。
        init_rm = torch.empty(self.num_links, self.dim).uniform_(-0.15, 0.15)

        # 惯性初值按随机质量和通用特征长度生成，避免惯性与质量尺度完全独立。
        # 特征长度仅用于初始化，不是 CAD 参数，也不是训练约束。
        init_length = torch.exp(
            torch.empty(self.num_links).uniform_(math.log(0.1), math.log(0.8))
        )
        inertia_shape = torch.exp(
            torch.empty(self.num_links, 3).uniform_(math.log(0.5), math.log(1.5))
        )
        init_theta_L = (
            torch.sqrt(init_mass).unsqueeze(1)
            * init_length.unsqueeze(1)
            * inertia_shape
            / math.sqrt(12.0)
        )

        # RPY 在靠近单位旋转的通用范围内随机初始化，以避开初始奇异姿态；
        # 训练期间仍不对角度施加边界。
        init_theta_R = torch.empty(self.num_links, 3).uniform_(
            -math.pi / 4.0, math.pi / 4.0
        )

        for i in range(self.num_links):
            for head in (self.net_theta_m_list[i], self.net_theta_L_list[i],
                         self.net_theta_R_list[i], self.net_rm_list[i]):
                # 较小的随机权重让初始物理量主要由每个连杆独立随机生成的偏置决定。
                nn.init.normal_(head.weight, mean=0.0, std=0.01)

            with torch.no_grad():
                self.net_theta_m_list[i].bias.copy_(init_theta_m[i:i+1])
                self.net_theta_L_list[i].bias.copy_(init_theta_L[i])
                self.net_theta_R_list[i].bias.copy_(init_theta_R[i])
                self.net_rm_list[i].bias.copy_(init_rm[i])

    def forward(self, batch_size, device='cuda'):
        # 共享特征提取
        u = torch.ones(batch_size, 1, device=device)
        y_shared = u
        for layer in self.layers:
            y_shared = layer(y_shared)

        rm_list = []
        m_list = []
        theta_L_list = []
        theta_R_list = []

        # 质量、质心、惯性方向参数均不依赖 CAD；不施加 CAD 边界或质量排序关系。
        for link_id in range(self.num_links):
            link_id_tensor = torch.full((batch_size,), link_id, dtype=torch.long, device=device)
            link_emb = self.link_embedding(link_id_tensor)
            y_conditional = torch.cat([y_shared, link_emb], dim=1)

            theta_m = self.net_theta_m_list[link_id](y_conditional).squeeze(1)
            m = self.mass_scale * theta_m ** 2 + self.epsilon
            m_list.append(m)

            rm = self.net_rm_list[link_id](y_conditional)
            rm_list.append(rm)

            theta_L = self.net_theta_L_list[link_id](y_conditional)
            theta_L_list.append(theta_L)
            theta_R = self.net_theta_R_list[link_id](y_conditional)
            theta_R_list.append(theta_R)

        m_tensor = torch.stack(m_list, dim=1)

        # 由虚拟惯性参数构造主惯量，保证正半定性与惯量三角不等式。
        IA_list = []
        eye_3x3 = self.eye_matrix.unsqueeze(0).expand(batch_size, 3, 3)
        for link_id in range(self.num_links):
            theta_L = theta_L_list[link_id]
            theta_L1, theta_L2, theta_L3 = theta_L[:, 0], theta_L[:, 1], theta_L[:, 2]

            I_x = theta_L2 ** 2 + theta_L3 ** 2
            I_y = theta_L3 ** 2 + theta_L1 ** 2
            I_z = theta_L1 ** 2 + theta_L2 ** 2
            I_p = torch.zeros(batch_size, 3, 3, device=device)
            I_p[:, 0, 0] = I_x
            I_p[:, 1, 1] = I_y
            I_p[:, 2, 2] = I_z

            # 无界 RPY 欧拉角；旋转矩阵仍通过三角函数保持合法旋转形式。
            phi_x, phi_y, phi_z = theta_R_list[link_id][:, 0], theta_R_list[link_id][:, 1], theta_R_list[link_id][:, 2]
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
            I_rotated = torch.bmm(R_I, torch.bmm(I_p, R_I.transpose(1, 2)))

            m = m_tensor[:, link_id]
            rm = rm_list[link_id]
            rm_norm_sq = torch.sum(rm ** 2, dim=1)
            rm_outer = torch.bmm(rm.unsqueeze(2), rm.unsqueeze(1))
            parallel_axis_term = m.view(batch_size, 1, 1) * (
                rm_norm_sq.view(batch_size, 1, 1) * eye_3x3 - rm_outer
            )
            IA_list.append(I_rotated + parallel_axis_term)

        # LuGre 摩擦参数计算，与原网络保持一致
        fc_feat = y_shared
        for i in range(len(self.net_fc_list) - 1):
            fc_feat = self.net_fc_list[i](fc_feat)
        fc_raw = self.net_fc_list[-1](fc_feat)
        fc_reshaped = fc_raw.view(batch_size, 3, 6)
        fc_scale = torch.exp(self.log_fc_scale)
        fc_tensor = (torch.exp(fc_reshaped * 10.0) * fc_scale).view(batch_size, 18)

        IA_tensor = torch.stack(IA_list, dim=1)
        rm_tensor = torch.stack(rm_list, dim=1)
        return IA_tensor, rm_tensor, m_tensor, fc_tensor
