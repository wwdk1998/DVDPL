import torch

def skew_symmetric(v):
    """
    构造反对称矩阵（批量）
    Args:
        v: [batch_size, 3] 向量
        
    Returns:
        S: [batch_size, 3, 3] 反对称矩阵
    """
    batch_size = v.shape[0]
    device = v.device
    S = torch.zeros(batch_size, 3, 3, device=device, dtype=v.dtype)
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    
    return S

def dynmic_batch(BV_batch, dBV_batch, g_batch, IA, rm, m, link_indices):
    """
    完全向量化的动力学计算（批量处理所有样本和连杆）
    Args:
        BV_batch: [batch_size, 72] 速度
        dBV_batch: [batch_size, 72] 加速度
        g_batch: [batch_size, 36] 重力
        IA: [batch_size, 12, 3, 3] 12个连杆的惯性张量
        rm: [batch_size, 12, 3] 12个连杆的质心位置
        m: [batch_size, 12] 12个连杆的质量
        link_indices: list, 需要计算的连杆索引，如[0,1,...,11]或[2,5,11]（0-indexed）
    Returns:
        F_link: dict, {link_idx: [batch_size, 6]} 力向量
    """
    batch_size = BV_batch.shape[0]
    device = BV_batch.device
    
    F_link = {}
    # 对每个连杆计算
    for link_i in link_indices:
        # 提取第link_i个连杆的数据
        idx_6_start = link_i * 6
        idx_6_end = (link_i + 1) * 6
        idx_3_start = link_i * 3
        idx_3_end = (link_i + 1) * 3
        
        BV_i = BV_batch[:, idx_6_start:idx_6_end]  # [batch_size, 6]
        dBV_i = dBV_batch[:, idx_6_start:idx_6_end]  # [batch_size, 6]
        g_i = g_batch[:, idx_3_start:idx_3_end]  # [batch_size, 3]
        IA_i = IA[:, link_i, :, :]  # [batch_size, 3, 3]
        rm_i = rm[:, link_i, :]  # [batch_size, 3]
        m_i = m[:, link_i]  # [batch_size]
        
        # 构造r_matrix (反对称矩阵) [batch_size, 3, 3]
        r_matrix = skew_symmetric(rm_i)
        
        # 构造M矩阵 [batch_size, 6, 6]
        m_expanded = m_i.view(batch_size, 1, 1)  # [batch_size, 1, 1]
        
        eye3 = torch.eye(3, device=device).unsqueeze(0)  # [1, 3, 3]
        
        # M矩阵分块
        M11 = m_expanded * eye3  # [batch_size, 3, 3]
        M12 = -m_expanded * r_matrix  # [batch_size, 3, 3]
        M21 = m_expanded * r_matrix  # [batch_size, 3, 3]
        M22 = IA_i - torch.bmm(m_expanded * r_matrix, r_matrix)  # [batch_size, 3, 3]
        
        # 组装M矩阵 [batch_size, 6, 6]
        M_top = torch.cat([M11, M12], dim=2)
        M_bottom = torch.cat([M21, M22], dim=2)
        M = torch.cat([M_top, M_bottom], dim=1)
        
        # 提取线速度和角速度
        v = BV_i[:, :3]  # [batch_size, 3]
        w = BV_i[:, 3:]  # [batch_size, 3]
        dv = dBV_i[:, :3]  # [batch_size, 3]
        dw = dBV_i[:, 3:]  # [batch_size, 3]
        
        # 构造角速度反对称矩阵 [batch_size, 3, 3]
        w_matrix = skew_symmetric(w)
        
        # 构造C矩阵 [batch_size, 6, 6]
        C11 = m_expanded * w_matrix  # [batch_size, 3, 3]
        C12 = torch.bmm(m_expanded * r_matrix, w_matrix)  # [batch_size, 3, 3]
        C21 = -torch.bmm(m_expanded * w_matrix, r_matrix)  # [batch_size, 3, 3]
        
        # C22计算
        w_IA = torch.bmm(w_matrix, IA_i)
        IA_w = torch.bmm(IA_i, w_matrix)
        r_w_r = torch.bmm(torch.bmm(m_expanded * r_matrix, w_matrix), r_matrix)
        C22 = w_IA + IA_w - r_w_r
        
        C_top = torch.cat([C11, C12], dim=2)
        C_bottom = torch.cat([C21, C22], dim=2)
        C = torch.cat([C_top, C_bottom], dim=1)
        
        # 构造重力项 [batch_size, 6]
        G_linear = (m_i.unsqueeze(1) * g_i)  # [batch_size, 3]
        G_angular = torch.bmm(m_expanded * r_matrix, g_i.unsqueeze(2)).squeeze(2)  # [batch_size, 3]
        G = torch.cat([G_linear, G_angular], dim=1)  # [batch_size, 6]

        # 组装速度和加速度向量
        q_dot = torch.cat([v, w], dim=1).unsqueeze(2)  # [batch_size, 6, 1]
        q_ddot = torch.cat([dv, dw], dim=1).unsqueeze(2)  # [batch_size, 6, 1]

        # 计算 F = M*q_ddot + C*q_dot + G
        M_q_ddot = torch.bmm(M, q_ddot).squeeze(2)
        C_q_dot = torch.bmm(C, q_dot).squeeze(2)

        F_i = M_q_ddot + C_q_dot + G  # [batch_size, 6]
        F_link[link_i] = F_i
    return F_link


def solve_actuator_forces(F_link_all, BU, PARA):
    """
    反向迭代求解执行器力（从末端执行器往基座）
    Args:
        F_link_all: dict, {link_id: [batch_size, 6]} 各连杆的惯性力
        BU: [batch_size, 102] 广义速度转换矩阵（预先计算）
        PARA: [batch_size, 3, 5] 机器人参数（每个平行体两个参数）
    Returns:
        Pre_F: [batch_size, 3] 执行器预测力 [F2, F4, F5]
    """
    batch_size = BU.shape[0]
    device = BU.device
    
    # 提取机器人参数
    x2 = PARA[:, 0, 0]   # [batch_size]
    x21 = PARA[:, 0, 1]
    x22 = PARA[:, 0, 2]
    l22 = PARA[:, 0, 3]
    q22 = PARA[:, 0, 4]
    
    x4 = PARA[:, 1, 0]
    x41 = PARA[:, 1, 1]
    x42 = PARA[:, 1, 2]
    l42 = PARA[:, 1, 3]
    q42 = PARA[:, 1, 4]
    
    x5 = PARA[:, 2, 0]
    x51 = PARA[:, 2, 1]
    x52 = PARA[:, 2, 2]
    l52 = PARA[:, 2, 3]
    q52 = PARA[:, 2, 4]
    
    # 提取转换矩阵
    B212_U_T2 = BU[:, 6:12, :]        # [batch_size, 6, 6]
    B221_U_B222 = BU[:, 18:24, :]    # [batch_size, 6, 6]
    B222_U_T2 = BU[:, 24:30, :]      # [batch_size, 6, 6]
    
    T2_U_B4 = BU[:, 30:36, :]        # [batch_size, 6, 6]
    B4_U_B412 = BU[:, 36:42, :]      # [batch_size, 6, 6]
    B412_U_T4 = BU[:, 42:48, :]      # [batch_size, 6, 6]
    B4_U_B421 = BU[:, 48:54, :]      # [batch_size, 6, 6]
    B421_U_B422 = BU[:, 54:60, :]    # [batch_size, 6, 6]
    B422_U_T4 = BU[:, 60:66, :]      # [batch_size, 6, 6]
    
    T4_U_B5 = BU[:, 66:72, :]        # [batch_size, 6, 6]
    B5_U_B512 = BU[:, 72:78, :]      # [batch_size, 6, 6]
    B512_U_T5 = BU[:, 78:84, :]      # [batch_size, 6, 6]
    B5_U_B521 = BU[:, 84:90, :]      # [batch_size, 6, 6]
    B521_U_B522 = BU[:, 90:96, :]    # [batch_size, 6, 6]
    B522_U_T5 = BU[:, 96:102, :]     # [batch_size, 6, 6]
    
    # 初始化基座力为零
    T5_F = torch.zeros(batch_size, 6, device=device)  # [batch_size, 6]
    z1 = torch.zeros(batch_size, 6, device=device)
    z1[:, 0] = 1.0  # [1, 0, 0, 0, 0, 0]
    z6 = torch.zeros(batch_size, 6, device=device)
    z6[:, 5] = 1.0  # [0, 0, 0, 0, 0, 1]
    
    # 执行器5的反向迭代求解
    theta5 = torch.atan2(T5_F[:, 1], T5_F[:, 0])  # [batch_size]
    sin_q52 = torch.sin(q52)
    cos_q52 = torch.cos(q52)
    sin_theta5 = torch.sin(theta5)
    cos_theta5 = torch.cos(theta5)
    
    k51 = cos_theta5 * torch.sin(q52 - theta5) / sin_q52  # [batch_size]
    k52 = sin_theta5 * torch.cos(q52 - theta5) / sin_q52  # [batch_size]
    
    tz5 = (z6 * k52.unsqueeze(1) * T5_F).sum(dim=1)  # [batch_size]
    
    B512_netF = F_link_all[9]   # 连杆10 
    B521_netF = F_link_all[10]  # 连杆11 
    B522_netF = F_link_all[11]  # 连杆12 
    
    # fy5 = (z6'*(B512_netF+B512_U_T5*k51*T5_F)+tz5)/(-l52)
    # B512_U_T5: [batch_size, 6, 6], k51: [batch_size], T5_F: [batch_size, 6]
    # 结果 [batch_size, 6]
    k51_T5_F = torch.bmm(B512_U_T5, (k51.view(batch_size, 1, 1) * T5_F.unsqueeze(2))).squeeze(2)
    B512_term = B512_netF + k51_T5_F  # [batch_size, 6]
    fy5 = ((z6 * B512_term).sum(dim=1) + tz5) / (-l52)  # [batch_size]
    
    # fx5计算
    # B522_U_T5 * k52 * T5_F
    k52_T5_F = torch.bmm(B522_U_T5, (k52.view(batch_size, 1, 1) * T5_F.unsqueeze(2))).squeeze(2)
    B522_interm = B522_netF + k52_T5_F  # [batch_size, 6]
    # B521_U_B522 * B522_interm
    B521_B522_term = B521_netF + torch.bmm(B521_U_B522, B522_interm.unsqueeze(2)).squeeze(2)  # [batch_size, 6]
    
    denominator_fx5 = sin_q52 * (-(x52 + x5 + x51))
    

    numerator_fx5 = ((z6 * B521_B522_term).sum(dim=1) - 
                     (x52 + x5 + x51) * cos_q52 * fy5 - tz5)
    fx5 = numerator_fx5 / denominator_fx5  # [batch_size]
    
    T5_inF = torch.zeros(batch_size, 6, device=device)
    T5_inF[:, 0] = fx5
    T5_inF[:, 1] = fy5
    T5_inF[:, 5] = tz5
    
    T52_F = k52.view(batch_size, 1) * T5_F - T5_inF
    B522_F = B522_netF + torch.bmm(B522_U_T5, T52_F.unsqueeze(2)).squeeze(2)
    
    B51_netF = F_link_all[8]  # 连杆9 
    B5_U_B512 = BU[:, 72:78, :]
    B5_U_B521 = BU[:, 84:90, :]
    
    B5_F = (B51_netF + 
            torch.bmm(B5_U_B512, B512_netF.unsqueeze(2)).squeeze(2) +
            torch.bmm(B5_U_B521, B521_netF.unsqueeze(2)).squeeze(2) +
            torch.bmm(B5_U_B521, torch.bmm(B521_U_B522, B522_netF.unsqueeze(2))).squeeze(2) +
            torch.bmm(B5_U_B512, torch.bmm(B512_U_T5, T5_F.unsqueeze(2))).squeeze(2))
    
    # 执行器4的反向迭代求解
    T4_F = (T4_U_B5 @ B5_F.unsqueeze(2)).squeeze(2)  # [batch_size, 6]
    
    theta4 = torch.atan2(T4_F[:, 1], T4_F[:, 0])
    sin_q42 = torch.sin(q42)
    cos_q42 = torch.cos(q42)
    sin_theta4 = torch.sin(theta4)
    cos_theta4 = torch.cos(theta4)
    
    k41 = cos_theta4 * torch.sin(q42 - theta4) / sin_q42
    k42 = sin_theta4 * torch.cos(q42 - theta4) / sin_q42
    
    tz4 = (z6 * k42.unsqueeze(1) * T4_F).sum(dim=1)
    
    B412_netF = F_link_all[5]  # 连杆6 
    B421_netF = F_link_all[6]  # 连杆7 
    B422_netF = F_link_all[7]  # 连杆8 
    
    B412_term = B412_netF + torch.bmm(B412_U_T4, (k41.view(batch_size, 1, 1) * T4_F.unsqueeze(2))).squeeze(2)
    fy4 = ((z6 * B412_term).sum(dim=1) + tz4) / (-l42)
    
    B422_interm = B422_netF + torch.bmm(B422_U_T4, (k42.view(batch_size, 1, 1) * T4_F.unsqueeze(2))).squeeze(2)
    B421_B422_term = B421_netF + torch.bmm(B421_U_B422, B422_interm.unsqueeze(2)).squeeze(2)
    
    denominator_fx4 = sin_q42 * (-(x42 + x4 + x41))
    
    numerator_fx4 = ((z6 * B421_B422_term).sum(dim=1) - 
                     (x42 + x4 + x41) * cos_q42 * fy4 - tz4)
    fx4 = numerator_fx4 / denominator_fx4
    
    
    T4_inF = torch.zeros(batch_size, 6, device=device)
    T4_inF[:, 0] = fx4
    T4_inF[:, 1] = fy4
    T4_inF[:, 5] = tz4
    
    T42_F = k42.view(batch_size, 1) * T4_F - T4_inF
    B422_F = B422_netF + torch.bmm(B422_U_T4, T42_F.unsqueeze(2)).squeeze(2)
    
    B41_netF = F_link_all[4]  # 连杆5 
    B4_F = (B41_netF +
            torch.bmm(B4_U_B412, B412_netF.unsqueeze(2)).squeeze(2) +
            torch.bmm(B4_U_B421, B421_netF.unsqueeze(2)).squeeze(2) +
            torch.bmm(B4_U_B421, torch.bmm(B421_U_B422, B422_netF.unsqueeze(2))).squeeze(2) +
            torch.bmm(B4_U_B412, torch.bmm(B412_U_T4, T4_F.unsqueeze(2))).squeeze(2))
    
    # 执行器2的反向迭代求解
    T2_F = (T2_U_B4 @ B4_F.unsqueeze(2)).squeeze(2)  # [batch_size, 6]
    
    theta2 = torch.atan2(T2_F[:, 1], T2_F[:, 0])
    sin_q22 = torch.sin(q22)
    cos_q22 = torch.cos(q22)
    sin_theta2 = torch.sin(theta2)
    cos_theta2 = torch.cos(theta2)
    
    k21 = cos_theta2 * torch.sin(q22 - theta2) / sin_q22
    k22 = sin_theta2 * torch.cos(q22 - theta2) / sin_q22
    
    tz2 = (z6 * k22.unsqueeze(1) * T2_F).sum(dim=1)
    
    B212_netF = F_link_all[1]  # 连杆2 
    B221_netF = F_link_all[2]  # 连杆3 
    B222_netF = F_link_all[3]  # 连杆4 
    
    B212_term = B212_netF + torch.bmm(B212_U_T2, (k21.view(batch_size, 1, 1) * T2_F.unsqueeze(2))).squeeze(2)
    fy2 = ((z6 * B212_term).sum(dim=1) + tz2) / (-l22)
    
    B222_interm = B222_netF + torch.bmm(B222_U_T2, (k22.view(batch_size, 1, 1) * T2_F.unsqueeze(2))).squeeze(2)
    B221_B222_term = B221_netF + torch.bmm(B221_U_B222, B222_interm.unsqueeze(2)).squeeze(2)
    
    denominator_fx2 = sin_q22 * (-(x22 + x2 + x21))

    numerator_fx2 = ((z6 * B221_B222_term).sum(dim=1) - 
                     (x22 + x2 + x21) * cos_q22 * fy2 - tz2)
    fx2 = numerator_fx2 / denominator_fx2
    
    T2_inF = torch.zeros(batch_size, 6, device=device)
    T2_inF[:, 0] = fx2
    T2_inF[:, 1] = fy2
    T2_inF[:, 5] = tz2
    
    T22_F = k22.view(batch_size, 1) * T2_F - T2_inF
    B222_F = B222_netF + torch.bmm(B222_U_T2, T22_F.unsqueeze(2)).squeeze(2)

    # 执行器力计算
    F2 = (z1 * B222_F).sum(dim=1)  # [batch_size]
    F4 = (z1 * B422_F).sum(dim=1)  # [batch_size]
    F5 = (z1 * B522_F).sum(dim=1)  # [batch_size]
    
    Fl = torch.stack([F2, F4, F5], dim=1)  # [batch_size, 3]

    return Fl

def solve_actuator_friction(fc_params, dx, dt, device, z_init=None):
    """
    计算执行器摩擦力（LuGre动态模型- 完整积分，无稳态近似）
    Args:
        fc_params: [batch_size, 3, 6] 摩擦参数
                   每个执行器的6个参数 [alpha_0, alpha_1, alpha_2, v_s, sigma_0, sigma_1]
        dx: [batch_size, 3] 各执行器的线速度 v
        dt: float 时间步长 (s)
        device: torch device
        z_init: [batch_size, 3] 可选，初始刚毛位移状态（如果为None则初始化为零）
    Returns:
        Ff: [batch_size, 3] 各执行器的摩擦力
        z_next: [batch_size, 3] 更新后的刚毛位移状态（用于下一时刻）
    
    物理模型：
        g(v) = alpha_0 + alpha_1 * exp(-(v/v_s)^2)
        dz/dt = v - sigma_0 * |v| / g(v) * z
        F_f = sigma_0 * z + sigma_1 * dz/dt + alpha_2 * v
    """
    batch_size = dx.shape[0]
    
    # 提取摩擦参数 [batch_size, 3]
    alpha_0 = fc_params[:, :, 0]  # [batch_size, 3]
    alpha_1 = fc_params[:, :, 1]
    alpha_2 = fc_params[:, :, 2]
    v_s = fc_params[:, :, 3]
    sigma_0 = fc_params[:, :, 4]
    sigma_1 = fc_params[:, :, 5]
    
    # 避免数值问题：仅做必要的clamp以避免除零
    v_s = torch.clamp(v_s, min=1e-4)  # 防止速度尺度参数过小
    sigma_0 = torch.clamp(sigma_0, min=1e-4)  # 防止刚度参数过小
    
    # 初始化z（刚毛位移状态）
    if z_init is None:
        z = torch.zeros(batch_size, 3, device=device)
    else:
        z = z_init.clone()
    
    # 当前速度
    v = dx  # [batch_size, 3]
    
    # 计算 g(v) = alpha_0 + alpha_1 * exp(-(v/v_s)^2)
    exp_term = torch.exp(-torch.clamp((v / v_s) ** 2, max=50))  # 限制指数输入
    g_v = alpha_0 + alpha_1 * exp_term  # [batch_size, 3]
    g_v = torch.clamp(g_v, min=1e-4, max=1e4)  # 限制g(v)范围
    
    # 计算 |v|
    v_abs = torch.abs(v) + 1e-8  # [batch_size, 3]
    
    # ====== 修改为精确离散化（Absolute Stable) ======
    # 消除高刚度(sigma_0高达10^7)条件下的显式欧拉造成的数值刚性发散
    
    # 微分方程: dz/dt = A * z + B
    # 其中 A = -sigma_0 * |v| / g(v)
    A_term = -sigma_0 * v_abs / g_v 
    
    # 稳态极限: dz/dt = 0 时的刚毛位移稳态值 z_ss
    z_ss = v / (-A_term)
    
    # 解析闭式解精确更新下一时刻的 z: 
    # z(t+dt) = z(t)*exp(A*dt) + z_ss*(1 - exp(A*dt))
    exp_A_dt = torch.exp(A_term * dt)
    z_next = z * exp_A_dt + z_ss * (1 - exp_A_dt)
    
    # 计算此时步内的等效状态平均导数，彻底消灭基于单点梯度的数值毛刺跳变
    dz_dt = (z_next - z) / dt
    # ==================================================
    
    # 计算摩擦力 F_f = sigma_0 * z + sigma_1 * dz/dt + alpha_2 * v
    term1 = sigma_0 * z
    term2 = sigma_1 * dz_dt
    term3 = alpha_2 * v
    Ff = term1 + term2 + term3  # [batch_size, 3]

    # # 【诊断代码：仅打印，不修改算法】
    # # 监控执行器4（索引1）摩擦力过大的情况
    # if torch.any(torch.abs(Ff[:, 1]) > 50000):
    #     max_idx = torch.argmax(torch.abs(Ff[:, 1])).item()
    #     print(f"\n[诊断] 执行器4 异常总摩擦力 F4 = {Ff[max_idx, 1].item():.1f}")
    #     print(f"项拆解 -> 刚度项(s0*z)={term1[max_idx, 1].item():.1f} | 阻尼项(s1*dz_dt)={term2[max_idx, 1].item():.1f} | 粘性项(a2*v)={term3[max_idx, 1].item():.1f}")
    #     print(f"状态量 -> z={z[max_idx, 1].item():.6f} | dz_dt={dz_dt[max_idx, 1].item():.6f} | v={v[max_idx, 1].item():.6f} | g(v)={g_v[max_idx, 1].item():.6f}")
    #     print(f"摩擦参数 -> sigma_0={sigma_0[max_idx, 1].item():.1f} | sigma_1={sigma_1[max_idx, 1].item():.1f} | alpha_2={alpha_2[max_idx, 1].item():.1f} | v_s={v_s[max_idx, 1].item():.5f}\n")

    return Ff, z_next