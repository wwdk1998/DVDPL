import torch
import torch.nn as nn
from activation_functions import ActivationFunctions


class NetLayer(nn.Module):
    """单个网络层"""
    
    def __init__(self, input_size, output_dim, activation='ReLU'):
        """
        Args:
            input_size: 输入维度
            output_dim: 输出维度
            activation: 激活函数类型 ('ReLU', 'SoftPlus', 'Cos', 'Tanh', 'Linear')
        """
        super(NetLayer, self).__init__()
        
        self.input_size = input_size
        self.dim = output_dim
        self.activation = activation
        self.softplus_beta = 1.0
        
        # 定义权重和偏置（原有随机初始化代码）
        self.weight = nn.Parameter(torch.randn(output_dim, input_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, u):
        # 线性变换
        a = torch.matmul(u, self.weight.T) + self.bias
        
        # 应用激活函数
        if self.activation == 'ReLU':
            out = ActivationFunctions.relu(a)
        elif self.activation == 'SoftPlus':
            out = ActivationFunctions.softplus(a, self.softplus_beta)
        elif self.activation == 'Cos':
            out = ActivationFunctions.cos_func(a)
        elif self.activation == 'Tanh':
            out = ActivationFunctions.tanh(a)
        elif self.activation == 'Linear':
            out = ActivationFunctions.linear(a)
        else:
            raise ValueError(f'Unknown activation type: {self.activation}')
        return out

