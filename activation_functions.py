"""
激活函数模块
"""
import torch
import torch.nn.functional as F


class ActivationFunctions:
    """激活函数集合"""
    
    @staticmethod
    def relu(x):
        """ReLU激活函数"""
        return F.relu(x)
    
    @staticmethod
    def softplus(x, beta=1.0):
        """SoftPlus激活函数"""
        return F.softplus(x, beta=beta)
    
    @staticmethod
    def cos_func(x):
        """余弦激活函数"""
        return torch.cos(x)
    
    @staticmethod
    def linear(x):
        """线性激活函数（恒等变换）"""
        return x
    
    @staticmethod
    def tanh(x):
        """Tanh激活函数"""
        return torch.tanh(x)

