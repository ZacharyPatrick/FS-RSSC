import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt.lmath as lmath


class ToLorentz(nn.Module):
    def __init__(self, c, clip_r=None):
        super(ToLorentz, self).__init__()
        self.clip_r = clip_r
        self.c = c

    def forward(self, x, c=None):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
        
        if c is None:
            c = self.c
        
        # 1.【关键修改】HLFormer 风格的洛伦兹投影
        # 先在第0维补0，扩展维度 d -> d+1
        # 假设 x 的 shape 是 (Batch, dim) -> (Batch, dim+1)
        x_padded = F.pad(x, pad=(1, 0), value=0)

        # 3. 【参数转换修正】
        # 如果 c 是 tensor，直接计算 1/c
        # 如果 c 是 float，转为 tensor
        if isinstance(c, torch.Tensor):
            k_val = 1.0 / c
        else:
            k_val = torch.tensor(1.0 / c, device=x.device, dtype=x.dtype)
        
        # 2. 调用 lmath.py 中的 expmap0
        # 注意：你需要确保引入了 lmath 模块，或者把 _expmap0 函数复制过来
        return lmath.expmap0(x_padded, k=k_val)


class ToEuclidean(nn.Module):
    def __init__(self, c):
        super(ToEuclidean, self).__init__()
        self.c = c

    def forward(self, x, c=None):
        """
        Input:  [Batch, ..., Dim + 1] (Hyperbolic)
        Output: [Batch, ..., Dim] (Euclidean)
        """
        if c is None:
            c = self.c
        
        if isinstance(c, torch.Tensor):
            k_val = 1.0 / c
        else:
            k_val = torch.tensor(1.0 / c, device=x.device, dtype=x.dtype)
            
        # 1. 对数映射: 流形 -> 切空间原点
        # shape 保持 [B, ..., D+1], 但第0维理论上接近0
        x_tangent = lmath.logmap0(x, k=k_val)
        
        # 2. 维度切片: 去除时间维
        # [B, ..., D+1] -> [B, ..., D]
        x_euc = x_tangent[..., 1:]
        
        return x_euc


class LorentzNormalization(nn.Module):
    """
    Normalizes spatial components to unit norm and recomputes time component
    to satisfy Lorentz geometry constraints.

    Args:
        manifold_in: Lorentz manifold object (input space).
        manifold_out: Optional target manifold for output projection.
        
    """
    def __init__(self, c, manifold_out=None, return_space=False):
        super(LorentzNormalization, self).__init__()
        self.manifold_out = manifold_out
        self.c = c

    def forward(self, x, norm_factor=None, space_only=False, return_space=False):
        """
        Forward pass of LorentzNormalization.

        Args:
            x (torch.Tensor): Input tensor with Lorentzian coordinates.
            norm_factor (torch.Tensor, optional): Precomputed normalization factors.
            space_only (bool): If true, the input is only the space-like dimension of the Lorentz vector
            return_space (bool): If true, returns only the space-like dimension of the results to save computation

        Returns:
            torch.Tensor: Lorentz-normalized tensor.
        """
        if space_only:
            x_space = x
        else:
            x_space = x[..., 1:]
        if norm_factor is not None:
            x_space = x_space * norm_factor
        else:
            x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        if return_space:
            x = x_space
        else:
            x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.c / self.c).sqrt()
        return x


def lorentz_linear_fusion(a, b, f, c):
    """
    洛伦兹模型下的线性加权融合: result = f * a + (1-f) * b
    
    Args:
        a: [1, way, dim+1] 洛伦兹向量
        b: [1, way, dim+1] 洛伦兹向量
        f: 标量 (float) 或 张量 (Tensor with shape broadcasting to a/b)
           f 代表 a 的权重 (0 <= f <= 1)
        c: 曲率参数 (curvature)
        
    Returns:
        Fused vector on the manifold, shape [1, way, dim+1]
    """
    # 1. 转换曲率参数 c -> k (半径平方)
    if isinstance(c, torch.Tensor):
        k = 1.0 / c
    else:
        k = torch.tensor(1.0 / c, device=a.device, dtype=a.dtype)
    
    # 2. 在环境空间(Ambient Space)进行欧氏线性加权
    # 这一步计算出的 mixed_raw 不在双曲面上
    # 如果 f 是 float，直接乘；如果是 tensor，注意维度广播
    mixed_raw = f * a + (1.0 - f) * b
    
    # 3. 投影回双曲面 (关键步骤)
    # 利用 lmath.project 修正时间维分量，使其满足双曲约束
    mixed_projected = lmath.project(mixed_raw, k=k, dim=-1)
    
    return mixed_projected


def lorentzian_centroid(x, c, weight=None, dim=-2, eps=1e-6):
    """
    独立版本的洛伦兹质心计算函数 (Standalone Lorentzian Centroid).
    严格遵循 HELM/Geoopt 中 Lorentz 类的实现逻辑。

    Args:
        x (Tensor): 输入的双曲张量 (包含时间维), shape [..., N, D+1]
        c (float or Tensor): 曲率 (curvature). 注意这里是 c, 不是 k.
        weight (Tensor, optional): 权重矩阵. 
            如果提供，执行 weight @ x (矩阵乘法聚合).
            如果不提供，执行 x.mean(dim).
        dim (int): 聚合的维度 (仅在 weight 为 None 时生效). 默认 -2.
        eps (float): 数值稳定性常数. 默认 1e-6 (对应 float32).

    Returns:
        Tensor: 投影回双曲流形的质心向量, shape [..., D+1]
    """
    
    # 1. 计算环境空间中的线性均值 (Ambient Average)
    # 对应原代码: if weight is not None: ave = weight @ x else: ave = x.mean(...)
    if weight is not None:
        # 注意：这里使用矩阵乘法进行聚合
        # 假设 weight: [..., Q_len, K_len], x: [..., K_len, Dim]
        # output: [..., Q_len, Dim]
        ave = weight @ x
    else:
        # 对应原代码: ave = x.mean(dim=-2)
        ave = x.mean(dim=dim)

    # 2. 计算分母 (双曲模长)
    # 对应原代码: denom = (-self.l_inner(ave, ave, ...)).abs().clamp_min(eps).sqrt()
    
    # 使用 lmath.inner 替代 self.l_inner
    # 注意：内积永远是在特征维度 (最后一维) 上计算的，所以这里固定 dim=-1
    ave_inner = lmath.inner(ave, ave, dim=-1, keepdim=True)
    
    # 计算模长并进行数值钳位 (Clamping)
    denom = (-ave_inner).abs().clamp_min(eps).sqrt()

    # 3. 投影与缩放
    # 对应原代码: return self.c.sqrt() * ave / denom
    
    # 处理 c 的类型 (支持 float 或 Tensor)
    if isinstance(c, torch.Tensor):
        sqrt_c = c.sqrt()
    else:
        # 保持与 x 相同的 device 和 dtype
        sqrt_c = torch.tensor(c, device=x.device, dtype=x.dtype).sqrt()
        
    return sqrt_c * ave / denom