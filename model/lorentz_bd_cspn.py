import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt import lmath  

class HyperbolicBDCSPN:
    """
    双曲空间中的原型校正 (Hyperbolic Prototype Rectification / H-BD-CSPN)
    
    严格遵循 BD-CSPN 的迭代思想：
    1. 计算 Query 到原型的双曲距离与软分配 (Soft Assignment)
    2. 计算 Query 的加权聚类中心 (在切空间进行)
    3. 将原型沿测地线向 Query 中心移动 alpha 比例
    """
    
    def __init__(self, curvature=1.0):
        self.c = curvature

    def _get_k(self, device, dtype):
        """辅助函数: 获取 k = 1/c"""
        if isinstance(self.c, torch.Tensor):
            k = 1.0 / self.c.to(device)
        else:
            k = torch.tensor(1.0 / self.c, device=device, dtype=dtype)
        return k

    def rectify(self, initial_protos, query_feats, n_iter=20, alpha=0.1, temperature=1.0):
        """
        执行迭代校正
        
        Args:
            initial_protos: (N_way, D+1) 初始支持集原型 (Lorentz Vector)
            query_feats: (N_query, D+1) 查询集特征 (Lorentz Vector)
            n_iter: 迭代次数 (默认 10)
            alpha: 更新步长/学习率 (默认 0.1)
            temperature: 温度系数 (默认 10.0)
            
        Returns:
            rectified_protos: (N_way, D+1) 校正后的原型
        """
        device = initial_protos.device
        dtype = initial_protos.dtype
        k = self._get_k(device, dtype)
        sqrt_k = torch.sqrt(k)

        # 复制一份当前原型用于迭代
        curr_protos = initial_protos.clone()

        # 开始迭代校正
        for i in range(n_iter):
            # ==========================================
            # 1. 计算当前原型与 Query 的相似度
            # ==========================================
            # 欧氏: logits = query @ proto.t()
            # 双曲: logits = -distance^2 * temperature
            
            # dists: (N_query, N_way)
            # 这里的 dists 是度量距离 (Metric Distance)
            dists = lmath.dist(
                query_feats.unsqueeze(1), 
                curr_protos.unsqueeze(0), 
                k=k
            )
            
            # 转换为 Logits
            logits = -dists.pow(2) * temperature

            # ==========================================
            # 2. 计算软分配 (Soft Assignment)
            # ==========================================
            # probs: (N_query, N_way)
            probs = torch.softmax(logits, dim=1)

            # ==========================================
            # 3. 计算 Query 的聚类中心 (基于伪标签加权)
            # ==========================================
            # 欧氏: centers = (probs.t() @ query) / sum_probs
            # 双曲: 无法直接加权。采用 "切空间近似" (Tangent Space Mean)。
            #       Logic: LogMap0 -> Weighted Mean -> ExpMap0
            
            # A. 投影到原点切空间 -> (N_query, D+1)
            v_query = lmath.logmap0(query_feats, k=k)
            
            # B. 在切空间进行加权平均 (利用矩阵乘法加速)
            # probs.T: (N_way, N_query)
            # v_query: (N_query, D+1)
            # weighted_sum: (N_way, D+1)
            weighted_sum = probs.t() @ v_query
            
            # 计算分母 (每个类的权重和): (N_way, 1)
            sum_probs = probs.sum(dim=0).unsqueeze(1) + 1e-8
            
            # 切空间均值
            v_centers = weighted_sum / sum_probs
            
            # C. 投影回双曲空间 -> (N_way, D+1)
            query_centers = lmath.expmap0(v_centers, k=k)
            
            # ==========================================
            # 4. 更新原型 (Geodesic Update)
            # ==========================================
            # 欧氏: curr = curr + alpha * (center - curr)
            # 双曲: 沿测地线从 curr 向 center 移动 alpha 比例
            #       使用之前推导的 geodesic_interpolation 公式
            
            # 计算两点间度量距离 D: (N_way, 1)
            # 注意: 这里计算的是 "当前原型" 到 "Query中心" 的距离
            D = lmath.dist(curr_protos, query_centers, k=k).view(-1, 1)
            
            # 转换为参数距离 zeta (Hyperbolic Angle) 用于 sinh
            zeta = D / sqrt_k
            
            # 数值稳定性处理 (防止 D=0)
            is_close = zeta < 1e-5
            zeta_safe = zeta.clone()
            zeta_safe[is_close] = 1.0
            
            sinh_zeta = torch.sinh(zeta_safe)
            
            # 插值系数: 
            # 我们要从 curr_protos (t=0) 移动到 query_centers (t=1) 的 alpha 处
            # 公式: p(t) = [sinh((1-t)z)/sinh(z)]*p0 + [sinh(tz)/sinh(z)]*p1
            # 这里 t = alpha
            
            coeff_curr = torch.sinh((1.0 - alpha) * zeta_safe) / sinh_zeta
            coeff_center = torch.sinh(alpha * zeta_safe) / sinh_zeta
            
            # 执行更新
            curr_protos = coeff_curr * curr_protos + coeff_center * query_centers
            
            # 恢复重合点 (如果原型和中心重合，保持不变)
            if is_close.any():
                curr_protos[is_close.squeeze()] = query_centers[is_close.squeeze()]

        return curr_protos


class HyperbolicTIM(nn.Module):
    """
    双曲空间直推式信息最大化 (H-TIM) - 最终确认版
    """
    def __init__(self, curvature=1.0):
        super().__init__()
        self.c = curvature

    def _get_k(self, device, dtype):
        if isinstance(self.c, torch.Tensor):
            k = 1.0 / self.c.to(device)
        else:
            k = torch.tensor(1.0 / self.c, device=device, dtype=dtype)
        return k

    def infer(self, initial_protos, query_feats, n_iter=20, lr=0.1, temperature=1.0, balance_weight=1.0):
        """
        执行 H-TIM 推理
        """
        device = initial_protos.device
        dtype = initial_protos.dtype
        k = self._get_k(device, dtype)
        
        with torch.enable_grad():
            # 1. 初始化可优化原型
            # 使用 detach() 创建副本，避免影响之前的计算图
            protos = initial_protos.clone().detach()
            protos.requires_grad = True
            
            # 定义优化器
            optimizer = torch.optim.AdamW([protos], lr=lr)

            for i in range(n_iter):
                optimizer.zero_grad()
                
                # ==========================
                # A. 计算双曲概率 P(y|x)
                # ==========================
                # 使用真实的度量距离平方来模拟高斯分布
                dists = lmath.dist(
                    query_feats.unsqueeze(1), 
                    protos.unsqueeze(0), 
                    k=k
                )
                
                # Logits = -d^2 * T
                logits = -dists.pow(2) * temperature
                
                # P_cond: 条件概率 (N_query, N_way)
                P_cond = F.softmax(logits, dim=1)
                
                # ==========================
                # B. 计算 TIM Loss
                # ==========================
                # 1. 条件熵 Loss (Min Conditional Entropy)
                loss_cond = - (P_cond * torch.log(P_cond + 1e-8)).sum(dim=1).mean()
                
                # 2. 边缘熵 Loss (Max Marginal Entropy)
                P_marg = P_cond.mean(dim=0)
                loss_marg = (P_marg * torch.log(P_marg + 1e-8)).sum()
                
                # 总 Loss
                loss = loss_cond + balance_weight * loss_marg
                
                # ==========================
                # C. 更新与投影 (Update & Project)
                # ==========================
                loss.backward()
                optimizer.step()
                
                # 【关键确认】投影回双曲流形
                # 利用您提供的 lmath.project 严格约束 x0
                with torch.no_grad():
                    protos.data = lmath.project(protos.data, k=k, dim=-1)

        return protos.detach()


# if __name__ == "__main__":
#     # 假设环境配置
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     n_way = 5
#     n_query = 15
#     dim = 385 # Lorentz dim
    
#     # 初始化工具
#     # rectifier = HyperbolicBDCSPN(curvature=1.0)
#     rectifier = HyperbolicTIM(curvature=1.0)
    
#     # 模拟数据 (确保在流形上)
#     def random_lorentz(n, d):
#         x = torch.randn(n, d).to(device) * 0.1
#         x0 = torch.sqrt(1 + torch.norm(x, dim=1, keepdim=True).pow(2))
#         return torch.cat([x0, x], dim=1)

#     # 初始原型 (来自 Support Set)
#     init_protos = random_lorentz(n_way, dim-1)
#     # Query 特征
#     query_feats = random_lorentz(n_way * n_query, dim-1)
    
#     # 执行校正
#     # refined_protos = rectifier.rectify(
#     #     init_protos, 
#     #     query_feats, 
#     #     n_iter=5,    # 迭代 5-10 次通常足够
#     #     alpha=0.2,   # 步长
#     #     temperature=10.0
#     # )
#     refined_protos = rectifier.infer(
#         init_protos, 
#         query_feats, 
#         n_iter=5,    # 迭代 5-10 次通常足够
#         lr=0.2,   # 步长
#         temperature=10.0,
#         balance_weight=1.0
#     )
    
#     print("Original Protos Shape:", init_protos.shape)
#     print("Refined Protos Shape: ", refined_protos.shape)
    
#     # 验证是否在流形上 (norm^2 should be -1)
#     norm_sq = lmath.inner(refined_protos, refined_protos, dim=-1)
#     print("Refined Norm^2 mean:", norm_sq.mean().item())