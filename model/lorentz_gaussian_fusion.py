import torch
import torch.nn.functional as F
from onmt import lmath  

class LorentzGaussianFusion:
    """
    适配洛伦兹模型 (Lorentz Model) 的双曲高斯原型融合模块。
    
    包含功能：
    1. estimate_parameters: 基于直推式 (Transductive) 设置估计双曲均值与标量方差。
    2. tangent_space_fusion (方案A): 切空间投影线性融合。
    3. geodesic_fusion (方案B, 推荐): 基于不确定度的测地线插值融合。
    """
    
    def __init__(self, curvature=0.01):
        """
        Args:
            curvature (float or Tensor): 双曲空间的曲率 c。
                                         注意: lmath 库通常使用 k = 1/c 作为参数。
        """
        self.c = curvature

    def _get_k(self, device, dtype):
        """
        辅助函数: 根据当前的 c 计算 lmath 所需的参数 k = 1/c
        """
        if isinstance(self.c, torch.Tensor):
            k = 1.0 / self.c.to(device)
        else:
            k = torch.tensor(1.0 / self.c, device=device, dtype=dtype)
        return k

    def estimate_parameters(self, support_feats, query_feats, support_labels, num_classes):
        """
        利用 Support (带标签) 和 Query (无标签) 集估计类别的原型 (均值) 和不确定度 (方差)。
        
        Args:
            support_feats: (N_sup, D+1) 支持集洛伦兹特征
            query_feats: (N_query, D+1) 查询集洛伦兹特征
            support_labels: (N_sup) 支持集标签
            num_classes: 类别数量 (Way)
            
        Returns:
            refined_means: (num_classes, D+1) 修正后的双曲原型
            refined_vars: (num_classes, 1) 修正后的标量方差
        """
        device = support_feats.device
        dtype = support_feats.dtype
        k = self._get_k(device, dtype)
        
        # =======================================================
        # 1. 初始原型估计 (Initial Prototype Estimation)
        # =======================================================
        # 仅使用 Support Set，利用 lmath.lorentz_mean_centroid (投影质心均值)
        initial_protos = []
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                # lmath.lorentz_mean_centroid 接受 c 作为参数
                # input shape: (1, N_sample, D), dim=1
                proto = lmath.lorentz_mean_centroid(
                    support_feats[mask].unsqueeze(0), dim=1, c=self.c
                ).squeeze(0)
            else:
                # 极其罕见的情况：某类无 Support 样本，默认设为原点
                # 原点在 Lorentz 模型中通常为 (sqrt(k), 0, ..., 0)
                # 可以通过 expmap0(0向量) 获得
                zero_vec = torch.zeros(support_feats.size(-1)-1, device=device, dtype=dtype)
                proto = lmath.expmap0(zero_vec, k=k)
            initial_protos.append(proto)
        
        initial_protos = torch.stack(initial_protos) # (N_way, D+1)

        # =======================================================
        # 2. 直推式校准 (Transductive Calibration)
        # =======================================================
        # 计算 Query 到初始原型的双曲距离
        # (N_query, 1, D) vs (1, N_way, D) -> (N_query, N_way)
        dists = lmath.dist(
            query_feats.unsqueeze(1), 
            initial_protos.unsqueeze(0), 
            k=k
        )
        
        # 将距离转换为概率 (Soft Assignment)
        # 假设服从双曲高斯分布: P(x) ~ exp(-d^2 / sigma^2)
        # Scale=10.0 是经验温度系数 (相当于 1/sigma^2)
        logits = -dists.pow(2) * 1.0
        query_prob = F.softmax(logits, dim=1) # (N_query, N_way)

        # =======================================================
        # 3. 参数重估与精炼 (Refined Estimation)
        # =======================================================
        refined_means = []
        refined_vars = []

        for c in range(num_classes):
            # 获取该类的硬标签样本 (Support)
            sup_c = support_feats[support_labels == c]
            n_sup = sup_c.size(0)
            
            # 构造权重: Support 权重为 1, Query 权重为预测概率
            w_sup = torch.ones(n_sup, device=device, dtype=dtype)
            w_query = query_prob[:, c]
            
            # 拼接特征与权重
            all_feats = torch.cat([sup_c, query_feats], dim=0)
            all_weights = torch.cat([w_sup, w_query], dim=0)
            sum_w = all_weights.sum() + 1e-8
            
            # --- A. 估计均值: 切空间加权平均 (Tangent Space Weighted Mean) ---
            # 原因: lmath 没有提供带权重的 Lorentz Mean 函数，
            # 因此我们使用 "LogMap -> Weighted Mean -> ExpMap" 策略组合原子 API 实现。
            
            # 1. 投影到原点切空间 (调用 lmath.logmap0)
            v_all = lmath.logmap0(all_feats, k=k)
            
            # 2. 在切空间进行欧氏加权平均
            v_mean = (v_all * all_weights.unsqueeze(1)).sum(0) / sum_w
            
            # 3. 投影回双曲空间 (调用 lmath.expmap0)
            mu_new = lmath.expmap0(v_mean, k=k)
            refined_means.append(mu_new)
            
            # --- B. 估计方差: 标量方差 (Scalar Variance) ---
            # 计算所有样本到新均值的双曲距离平方的加权平均
            dists_c = lmath.dist(all_feats, mu_new, k=k) # (N_all,)
            var_c = (dists_c.pow(2) * all_weights).sum() / sum_w
            refined_vars.append(var_c + 1e-5) # 防止方差为0

        refined_means = torch.stack(refined_means) # (N_way, D+1)
        refined_vars = torch.stack(refined_vars).unsqueeze(1) # (N_way, 1)

        return refined_means, refined_vars

    def tangent_space_fusion(self, mu1, var1, mu2, var2):
        """
        【方案 A】切空间投影融合 (Tangent Space Fusion)
        原理: 将原型投影到切空间，利用欧氏线性加权进行融合，再投影回双曲面。
        优点: 计算快速，数值稳定。
        缺点: 在远离原点处存在几何畸变。
        
        Args:
            mu1, mu2: (N_way, D+1) 两个双曲原型
            var1, var2: (N_way, 1) 对应的标量方差
        """
        device = mu1.device
        k = self._get_k(device, mu1.dtype)
        
        # 1. 映射到原点切空间 (调用 lmath API)
        v1 = lmath.logmap0(mu1, k=k)
        v2 = lmath.logmap0(mu2, k=k)
        
        # 2. 计算融合权重 (基于逆方差)
        # 方差越小，权重越大。w1 对应 mu1，故分子为 var2
        sum_var = var1 + var2 + 1e-8
        w1 = var2 / sum_var
        w2 = var1 / sum_var
        
        # 3. 欧氏空间线性融合
        v_fused = w1 * v1 + w2 * v2
        
        # 4. 映射回双曲空间 (调用 lmath API)
        mu_fused = lmath.expmap0(v_fused, k=k)
        
        return mu_fused

    def geodesic_fusion(self, mu1, var1, mu2, var2):
        """
        【方案 B】测地线插值融合 (Geodesic Interpolation Fusion) - 推荐用于遥感
        原理: 在双曲流形上，沿着连接两个原型的最短路径 (测地线) 进行插值。
        优点: 几何完全保真，无畸变，适合层级深 (High Norm) 的特征。
        
        Args:
            mu1, mu2: (N_way, D+1) 两个双曲原型
            var1, var2: (N_way, 1) 对应的标量方差
        """
        device = mu1.device
        dtype = mu1.dtype
        k = self._get_k(device, dtype)
        sqrt_k = torch.sqrt(k)

        # 1. 计算插值系数 lambda (代表距离 mu1 的比例)
        # 我们希望融合点靠近方差小的一端。
        # 如果 var1 极小 -> 我们希望结果靠近 mu1 -> 距离 mu1 的比例 lambda 应趋近 0。
        # 因此分子应该是 var1。
        sum_var = var1 + var2 + 1e-8
        lam = var1 / sum_var  # (N_way, 1)
        
        # 2. 计算两点间度量距离 D (Metric Distance)
        # lmath.dist 返回的是真实的度量距离: D = sqrt(k) * acosh(...)
        D = lmath.dist(mu1, mu2, k=k).view(-1, 1) 
        
        # 3. 【关键修正】转换为双曲角 zeta (Hyperbolic Angle/Parameter)
        # 测地线插值公式中的 sinh 需要参数距离 zeta，而不是度量距离 D。
        # 关系: D = sqrt(k) * zeta  =>  zeta = D / sqrt(k)
        zeta = D / sqrt_k

        # 4. 测地线插值公式 (Geodesic Interpolation)
        # p(t) = [sinh((1-t)zeta) / sinh(zeta)] * p1 + [sinh(t*zeta) / sinh(zeta)] * p2
        
        # 数值稳定性处理: 当 zeta -> 0 (两点重合) 时，sinh(zeta) -> 0，会导致 NaN。
        # 阈值设为 1e-5
        is_close = zeta < 1e-5
        zeta_safe = zeta.clone()
        zeta_safe[is_close] = 1.0  # 避免分母为 0，这部分结果随后会被 mask 覆盖
        
        sinh_zeta = torch.sinh(zeta_safe)
        
        # 计算双曲正弦系数
        # 注意: 这里使用 zeta (参数距离) 进行 sinh 运算
        coeff1 = torch.sinh((1.0 - lam) * zeta_safe) / sinh_zeta
        coeff2 = torch.sinh(lam * zeta_safe) / sinh_zeta
        
        mu_fused = coeff1 * mu1 + coeff2 * mu2
        
        # 5. 恢复重合点
        # 当 mu1 和 mu2 重合时，插值结果理应就是 mu1 (或 mu2)
        if is_close.any():
            mu_fused[is_close.squeeze()] = mu1[is_close.squeeze()]
        
        return mu_fused


# if __name__ == '__main__':
#     # 测试切空间投影融合
#     # mu1 = torch.randn(5, 3)
#     # var1 = torch.rand(5, 1)
#     # mu2 = torch.randn(5, 3)
#     # var2 = torch.rand(5, 1)
    
#     way, shot, query = 5, 1, 15
#     dim = 384
#     support_feats = torch.randn(way*shot, dim+1)
#     query_feats = torch.randn(way*query, dim+1)
#     support_labels = torch.arange(way).unsqueeze(1).repeat(1, shot).view(-1)

#     gen_support_feats = torch.randn(way*shot, dim+1)
#     gen_support_labels = torch.arange(way).unsqueeze(1).repeat(1, shot).view(-1)

#     fusion = LorentzGaussianFusion()

#     mu1, var1 = fusion.estimate_parameters(support_feats, query_feats, support_labels, way)
#     mu2, var2 = fusion.estimate_parameters(gen_support_feats, query_feats, gen_support_labels, way)
    
#     print("mu1:", mu1.shape)
#     print("var1:", var1.shape)
#     print("mu2:", mu2.shape)
#     print("var2:", var2.shape)

#     mu_fused = fusion.tangent_space_fusion(mu1, var1, mu2, var2)
#     print("切空间投影融合结果:", mu_fused.shape)
    
#     # # 测试测地线插值融合
#     mu_fused_geo = fusion.geodesic_fusion(mu1, var1, mu2, var2)
#     print("测地线插值融合结果:", mu_fused_geo.shape)