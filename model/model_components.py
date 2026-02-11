import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt.lmath as lmath
from onmt.nn import ToLorentz, ToEuclidean, LorentzNormalization


class EuclideanCrossAttentionBlock(nn.Module):
    """
    修改版欧氏空间交叉注意力模块 (Euclidean Cross-Attention Block).
    
    Changes:
    1. Added 'normalize' flag to control LayerNorm.
    2. Removed Residual Connections (Add).
    
    Structure (normalize=True):
    Input -> MultiheadCrossAttention -> LayerNorm -> FeedForward -> LayerNorm -> Output
    """
    def __init__(self, in_channels, num_heads, dropout=0.0, use_feedforward=False, normalize=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_feedforward = use_feedforward
        self.normalize = normalize  # 新增：控制是否归一化
        
        # 1. 核心注意力层
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, 
                                               num_heads=num_heads, 
                                               dropout=dropout, 
                                               batch_first=True)
        
        # 2. 层归一化 (仅当 normalize=True 时初始化)
        if self.normalize:
            self.norm1 = nn.LayerNorm(in_channels)
        
        # 3. 前馈网络
        if self.use_feedforward:
            if self.normalize:
                self.norm2 = nn.LayerNorm(in_channels)
            
            self.ffn = nn.Sequential(
                nn.Linear(in_channels, 4 * in_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * in_channels, in_channels),
                nn.Dropout(dropout)
            )

    def forward(self, query_input, source_input, output_attentions=False):
        """
        Args:
            query_input:  [Batch, N_q, D] (Image Features) -> Query
            source_input: [Batch, N_k, D] (Text Features)  -> Key/Value
            output_attentions: bool
            
        Returns:
            output: [Batch, N_q, D]
        """
        # 1. Cross-Attention 计算
        # attn_output: [Batch, N_q, D]
        attn_output, attn_weights = self.attention(query=query_input, 
                                                   key=source_input, 
                                                   value=source_input, 
                                                   need_weights=output_attentions)
        
        # 2. [修改] 去除残差连接，可选归一化
        # 原逻辑: x = self.norm1(query_input + attn_output)
        # 新逻辑: 直接使用 attn_output
        x = attn_output
        
        if self.normalize:
            x = self.norm1(x)
        
        # 3. 前馈网络 (如果启用)
        if self.use_feedforward:
            ffn_output = self.ffn(x)
            
            # [修改] 去除残差连接
            # 原逻辑: x = self.norm2(x + ffn_output)
            # 新逻辑: 直接使用 ffn_output
            x = ffn_output
            
            if self.normalize:
                x = self.norm2(x)
            
        if output_attentions:
            return x, attn_weights
        else:
            return x


class LorentzCrossAttention(nn.Module):
    """
    集成了 ToLorentz 和 ToEuclidean 的完整双曲交叉注意力模块。
    对外接口完全是欧氏的，内部计算是全双曲的。
    """
    def __init__(self, c, in_channels, num_heads, use_weight=True, attention_type='full', trans_heads_concat=True, normalize=False):
        super().__init__()
        self.c = c
        self.in_channels = in_channels       # 欧氏维度 D (e.g., 384)
        self.hyperbolic_dim = in_channels + 1 # 双曲维度 D+1 (e.g., 385)
        self.num_heads = num_heads
        
        # 几何变换层
        self.to_lorentz = ToLorentz(c=c)
        self.to_euclidean = ToEuclidean(c=c)

        # 归一化层 (修复: 在 init 中实例化)
        self.normalize = normalize
        if self.normalize:
            self.ln_norm = LorentzNormalization(c=c)
        
        # 计算每个头的空间维度
        # 我们希望最终输出能还原回 in_channels，所以每个头的空间维度是 D // H
        self.head_dim_space = in_channels // num_heads 
        
        self.use_weight = use_weight
        self.attention_type = attention_type
        self.normalize = normalize
        self.trans_heads_concat = trans_heads_concat

        # === 线性层定义 ===
        # 输入: 双曲向量 (D+1) -> 视为环境空间坐标
        # 输出: 变换后的空间分量 (num_heads * head_dim_space)
        self.Wq = nn.Linear(self.hyperbolic_dim, num_heads * self.head_dim_space)
        self.Wk = nn.Linear(self.hyperbolic_dim, num_heads * self.head_dim_space)
        if use_weight:
            self.Wv = nn.Linear(self.hyperbolic_dim, num_heads * self.head_dim_space)

        # 缩放因子 (Scale)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(num_heads * (self.head_dim_space + 1))]))
        self.bias = nn.Parameter(torch.zeros(()))
        
        # === 最终融合层 ===
        if self.trans_heads_concat:
            # 输入: 所有头拼接后的双曲向量 (H * (D_head + 1))
            # 输出: 目标欧氏空间的维度 (D) -> 随后 project 变成 D+1 -> to_euclidean 变成 D
            # 这样保证了输入输出维度严格一致
            input_dim_concat = num_heads * (self.head_dim_space + 1)
            self.final_linear = nn.Linear(input_dim_concat, self.in_channels)

    def cinner(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算全对全的闵可夫斯基内积矩阵 (用于 Attention Score)。
        Input: 
            x: [Batch, Head, N_q, D+1]
            y: [Batch, Head, N_k, D+1]
        Output:
            out: [Batch, Head, N_q, N_k]
        """
        # 1. 克隆 x 以避免原地修改破坏梯度图
        x = x.clone()
        
        # 2. 反转时间维符号: x0 -> -x0
        # narrow(-1, 0, 1) 选中最后一维的第0个元素
        x.narrow(-1, 0, 1).mul_(-1)
        
        # 3. 利用矩阵乘法计算 (-x0*y0 + x1*y1 + ...)
        # y.transpose(-1, -2) 将最后两维转置，适配矩阵乘法
        return x @ y.transpose(-1, -2)

    def lorentzian_centroid(self, x, weight=None, dim=-2, eps=1e-6):
        """
        独立版本的洛伦兹质心计算函数 (Standalone Lorentzian Centroid).
        严格遵循 HELM/Geoopt 中 Lorentz 类的实现逻辑。

        Args:
            x (Tensor): 输入的双曲张量 (包含时间维), shape [..., N, D+1]
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
        if isinstance(self.c, torch.Tensor):
            sqrt_c = self.c.sqrt()
        else:
            # 保持与 x 相同的 device 和 dtype
            sqrt_c = torch.tensor(self.c, device=x.device, dtype=x.dtype).sqrt()
            
        return sqrt_c * ave / denom

    def project(self, x_space):
        """
        辅助函数：根据空间分量计算时间分量，投影回双曲流形
        x_space: [..., D_space]
        return:  [..., D_space + 1]
        """
        # x0 = sqrt( ||x_s||^2 + c )
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return x

    def forward(self, query_input, source_input, output_attentions=False):
        """
        Forward Pass with Shape Tracking
        
        Args:
            query_input:  [Batch, N_q, D] (Euclidean Image Features)
            source_input: [Batch, N_k, D] (Euclidean Text Features, usually N_k=1)
        
        Returns:
            final_output: [Batch, N_q, D] (Euclidean Updated Image Features)
        """
        batch_size, q_len, _ = query_input.size()
        _, kv_len, _ = source_input.size()
        
        # ======================================================
        # Step 1: Geometric Lifting (欧氏 -> 双曲)
        # ======================================================
        # Input: [B, N, D]
        query_hyp = self.to_lorentz(query_input)   # Shape: [B, N_q, D+1]
        source_hyp = self.to_lorentz(source_input) # Shape: [B, N_k, D+1]

        # ======================================================
        # Step 2: Linear Transformation in Ambient Space
        # ======================================================
        # 2.1 Query
        # Linear: [B, N_q, D+1] -> [B, N_q, H * D_head]
        q_space = self.Wq(query_hyp)
        # Reshape: -> [B, N_q, H, D_head]
        q_space = q_space.view(batch_size, q_len, self.num_heads, self.head_dim_space).contiguous()
        # Project: -> [B, N_q, H, D_head + 1]
        qs = self.project(q_space)

        # 2.2 Key
        k_space = self.Wk(source_hyp).view(batch_size, kv_len, self.num_heads, self.head_dim_space).contiguous()
        ks = self.project(k_space) # Shape: [B, N_k, H, D_head + 1]

        # 2.3 Value
        if self.use_weight:
            v_space = self.Wv(source_hyp).view(batch_size, kv_len, self.num_heads, self.head_dim_space).contiguous()
            vs = self.project(v_space) # Shape: [B, N_k, H, D_head + 1]
        else:
            # 如果不使用权重，需要对 source_hyp 进行维度调整以匹配头的数量
            # 这里为了简化，假设必须使用权重以进行空间适配
            raise NotImplementedError("use_weight=False is not supported in this complete block implementation.")

        # ======================================================
        # Step 3: Hyperbolic Attention Mechanism
        # ======================================================
        # Transpose for attention: [B, H, N, D+1]
        qs_t = qs.transpose(1, 2).contiguous() # [B, H, N_q, D_head + 1]
        ks_t = ks.transpose(1, 2).contiguous() # [B, H, N_k, D_head + 1]
        vs_t = vs.transpose(1, 2).contiguous() # [B, H, N_k, D_head + 1]

        # 3.1 Normalize (修复: 调用 init 中实例化的层)
        if self.normalize:
            # 此时 qs_t 是 [B, H, N, D+1]，传入 forward 会自动切片处理空间部分
            qs_t = self.ln_norm(qs_t) 
            ks_t = self.ln_norm(ks_t)

        # 3.2 Calculate Scores (Hyperbolic Distance)
        # cinner output: [B, H, N_q, N_k]
        att_weight = 2 * self.c + 2 * self.cinner(qs_t, ks_t)
        att_weight = att_weight / self.scale + self.bias
        att_weight = nn.Softmax(dim=-1)(att_weight) # [B, H, N_q, N_k]

        # 3.3 Aggregation (Lorentzian Centroid)
        # Output: [B, H, N_q, D_head + 1]
        att_output = self.lorentzian_centroid(vs_t, att_weight)
        att_output = att_output.transpose(1, 2).contiguous() # [B, N_q, H, D_head + 1]

        # ======================================================
        # Step 4: Final Fusion & Output Projection
        # ======================================================
        if self.trans_heads_concat:
            # Flatten heads: [B, N_q, H * (D_head + 1)]
            flat_input = att_output.reshape(batch_size, q_len, -1).contiguous()
            
            # Linear Fusion: Map to target Euclidean dimension D
            # [B, N_q, H*(D_head+1)] -> [B, N_q, D]
            att_output_space = self.final_linear(flat_input)
            
            # Project back to Manifold: [B, N_q, D+1]
            att_output_hyp = self.project(att_output_space)
        else:
            # Fallback (Average): [B, N_q, D_head + 1]
            att_output_hyp = self.lorentzian_centroid(att_output)
            # 注意：这种情况下输出维度会变小，通常不建议用于 Backbone 插入

        # ======================================================
        # Step 5: Geometric Grounding (双曲 -> 欧氏)
        # ======================================================
        # Input: [B, N_q, D+1]
        # Output: [B, N_q, D]
        final_output_euc = self.to_euclidean(att_output_hyp)

        if output_attentions:
            return final_output_euc, att_weight
        else:
            return final_output_euc, None


class LorentzCrossAttentionBlock(nn.Module):
    """
    [Block 2] 洛伦兹交叉注意力块.
    Structure: Input(Euc) -> LorentzCore -> EucOutput -> [LN] -> [FFN] -> [LN] -> Output
    """
    def __init__(self, c, in_channels, num_heads, dropout=0.0, use_feedforward=False, normalize=True):
        super().__init__()
        self.use_feedforward = use_feedforward
        self.normalize = normalize

        # Core: Custom Lorentz Attention Engine
        # 注意：这里的 normalize=True 是指双曲空间内部的归一化，与 Block 外部的 LN 不同
        self.core = LorentzCrossAttention(
            c=c,
            in_channels=in_channels,
            num_heads=num_heads,
            use_weight=True,
            attention_type='full',
            trans_heads_concat=True,
            normalize=False 
        )

        # Wrapper Layers (Applied on Euclidean Output)
        if self.normalize:
            self.norm1 = nn.LayerNorm(in_channels)
        
        if self.use_feedforward:
            if self.normalize:
                self.norm2 = nn.LayerNorm(in_channels)
            self.ffn = nn.Sequential(
                nn.Linear(in_channels, 4 * in_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * in_channels, in_channels),
                nn.Dropout(dropout)
            )

    def forward(self, query_input, source_input, output_attentions=False):
        # 1. Geometric Attention (Euc -> Hyp -> Euc)
        attn_output, attn_weights = self.core(query_input, source_input, output_attentions)
        
        # 2. Post-processing (Standard Euclidean LN)
        x = attn_output
        if self.normalize:
            x = self.norm1(x)
            
        # 3. FFN
        if self.use_feedforward:
            ffn_output = self.ffn(x)
            x = ffn_output
            if self.normalize:
                x = self.norm2(x)
        
        if output_attentions:
            return x, attn_weights
        else:
            return x


class CrossAttentionBlock(nn.Module):
    """
    极简版交叉注意力模块 (Pure Cross-Attention).
    
    Removed:
    1. Attention Mask functionality.
    2. Add & Norm (Residual connections & LayerNorm).
    3. FeedForward Network (FFN).
    
    Operation:
    Output = MultiheadAttention(Query, Key, Value)
    """
    def __init__(self, in_channels, num_heads, dropout=0.0, use_feedforward=False, normalize=True):
        super(CrossAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.use_feedforward = use_feedforward
        self.normalize = normalize

        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  
        )

        if self.normalize:
            self.norm1 = nn.LayerNorm(in_channels)

        if self.use_feedforward:
            if self.normalize:
                self.norm2 = nn.LayerNorm(in_channels)
            self.ffn = nn.Sequential(
                nn.Linear(in_channels, 4 * in_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * in_channels, in_channels),
                nn.Dropout(dropout)
            )

    def forward(self, query_input, source_input, output_attentions=False):
        """
        Args:
            query_input:        [Batch, 1, D] (即 weight_token)
            source_input: [Batch, N, D] (即某个分支的输出)
            
        Returns:
            output:       [Batch, 1, D] (聚合后的特征)
        """
        # query=weight_token, key=input_tensor, value=input_tensor
        # key_padding_mask=None: 显式声明不使用 mask
        attn_output, attn_weights = self.attn(
            query=query_input,
            key=source_input,
            value=source_input,
            need_weights=output_attentions
        )

        x = attn_output

        if self.normalize:
            x = self.norm1(x)

        if self.use_feedforward:
            ffn_output = self.ffn(x)
            x = ffn_output
            if self.normalize:
                x = self.norm2(x)

        if output_attentions:
            return x, attn_weights
        else:
            return x


class HLFormerBlock(nn.Module):
    def __init__(self, config):
        super(HLFormerBlock, self).__init__()

        # ============================================================
        # 1. 欧氏分支 (Euclidean Branch)
        # ============================================================
        # 替换为 1 个标准的 EuclideanCrossAttentionBlock
        self.e_attn = EuclideanCrossAttentionBlock(
            in_channels=config.feat_size,
            num_heads=config.num_attention_heads, 
            dropout=config.dropout_prob
        )

        # ============================================================
        # 2. 洛伦兹分支 (Lorentz Branch)
        # ============================================================
        # 替换为 1 个 LorentzCrossAttention
        # 获取曲率 c
        c_init = getattr(config, 'c', 0.01)

        self.h_attn = LorentzCrossAttentionBlock(
            c=c_init,
            in_channels=config.feat_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_prob
        )

        # ============================================================
        # 3. 融合模块 (Adaptive Interaction)
        # ============================================================
        self.ca = CrossAttentionBlock(
            in_channels=config.feat_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_prob
        ) # 使用精简版 CrossAttention

        self.layer1 = nn.Linear(config.feat_size, config.feat_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer2 = nn.Linear(config.feat_size, config.patch_len) 

        self.sft_factor = config.sft_factor

    def forward(self, image_features, text_features, weight_token=None):
        """
        Args:
            image_features: [Batch, N, D] (Query)
            text_features:  [Batch, M, D] (Key/Value)
            weight_token:   [Batch, 1, D] (可选，全局引导特征)
        """
        
        outputs = []

        # 1. 欧氏分支前向传播
        o_e, attn_weight_e = self.e_attn(query_input=image_features, source_input=text_features, output_attentions=True)
        outputs.append(o_e.unsqueeze(-1)) # [B, N, D, 1]

        # 2. 洛伦兹分支前向传播
        o_h, atten_weight_h = self.h_attn(query_input=image_features, source_input=text_features, output_attentions=True)
        outputs.append(o_h.unsqueeze(-1)) # [B, N, D, 1]

        # 3. 融合 (Fusion)
        oo = torch.cat(outputs, dim=-1) # [B, N, D, 2]

        if weight_token is None:
            # 计算全局 Token
            mean_oo = torch.mean(oo, dim=-1) # [B, N, D]
            weight_token = torch.mean(mean_oo, dim=1, keepdim=True) # [B, 1, D]
        else:
            weight_token = weight_token.to(oo.device).type_as(oo).repeat(oo.shape[0], 1, 1)

        weight = []
        # 计算每个分支的融合权重 (无需 Mask)
        for i in range(oo.shape[-1]):
            # 使用全局 weight_token 查询分支特征
            temp_token, _ = self.ca(weight_token, oo[..., i], output_attentions=True)
            weight.append(temp_token)

        weight = torch.cat(weight, dim=1)    # [B, 2, D]
        weight = self.layer1(weight)
        weight = self.dropout(F.relu(weight))
        weight = self.layer2(weight)         # [B, 2, N]
        
        # Softmax 归一化权重
        # permute -> [B, N, 2]
        weight = F.softmax(weight.permute(0, 2, 1) / self.sft_factor, dim=-1)
        
        # 加权求和输出 -> [B, N, D]
        out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)

        return out


# if __name__ == '__main__':
#     from method.train_lorentz_hl import args
#     args.patch_len = 49
#     args.num_attention_heads = 8
#     args.dropout_prob = 0.0
#     args.sft_factor = 0.6

#     block = HLFormerBlock(args)
    
#     image_features = torch.randn(1, args.patch_len, args.feat_size)
#     text_features = torch.randn(1, 1, args.feat_size)
    
#     output = block(image_features, text_features)
#     print(output.shape)
