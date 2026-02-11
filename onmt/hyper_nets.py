import torch.nn as nn
from onmt.lorentz import Lorentz
import math
import torch
import onmt.lmath as lmath
from onmt.nn import ToLorentz, ToEuclidean, LorentzNormalization

class LorentzLinear(nn.Module):
    """
        Perform the Lorentz linear transformation.

        args:
            in_features, out_features, bias: Same as nn.Linear
            dropout: Dropout rate in lorentz linear
            manifold: THe manifold that the linear layer operated in.
            nonlin: Non-linear function before the linear operation.
            merge: If set to True, it means that the input has the shape of [..., head_num, head_dim], and the output will has the shape of [..., head_num * head_dim]. The heads are merged.
            head_num: If `merge` is set to True, then head_num specifies the number of heads in input, otherwise it means that the output should be split into `head_num` heads, i.e., [..., head_num, head_dim]. If set to 0, then it is a normal lorentz linear layer.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=  None,
                 dropout=0.1,
                 manifold=Lorentz(),
                 nonlin=None,
                 head_num=0,
                 merge=False):
        super().__init__()
        self.nonlin = nonlin
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.head_num = head_num
        self.merge = merge
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)

    def forward(self, x, bias=None):
        if self.nonlin is not None:
            x = self.nonlin(x)
        if not self.merge:
            x = self.weight(self.dropout(x))
            if self.head_num > 0:
                x = x.view(x.shape[0], x.shape[1], self.head_num, -1)
        else:
            x = self.weight(
                self.dropout(x.flatten(-2)))
        # The following code has some inconsistency to Eq.7 in the paper. When calculating the time axis, 
        # we do not consider the bias in Eq.7, while we add the bias before determining time axis. 
        # It is a small bug here. However, both methods give mathematically correct results.
        # For reproducibility, we leave it unchanged here.
        if bias is not None:
            x = x + bias
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - self.manifold.k) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features // self.head_num if self.head_num > 0 else self.in_features
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)




class LorentzMultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """
    def __init__(self,
                 head_count,
                 model_dim,
                 manifold = Lorentz(),
                 dropout=0.1,
                 wid=None
                 ):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(LorentzMultiHeadedAttention, self).__init__()
        self.manifold = manifold
        self.head_count = head_count

        self.linear_keys = LorentzLinear(model_dim,
                                         head_count * self.dim_per_head,
                                         dropout=dropout,
                                         manifold=manifold,
                                         head_num=head_count)
        self.linear_values = LorentzLinear(model_dim,
                                           head_count * self.dim_per_head,
                                           dropout=dropout,
                                           manifold=manifold,
                                           head_num=head_count)
        self.linear_query = LorentzLinear(model_dim,
                                          head_count * self.dim_per_head,
                                          dropout=dropout,
                                          manifold=manifold,
                                          head_num=head_count)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(model_dim)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.wid = wid

    def generate_gauss_weight(self, props_len, width):

        center = torch.arange(props_len).cuda() / props_len
        width = width*torch.ones(props_len).cuda()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]
    def forward(self,
                key,
                value,
                query,
                mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
        """



        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, head_count, dim_per_head)
            return x.transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).reshape(batch_size,-1,head_count*dim_per_head)  #.contiguous()
            # .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        # print(f"key shape:{key.shape}")
        key = shape(key)
        value = shape(value)
        
        query = shape(query)

        attn = (2 +
                2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attn.shape[-1], self.wid)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attn = attn * gmm_mask
        if mask is not None:
            mask = (1-mask).unsqueeze(1)  # [B, 1, 1, T_values]
            mask = mask.to(torch.bool)
            attn = attn.masked_fill(mask, -1e18)
        attn = self.softmax(attn)

        context = self.manifold.mid_point(value, attn)

        context = unshape(context)


        return context

    def update_dropout(self, dropout):
        self.dropout.p = dropout


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
            return final_output_euc