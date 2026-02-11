import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    修改版欧氏空间交叉注意力模块 (Cross-Attention Block).
    
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


class SpatialAttentionBlock(nn.Module):
    """
    [Block 2] Channel Attention with Text Injection.
    Replaces LorentzCrossAttention with a channel-wise attention mechanism that injects text features.
    
    Structure:
    1. Global Average Pooling on Query (Image) -> Context [B, D]
    2. Concat with Source (Text) [B, D] -> [B, 2D]
    3. MLP (SE-Block) -> [B, D]
    4. Channel Modulation: Query + Context
    5. [LN] -> [FFN] -> [LN] -> Output
    """
    def __init__(self, in_channels, dropout=0.0, use_feedforward=False, normalize=False):
        super().__init__()
        self.use_feedforward = use_feedforward
        self.normalize = normalize
        
        # Note: 'c' (curvature) and 'num_heads' are kept for compatibility but not used in the core logic.
        
        # Core: Channel Attention (SE-Block style with Text Injection)
        # Input: [B, 2*C] (Image Context + Text Context) -> Output: [B, C]
        self.se_block = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.Sigmoid(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        # Wrapper Layers (Applied on Output)
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
            query_input:  [Batch, N, C] (Image Features)
            source_input: [Batch, M, C] (Text Features)
            output_attentions: bool (Unused for channel attention, returns None weights)
        """
        B, N, C = query_input.shape
        
        # 1. Global Average Pooling on Image Features -> [B, C]
        # query_input is [B, N, C], so mean over N (dim=1)
        image_context = query_input.mean(dim=1) 
        
        # 2. Process Text Features -> [B, C]
        # source_input is [B, M, C]. If M > 1, take mean.
        if source_input.dim() == 3:
            text_context = source_input.mean(dim=1)
        else:
            text_context = source_input
            
        # 3. Concatenate -> [B, 2C]
        combined_context = torch.cat([image_context, text_context], dim=-1)
        
        # 4. MLP (SE Block) -> [B, C]
        channel_attn = self.se_block(combined_context)
        
        # 5. Center the attention weights
        channel_attn = channel_attn - channel_attn.mean(dim=-1, keepdim=True)
        
        # 6. Channel Modulation (Add to original features)
        # Broadcast [B, C] to [B, N, C]
        x = query_input + channel_attn.unsqueeze(1)
        
        # 7. Post-processing (LN -> FFN -> LN)
        if self.normalize:
            x = self.norm1(x)
            
        if self.use_feedforward:
            ffn_output = self.ffn(x)
            x = ffn_output
            if self.normalize:
                x = self.norm2(x)
        
        if output_attentions:
            return x, None # No spatial attention weights in this mechanism
        else:
            return x


class MAIMBlock(nn.Module):
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
        super(MAIMBlock, self).__init__()
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


class DAB(nn.Module):
    def __init__(self, config):
        super(DAB, self).__init__()

        # ============================================================
        # 1. 欧氏分支 (Euclidean Branch)
        # ============================================================
        # 替换为 1 个标准的 CrossAttentionBlock
        self.c_attn = CrossAttentionBlock(
            in_channels=config.feat_size,
            num_heads=config.num_attention_heads, 
            dropout=config.dropout_prob
        )

        # ============================================================
        # 2. 洛伦兹分支 (Lorentz Branch)
        # ============================================================
        # 替换为 1 个 LorentzCrossAttention
        # 获取曲率 c

        self.s_attn = SpatialAttentionBlock(
            in_channels=config.feat_size,
            dropout=config.dropout_prob
        )

        # ============================================================
        # 3. 融合模块 (Adaptive Interaction)
        # ============================================================
        self.maim = MAIMBlock(
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
            image_features: [Batch, N, C] (Query)
            text_features:  [Batch, M, C] (Key/Value)
            weight_token:   [Batch, 1, C] (可选，全局引导特征)
        """
        
        outputs = []

        # 1. 欧氏分支前向传播
        o_c, attn_weight_c = self.c_attn(query_input=image_features, source_input=text_features, output_attentions=True)
        outputs.append(o_c.unsqueeze(-1)) # [B, N, D, 1]

        # 2. 洛伦兹分支前向传播
        o_s, atten_weight_s = self.s_attn(query_input=image_features, source_input=text_features, output_attentions=True)
        outputs.append(o_s.unsqueeze(-1)) # [B, N, D, 1]

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
            temp_token, _ = self.maim(weight_token, oo[..., i], output_attentions=True)
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


if __name__ == '__main__':
    import sys
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_DIR)

    from method.train_han_hl_evo_lgf import args
    args.patch_len = 49
    args.num_attention_heads = 8
    args.dropout_prob = 0.0
    args.sft_factor = 0.6

    block = DAB(args)
    
    image_features = torch.randn(1, args.patch_len, args.feat_size)
    text_features = torch.randn(1, 1, args.feat_size)
    
    output = block(image_features, text_features)
    print(output.shape)
