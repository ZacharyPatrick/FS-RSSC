# https://github.com/danczs/Visformer/blob/main/models.py

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .weight_init import to_2tuple, trunc_normal_
import torch.nn.functional as F

from model.rerank_Controller import rerank_Controller
from model.support_Controller import support_Controller


__all__=[
    'visformer_tiny', 'visformer_tiny_80'
]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """ Layernorm f or channels of '2d' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__([num_channels, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    

class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., group=8, spatial_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2: #net setting
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.drop = nn.Dropout(drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, stride=1, padding=1,
                                   groups=self.group, bias=False)
            self.act2 = act_layer()
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)

        if self.spatial_conv:
            x = self.conv2(x)
            x = self.act2(x)

        x = self.conv3(x)
        x = self.drop(x)
        return x


class MlpAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim) 
        # self.act = nn.GELU()
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.fc2 = nn.Linear(in_dim, out_dim) 
        # self.norm = nn.LayerNorm(out_dim)     

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim_ratio=1., qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = head_dim ** qk_scale_factor

        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, stride=1, padding=0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.qkv(x)
        qkv = rearrange(x, 'b (x y z) h w -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ( (q * self.scale) @ (k.transpose(-2,-1) * self.scale) )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        
        x = rearrange(x, 'b y (h w) z -> b (y z) h w', h=H, w=W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim_ratio=1., mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm,
                 group=8, attn_disabled=False, spatial_conv=False):
        super().__init__()
        self.attn_disabled = attn_disabled
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if not attn_disabled:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, head_dim_ratio=head_dim_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       group=group, spatial_conv=spatial_conv) # new setting

    def forward(self, x):
        if not self.attn_disabled:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_pe = norm_layer is not None
        if self.norm_pe:
            self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) does not match model ({self.img_size[1]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.norm_pe:
            x = self.norm(x)
        return x


class Visformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, init_channels=32, num_classes=64, embed_dim=384, depth=12, 
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.5, norm_layer=LayerNorm, attn_stage='111', pos_embed=True, spatial_conv='111',
                 vit_embedding=False, group=8, pool=True, conv_init=False, embedding_norm=None, small_stem=False): #drop_path_rate=0.
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.vit_embedding = vit_embedding
        self.pool = pool
        self.conv_init = conv_init
        self.log_tau = torch.nn.Parameter(torch.tensor(np.log(0.1)))#0.07

        
        if isinstance(depth, list) or isinstance(depth, tuple):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
        self.pos_embed = pos_embed
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # stage 1
        if self.vit_embedding:
            self.using_stem = False
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=embed_dim,
                                           norm_layer=embedding_norm)
            img_size //= 16
        else:
            if self.init_channels is None:
                self.using_stem = False
                self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=8, in_chans=3, embed_dim=embed_dim//2,
                                               norm_layer=embedding_norm)
                img_size //= 8
            else:
                self.using_stem = True
                if not small_stem:
                    self.stem = nn.Sequential(
                        nn.Conv2d(3, self.init_channels, 7, stride=2, padding=3, bias=False),
                        BatchNorm(self.init_channels),
                        nn.ReLU(inplace=True)
                    )
                    img_size //= 2
                    self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, in_chans=self.init_channels,
                                                   embed_dim=embed_dim//2, norm_layer=embedding_norm)
                    img_size //= 4
                else:
                    self.stem = nn.Sequential(
                        nn.Conv2d(3, self.init_channels, 3, stride=1, padding=1, bias=False),
                        BatchNorm(self.init_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.init_channels, self.init_channels, 3, stride=1, padding=1, bias=False),
                        BatchNorm(self.init_channels),
                        nn.ReLU(inplace=True),
                    )
                    self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, in_chans=self.init_channels,
                                                   embed_dim=embed_dim // 2, norm_layer=embedding_norm)
                    img_size //= 4

        if self.pos_embed:
            if self.vit_embedding:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))
            else:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim//2, img_size, img_size))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage1 = nn.ModuleList([
            Block(
                dim=embed_dim//2, num_heads=num_heads, head_dim_ratio=0.5, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[0] == '0'), spatial_conv=(spatial_conv[0] == '1',)
            )
            for i in range(self.stage_num1)
        ])

        
        #stage2
        if not self.vit_embedding:
            self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim//2, embed_dim=embed_dim,
                                           norm_layer=embedding_norm)
            img_size //= 2
            if self.pos_embed:
                self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))
        self.stage2 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[1] == '0'), spatial_conv=(spatial_conv[1] == '1'),  
            )
            for i in range(self.stage_num1, self.stage_num1+self.stage_num2)
        ])

        # stage 3
        if not self.vit_embedding:
            self.patch_embed3 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim*2,
                                           norm_layer=embedding_norm)
            img_size //= 2
            if self.pos_embed:
                self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim*2, img_size, img_size))
        self.stage3 = nn.ModuleList([
            Block(
                dim=embed_dim*2, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[2] == '0'), spatial_conv=(spatial_conv[2] == '1'), 
            )
            for i in range(self.stage_num1+self.stage_num2, depth)
        ])

        # head
        if self.pool:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if not self.vit_embedding:
            self.norm = norm_layer(embed_dim*2)
            self.head = nn.Linear(embed_dim*2, num_classes)
        else:
            self.norm = norm_layer(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)

        # weights init
        if self.pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            if not self.vit_embedding:
                trunc_normal_(self.pos_embed2, std=0.02)
                trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        if self.using_stem:
            x = self.stem(x)

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
            x = self.pos_drop(x)
        for b in self.stage1:
            x = b(x)

        # stage 2
        if not self.vit_embedding:
            x = self.patch_embed2(x)
            if self.pos_embed:
                x = x + self.pos_embed2
                x = self.pos_drop(x)
        for b in self.stage2:
            x = b(x)

        # stage3
        if not self.vit_embedding:
            x = self.patch_embed3(x)
            if self.pos_embed:
                x = x + self.pos_embed3
                x = self.pos_drop(x)
        for b in self.stage3:
            x = b(x)

        # head
        x = self.norm(x)    #128, 384, 7, 7 finetune:75,384,7,7
        if self.pool:
            x = self.global_pooling(x)  #128, 384, 1, 1
        else:
            x = x[:, :, 0, 0]

        logit = self.head( x.view(x.size(0), -1) )  #128, 64
        return logit, x.squeeze()   #128, 384
        # return x.view(x.size(0), -1), x.squeeze()
    


    def fusion(self, x, semantic_prompt, args):
        prompt1 = self.t2i(semantic_prompt)
        prompt2 = self.t2i2(semantic_prompt)

        if self.using_stem:
            x = self.stem(x)

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
            x = self.pos_drop(x)
        for b in self.stage1:
            x = b(x)

        # stage 2
        if not self.vit_embedding:
            x = self.patch_embed2(x)
            if self.pos_embed:
                x = x + self.pos_embed2
                x = self.pos_drop(x)
        stage = 2.0
        for b in self.stage2:
            if np.absolute(stage - args.stage) < 1e-6:
                B, C, H, W = x.shape
                #channel
                context = x.view(B, C, -1).mean(-1)
                context = torch.cat([context, prompt2], dim=-1)
                context = self.se_block(context)#two-layer MLP
                context = context - context.mean(dim=-1, keepdim=True)
                x = x + context.view(B, C, 1, 1)
                #spatial
                prompt1 = prompt1.view(B, C, 1, 1).repeat(1, 1, 1, W)
                x = torch.cat([x, prompt1], dim=2)
            x = b(x)
            stage += 0.1
        if 2 <= args.stage < 3:
            x = x[:, :, :H]

        # stage3
        if not self.vit_embedding:
            x = self.patch_embed3(x)
            if self.pos_embed:
                x = x + self.pos_embed3
                x = self.pos_drop(x)
        stage = 3.0
        for b in self.stage3:
            if np.absolute(stage - args.stage) < 1e-6:
                B, C, H, W = x.shape    
                #channel
                # context = x.view(B, C, -1).mean(-1) 
                # context = torch.cat([context, prompt2], dim=-1)
                # context = self.se_block(context)    
                # context = context - context.mean(dim=-1, keepdim=True)  
                # x = x + context.view(B, C, 1, 1)    
                # image_features = x.view(B, -1, C).contiguous()    # [B, H*W, C]
                image_features = x.permute(0, 2, 3, 1).reshape(B, H*W, C).contiguous()
                text_features = prompt2.unsqueeze(1).contiguous()    # [B, 1, C]
                out = self.dab_block(image_features, text_features)
                # out = out.reshape(B, C, H, W).contiguous()
                out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
                # x = x + self.gate * out
                # x = x + torch.sigmoid(self.gate) * out
                g_max = 1.3
                x = x + g_max * torch.tanh(self.gate) * out    # [B, C, H, W]
                #spatial
                prompt1 = prompt1.view(B, C, 1, 1).repeat(1, 1, 1, W)    # [B, C, 1, W]
                x = torch.cat([x, prompt1], dim=2)    # [B, C, H+1, W]
            x = b(x)
            stage += 0.1

        # head
        x = self.norm(x)
        if self.pool:
            if args.stage < 3:
                x = self.global_pooling(x)
            else:
                B, C, H, W = x.shape
                x = x.view(B, C, -1)[:, :, :(H - 1) * W + 1].mean(-1)
        else:
            x = x[:, :, 0, 0]

        logit = self.head( x.view(x.size(0), -1) )
        return logit, x.squeeze()
        # return x.view(x.size(0), -1), x.squeeze()


class Visformer_DCMA_APCM_CPF(nn.Module):
    """
    Visformer_DCMA_APCM_CPF类：用于以APCM（Adaptive Prototype Construction Module）为核心的Few-Shot图像分类推理

    主要结构与功能说明：
    1. 主体结构
        - 继承自nn.Module，实现了元学习（Meta-Learning）领域下的Few-Shot分类方法。
        - 依赖于Visformer作为底层视觉特征提取Backbone。

    2. 构造方法（__init__）
        - 根据参数args的backbone字段选择编码器，目前仅支持"visformer"。
        - 提供支持集/重排序控制器（support_controller, rerank_controller），
          分别实现论文方法中的MLP2、MLP3，用于聚合支持集特征及多样本权重自适应。
        - 定义phi投影层，仅对推断时的特征集合Z_j进行线性变换，投影到相同维度空间。
    
    3. 内部辅助方法
        - _get_way：获得当前分类任务的类别数（用于区分训练/测试阶段）。
        - _cos_dist：对输入特征a、b计算归一化余弦距离矩阵（1-cosine_sim）。
    
    4. Few-Shot分类主流程（_forward_from_feats）：
        - 输入支持集与查询集特征，执行两阶段原型生成与类别判别流程：
            * Stage A: 对每个类，通过MLP2（support_controller）根据类内距离自适应权重支持集特征，
              获得临时原型p_prime。
            * Stage B: 对每个类，使用phi投影后的"外部"特征集合Z_j（其他类+可能有查询集），
              查找距离p'_j最近的m个样本并分组（H_j/barH_j），与其他类原型组合后，通过MLP3
              （rerank_controller）自适应合成最终原型p_final。
        - 分类时以查询集特征与最终原型的余弦距离为准，输出logits。
    
    5. forward方法
        - 输入原始支持集和查询集图像，先通过编码器提取特征，再走主推理流程，返回分类得分与特征。

    6. fusion方法
        - 支持与外部语义提示融合推理（如多模态或prompt场景），
          调用编码器的fusion接口获取融合特征，流程同forward。

    适用场景：
        支持inductive（仅支持集）及transductive（支持集+查询集）设定。

    主要参数说明（args）：
        - way: 类别数
        - shot: 每类别支持样本数
        - rerank: 外部样本选择数m
        - dim: 特征维度
        - setting: 任务设置（如inductive, transductive）
        - backbone: 支持的视觉主干网络
        - num_classes: 类别总数
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # ---------------- Backbone ----------------
        model_name = args.backbone
        if model_name == "visformer":
            self.encoder = visformer_tiny(num_classes=args.num_classes)
        else:
            raise ValueError("Model not found")

        # ---------------- Controllers (paper's MLP2 / MLP3) ----------------
        self.support_controller = support_Controller(
            self.args.shot * self.args.shot, self.args.shot, self.args.shot
        )
        self.rerank_controller = rerank_Controller(
            self.args.rerank * 2 + 2, self.args.rerank, self.args.rerank + 1
        )

        # ---------------- phi(.) projection ONLY for Z_j ----------------
        self.phi = nn.Linear(args.dim, args.dim)
        nn.init.xavier_normal_(self.phi.weight)
        if self.phi.bias is not None:
            nn.init.zeros_(self.phi.bias)

    def _get_way(self) -> int:
        if self.training:
            return int(self.args.way)
        return int(getattr(self.args, "validation_way", self.args.way))

    @staticmethod
    def _cos_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Cosine distance between all pairs.
        a: [N, D], b: [M, D] -> [N, M]
        dist = 1 - cos(a_i, b_j)
        """
        a_n = F.normalize(a, dim=-1, eps=eps)
        b_n = F.normalize(b, dim=-1, eps=eps)
        sim = a_n @ b_n.t()  # [N, M]
        return 1.0 - sim.clamp(-1.0, 1.0)

    def _forward_from_feats(self, data_shot: torch.Tensor, data_query: torch.Tensor) -> torch.Tensor:
        """
        Run APCM (Cosine) given support/query features.
        data_shot: [way*shot, dim]
        data_query: [num_q, dim]
        return logits: [num_q, way]
        """
        device = data_shot.device
        dtype = data_shot.dtype

        way = self._get_way()
        shot_k = int(self.args.shot)
        dim = data_shot.shape[-1]
        num_q = data_query.shape[0]

        data_shot_category = data_shot.reshape(way, shot_k, dim)  # [way, shot, dim]

        # =====================================================================
        # Stage A: a_j = MLP2(G_j) with RAW cosine-distance matrix, p'_j = mean(a*x)
        # =====================================================================
        p_prime = torch.zeros(way, dim, device=device, dtype=dtype)
        a_list = torch.zeros(way, shot_k, device=device, dtype=dtype)

        for j in range(way):
            X_j = data_shot_category[j]                              # [shot, dim]
            G_j = self._cos_dist(X_j, X_j)                           # [shot, shot] raw distances

            # feed RAW flattened distance matrix (NO softmax(-d))
            a_j = self.support_controller(G_j.view(1, -1)).squeeze(0)  # [shot]
            a_list[j] = a_j

            X_scaled = X_j * a_j.unsqueeze(1)                         # [shot, dim]
            p_prime[j] = X_scaled.mean(dim=0)                         # [dim]

        # =====================================================================
        # Stage B: Z_j, phi(Z_j), select H_j, build RAW dis, MLP3, union + mean
        # =====================================================================
        p_final = torch.zeros(way, dim, device=device, dtype=dtype)

        setting = getattr(self.args, "setting", "inductive")
        m = int(self.args.rerank)

        for j in range(way):
            # ---- build Z_j with semantics ----
            Z_parts = []

            if setting != "inductive":
                Z_parts.append(data_query)  # transductive: include queries

            for k in range(way):
                if k == j:
                    continue
                Z_parts.append(data_shot_category[k])

            Z_j = torch.cat(Z_parts, dim=0)  # [|Z_j|, dim]
            if m > Z_j.shape[0]:
                raise ValueError(
                    f"[APCM | Cosine] args.rerank(m)={m} must be <= |Z_j|={Z_j.shape[0]} "
                    f"(setting={setting})."
                )

            # phi(.) ONLY for Z_j
            Z_h = self.phi(Z_j)

            # ---- select H_j (m nearest to p'_j) ----
            p_j_prime = p_prime[j].unsqueeze(0)                      # [1, dim]
            d_p_to_Z = self._cos_dist(p_j_prime, Z_h).squeeze(0)      # [|Z_j|]

            d_sorted, idx_sorted = torch.sort(d_p_to_Z)
            idx_H = idx_sorted[:m]
            idx_barH = idx_sorted[m:]

            H_j = Z_h[idx_H]                                         # [m, dim]
            barH_j = Z_h[idx_barH]                                   # [|Z|-m, dim]

            # ---- raw distances ----
            dis1 = d_sorted[:m]                                      # [m]
            dis2 = d_sorted[m:].mean() if barH_j.shape[0] > 0 else torch.zeros((), device=device, dtype=dtype)

            other_protos = torch.stack([p_prime[k] for k in range(way) if k != j], dim=0)  # [way-1, dim]

            dis_other_H = self._cos_dist(other_protos, H_j)          # [way-1, m]
            dis3 = dis_other_H.mean(dim=0)                           # [m]

            if barH_j.shape[0] > 0:
                dis_other_barH = self._cos_dist(other_protos, barH_j)  # [way-1, |barH|]
                dis4 = dis_other_barH.mean()
            else:
                dis4 = torch.zeros((), device=device, dtype=dtype)

            # ---- controller ----
            dis_vec = torch.cat([dis1, dis3, dis2.unsqueeze(0), dis4.unsqueeze(0)], dim=0)  # [2m+2]
            w_j, lambda_j = self.rerank_controller(dis_vec)
            w_j = w_j.squeeze(0)                                     # [m]
            lambda_j = lambda_j.squeeze(0)                           # scalar

            # ---- union + mean ----
            X_j = data_shot_category[j]                               # [shot, dim]
            a_j = a_list[j]                                           # [shot]

            T_in = X_j * (lambda_j * a_j).unsqueeze(1)                # [shot, dim]
            T_out = H_j * ((1.0 - lambda_j) * w_j).unsqueeze(1)        # [m, dim]

            T_j = torch.cat([T_in, T_out], dim=0)                      # [shot+m, dim]
            p_final[j] = T_j.mean(dim=0)

        # =====================================================================
        # Classification: cosine-distance to final prototypes
        # =====================================================================
        dist_to_classes = torch.zeros(way, num_q, device=device, dtype=dtype)
        for j in range(way):
            dist_to_classes[j] = self._cos_dist(p_final[j].unsqueeze(0), data_query).squeeze(0)

        # temperature = float(getattr(self.args, "temperature", 1.0))
        logits = -dist_to_classes.t()
        return logits, p_final

    def forward(self, shot: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        _, data_shot = self.encoder(shot)
        _, data_query = self.encoder(query)
        # if self.training:
        #     return self._forward_from_feats(data_shot, data_query)
        return self._forward_from_feats(data_shot, data_query), data_shot, data_query
        # return self._forward_from_feats(data_shot, data_query)

    def fusion(self, shot: torch.Tensor, query: torch.Tensor, semantic_prompt: torch.Tensor) -> torch.Tensor:
        _, data_shot = self.encoder.fusion(shot, semantic_prompt, self.args)
        _, data_query = self.encoder(query)
        # if self.training:
        #     return self._forward_from_feats(data_shot, data_query)
        return self._forward_from_feats(data_shot, data_query), data_shot, data_query
        # return self._forward_from_feats(data_shot, data_query)


def visformer_tiny(**kwargs):
    model = Visformer(img_size=224, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True, 
                      embedding_norm=BatchNorm, **kwargs)
    return model


def visformer_tiny_84(**kwargs):
    model = Visformer(img_size=84, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, small_stem=True, **kwargs)
    return model

def visformer_tiny_80(**kwargs):
    model = Visformer(img_size=80, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, small_stem=True, **kwargs)
    return model


def visformer_small_80(**kwargs):
    model = Visformer(img_size=80, init_channels=64, embed_dim=256, depth=[4,2,3], num_heads=6, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm,small_stem=True, **kwargs)
    return model


def visformer_small(**kwargs):
    model = Visformer(img_size=224, init_channels=32, embed_dim=384, depth=[7,4,4], num_heads=6, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model


def visformer_small_84(**kwargs):
    model = Visformer(img_size=84, init_channels=64, embed_dim=256, depth=[4, 2, 3], num_heads=6, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, small_stem=False, **kwargs)
    return model




def visformer_tiny_84_ori(**kwargs):
    model = Visformer(img_size=84, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, small_stem=False, **kwargs)
    return model





if __name__ == '__main__':

    torch.manual_seed(0)
    inputs = torch.rand(2, 3, 84, 84)

    net = visformer_tiny_84()
    print(net)

    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of parameters:{}'.format(parameters))
    x = net(inputs)
    print(x[1].shape)
