# https://github.com/danczs/Visformer/blob/main/models.py

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .weight_init import to_2tuple, trunc_normal_
import torch.nn.functional as F
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, scalar_mul_matrix
from onmt.nn import ToLorentz
from onmt.lmath import lorentz_mean_centroid, lorentz_mean_tangent, dist

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
                image_features = x.view(B, -1, C).contiguous()    # [B, H*W, C]
                text_features = prompt2.unsqueeze(1).contiguous()    # [B, 1, C]
                out = self.dab_block(image_features, text_features)
                out = out.reshape(B, C, H, W).contiguous()
                x = x + self.gate * out
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


class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args.backbone

        if model_name == "visformer":
            self.encoder = visformer_tiny(num_classes=args.num_classes)
        else:
            raise ValueError("Model not found")

        if args.hyperbolic:
            self.e2p = ToPoincare(c=args.c, train_c=args.train_c, train_x=args.train_x)

    def forward(self, data_shot, data_query, semantic_prompt):
        _, proto = self.encoder(data_shot)
        if self.args.hyperbolic:
            proto = self.e2p(proto)

            proto = proto.reshape(self.args.shot, self.args.way, -1)

            proto = poincare_mean(proto, dim=0, c=self.e2p.c)
            _, data_query = self.encoder(data_query)
            data_query = self.e2p(data_query)
            logits = (
                -dist_matrix(data_query, proto, c=self.e2p.c)
            )

        else:
            if self.training:
                proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
            else:
                proto = proto.reshape(
                    self.args.shot, self.args.validation_way, -1
                ).mean(dim=0)

            logits = (
                euclidean_metric(self.encoder(data_query), proto)
            )
            # logits = (
            #     cosine_metric(self.encoder(data_query), proto)
            #     / self.args.temperature
            # )
        return logits


class Curvature_generation_Visformer_Cosine(nn.Module):
    """
    Cosine-distance version (distance = 1 - cosine_similarity).
    Keeps the same controllers / weighting / rerank / old-new mixing logic.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        model_name = args.backbone

        if model_name == "visformer":
            self.encoder = visformer_tiny(num_classes=args.num_classes)
        else:
            raise ValueError("Model not found")

        self.rerank_controller = rerank_Controller(self.args.rerank * 2 + 2, self.args.rerank, self.args.rerank + 1)
        self.support_controller = support_Controller(self.args.shot * self.args.shot, self.args.shot, self.args.shot)

        self.proj_k = nn.Linear(args.dim, args.dim)
        self.proj_q = nn.Linear(args.dim, args.dim)
        self.proj_v = nn.Linear(args.dim, args.dim)

        nn.init.normal_(self.proj_k.weight, mean=0, std=np.sqrt(2.0 / (args.dim + args.dim)))
        nn.init.normal_(self.proj_q.weight, mean=0, std=np.sqrt(2.0 / (args.dim + args.dim)))
        nn.init.normal_(self.proj_v.weight, mean=0, std=np.sqrt(2.0 / (args.dim + args.dim)))

        self.layer_norm = nn.LayerNorm(args.dim)
        self.layer_norm2 = nn.LayerNorm(args.dim)

        self.fc_new = nn.Linear(args.dim, args.dim)
        nn.init.xavier_normal_(self.fc_new.weight)

        self.softmax = nn.Softmax()

    @staticmethod
    def _cos_dist_matrix(x: torch.Tensor) -> torch.Tensor:
        """
        Cosine distance matrix for x: [N, D] -> [N, N]
        dist = 1 - cos(x_i, x_j)
        """
        x_n = F.normalize(x, dim=-1)
        sim = x_n @ x_n.t()                 # [N, N]
        return 1.0 - sim.clamp(-1.0, 1.0)

    @staticmethod
    def _cos_dist_to_set(proto: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        Cosine distance from proto to feats.
        proto: [1, D] or [D]
        feats: [N, D]
        return: [N]
        """
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)
        proto_n = F.normalize(proto, dim=-1)     # [1, D]
        feats_n = F.normalize(feats, dim=-1)     # [N, D]
        sim = (feats_n @ proto_n.t()).squeeze(-1)  # [N]
        return 1.0 - sim.clamp(-1.0, 1.0)

    def forward(self, shot, query):
        device = shot.device

        _, data_shot = self.encoder(shot)      # [way*shot, dim]
        _, data_query = self.encoder(query)    # [way*query, dim]

        if self.args.setting == "inductive":
            rerank_data = data_shot
        else:
            rerank_data = torch.cat([data_query, data_shot], dim=0)

        rerank_data = rerank_data.repeat(self.args.multihead, 1)
        rerank_num = rerank_data.shape[0]

        data_shot_category = data_shot.reshape(self.args.way, self.args.shot, -1)
        mean_proto_category = torch.mean(data_shot_category, 1)  # [way, dim]

        dis_mat = torch.zeros(self.args.way, rerank_num, device=device)
        proto_p = torch.zeros(mean_proto_category.shape[0], self.args.dim, device=device)

        if (self.args.backbone == "resnet12" or self.args.backbone == "bigres12") and self.args.setting == "transductive":
            data_shot_prooject_k = mean_proto_category
            data_shot_prooject_q = rerank_data
            data_shot_prooject_v = rerank_data
        if self.args.backbone == "convnet" and self.args.setting == "transductive" and self.args.shot == 1:
            data_shot_prooject_k = mean_proto_category
            data_shot_prooject_q = rerank_data
            data_shot_prooject_v = rerank_data
        else:
            data_shot_prooject_k = self.proj_k(mean_proto_category)
            data_shot_prooject_q = self.proj_q(rerank_data)
            data_shot_prooject_q1 = F.relu(data_shot_prooject_q)
            data_shot_prooject_v = self.proj_v(data_shot_prooject_q1)

        # ------------------------ Support weighting + proto init ------------------------
        for i in range(self.args.way):
            if self.args.shot == 1:
                proto_i = data_shot_prooject_k[i].unsqueeze(0)  # [1, dim]
                proto_p[i] = proto_i.squeeze(0)
            else:
                data_shot_i = data_shot_category[i, :, :]  # [shot, dim]

                # (1) Replace hyperbolic support distance matrix with cosine distance matrix
                support_dis_mat = self._cos_dist_matrix(data_shot_i)  # [shot, shot]

                support_dis_mat = support_dis_mat.view(1, -1) / np.power(self.args.dim, 0.5)
                support_dis_mat = self.softmax(-1 * support_dis_mat)

                support_weight = self.support_controller(support_dis_mat)            # [1, shot]
                support_weight_i = support_weight * support_weight.shape[1]          # [1, shot]
                weight_data_support_i = data_shot_i * support_weight_i.squeeze(0).unsqueeze(1)

                # (2) Replace hyperbolic mean with Euclidean weighted mean (space closure)
                proto_i = weight_data_support_i.mean(dim=0, keepdim=True)  # [1, dim]
                proto_p[i] = proto_i.squeeze(0)

            # (3) Replace proto-to-rerank hyperbolic distance with cosine distance
            dis_mat[i] = self._cos_dist_to_set(proto_i, data_shot_prooject_q)

        # ------------------------ Rerank / context-aware rectification ------------------------
        _, indices = torch.sort(dis_mat)
        test_proto = torch.zeros_like(proto_p)

        for i in range(self.args.way):
            n_i_d = dis_mat[i, indices[i, 0:self.args.rerank]]
            o_i_d = dis_mat[i, indices[i, self.args.rerank:dis_mat.shape[1]]].mean()

            n_o_d = torch.cat(
                [dis_mat[0:i, indices[i, 0:self.args.rerank]],
                 dis_mat[i+1:self.args.way, indices[i, 0:self.args.rerank]]],
                dim=0
            ).mean(dim=0)

            o_o_d = torch.cat(
                [dis_mat[0:i, indices[i, self.args.rerank:dis_mat.shape[1]]],
                 dis_mat[i+1:self.args.way, indices[i, self.args.rerank:dis_mat.shape[1]]]],
                dim=0
            ).mean()

            n_i_d = self.softmax(-1 * (n_i_d / np.power(self.args.dim, 0.5)))
            n_o_d = self.softmax(-1 * (n_o_d / np.power(self.args.dim, 0.5)))

            i_weight, old_new_weight = self.rerank_controller(
                torch.cat([n_i_d, n_o_d, o_i_d.unsqueeze(0), o_o_d.unsqueeze(0)], dim=0)
            )

            i_weight = (i_weight * old_new_weight[0]) * (i_weight.shape[1] + 1)

            if (self.args.backbone == "resnet12" or self.args.backbone == "bigres12") and self.args.setting == "transductive":
                weight_data_query_i = rerank_data[indices[i, 0:self.args.rerank], :] * i_weight.squeeze(0).unsqueeze(1)
            if self.args.backbone == "convnet" and self.args.setting == "transductive" and self.args.shot == 1:
                weight_data_query_i = rerank_data[indices[i, 0:self.args.rerank], :] * i_weight.squeeze(0).unsqueeze(1)
            else:
                weight_data_query_i = data_shot_prooject_v[indices[i, 0:self.args.rerank], :] * i_weight.squeeze(0).unsqueeze(1)
                weight_data_query_i = self.fc_new(weight_data_query_i)
                weight_data_query_i = self.layer_norm(weight_data_query_i)

            mean_i = (mean_proto_category[i] * (1 - old_new_weight) * (i_weight.shape[1] + 1)).unsqueeze(0)

            # (4) Replace hyperbolic centroid with Euclidean mean
            test_proto[i] = torch.cat([weight_data_query_i, mean_i], dim=0).mean(dim=0)

        # ------------------------ Query classification distances ------------------------
        new_dis_mat = torch.zeros(self.args.way, data_query.shape[0], device=device)

        for i in range(self.args.way):
            proto_i = test_proto[i].unsqueeze(0)
            # (5) Replace hyperbolic proto-query distance with cosine distance
            new_dis_mat[i] = self._cos_dist_to_set(proto_i, data_query)

        logits = -new_dis_mat.t()

        if self.training:
            return logits
        else:
            support_feats = data_shot.reshape(-1, data_shot.size(-1))
            query_feats = data_query.reshape(-1, data_query.size(-1))
            c_dummy = torch.tensor(0.0, device=device)
            return logits, support_feats, query_feats, test_proto, c_dummy

    def fusion(self, shot, query, semantic_prompt):
        device = shot.device

        _, data_shot = self.encoder.fusion(shot, semantic_prompt, self.args)
        _, data_query = self.encoder(query)

        if self.args.setting == "inductive":
            rerank_data = data_shot
        else:
            rerank_data = torch.cat([data_query, data_shot], dim=0)

        rerank_data = rerank_data.repeat(self.args.multihead, 1)
        rerank_num = rerank_data.shape[0]

        data_shot_category = data_shot.reshape(self.args.way, self.args.shot, -1)
        mean_proto_category = torch.mean(data_shot_category, 1)

        dis_mat = torch.zeros(self.args.way, rerank_num, device=device)
        proto_p = torch.zeros(mean_proto_category.shape[0], self.args.dim, device=device)

        if (self.args.backbone == "resnet12" or self.args.backbone == "bigres12") and self.args.setting == "transductive":
            data_shot_prooject_k = mean_proto_category
            data_shot_prooject_q = rerank_data
            data_shot_prooject_v = rerank_data
        if self.args.backbone == "convnet" and self.args.setting == "transductive" and self.args.shot == 1:
            data_shot_prooject_k = mean_proto_category
            data_shot_prooject_q = rerank_data
            data_shot_prooject_v = rerank_data
        else:
            data_shot_prooject_k = self.proj_k(mean_proto_category)
            data_shot_prooject_q = self.proj_q(rerank_data)
            data_shot_prooject_q1 = F.relu(data_shot_prooject_q)
            data_shot_prooject_v = self.proj_v(data_shot_prooject_q1)

        for i in range(self.args.way):
            if self.args.shot == 1:
                proto_i = data_shot_prooject_k[i].unsqueeze(0)
                proto_p[i] = proto_i.squeeze(0)
            else:
                data_shot_i = data_shot_category[i, :, :]
                support_dis_mat = self._cos_dist_matrix(data_shot_i)
                support_dis_mat = support_dis_mat.view(1, -1) / np.power(self.args.dim, 0.5)
                support_dis_mat = self.softmax(-1 * support_dis_mat)

                support_weight = self.support_controller(support_dis_mat)
                support_weight_i = support_weight * support_weight.shape[1]
                weight_data_support_i = data_shot_i * support_weight_i.squeeze(0).unsqueeze(1)

                proto_i = weight_data_support_i.mean(dim=0, keepdim=True)
                proto_p[i] = proto_i.squeeze(0)

            dis_mat[i] = self._cos_dist_to_set(proto_i, data_shot_prooject_q)

        _, indices = torch.sort(dis_mat)
        test_proto = torch.zeros_like(proto_p)

        for i in range(self.args.way):
            n_i_d = dis_mat[i, indices[i, 0:self.args.rerank]]
            o_i_d = dis_mat[i, indices[i, self.args.rerank:dis_mat.shape[1]]].mean()

            n_o_d = torch.cat(
                [dis_mat[0:i, indices[i, 0:self.args.rerank]],
                 dis_mat[i+1:self.args.way, indices[i, 0:self.args.rerank]]],
                dim=0
            ).mean(dim=0)

            o_o_d = torch.cat(
                [dis_mat[0:i, indices[i, self.args.rerank:dis_mat.shape[1]]],
                 dis_mat[i+1:self.args.way, indices[i, self.args.rerank:dis_mat.shape[1]]]],
                dim=0
            ).mean()

            n_i_d = self.softmax(-1 * (n_i_d / np.power(self.args.dim, 0.5)))
            n_o_d = self.softmax(-1 * (n_o_d / np.power(self.args.dim, 0.5)))

            i_weight, old_new_weight = self.rerank_controller(
                torch.cat([n_i_d, n_o_d, o_i_d.unsqueeze(0), o_o_d.unsqueeze(0)], dim=0)
            )
            i_weight = (i_weight * old_new_weight[0]) * (i_weight.shape[1] + 1)

            if (self.args.backbone == "resnet12" or self.args.backbone == "bigres12") and self.args.setting == "transductive":
                weight_data_query_i = rerank_data[indices[i, 0:self.args.rerank], :] * i_weight.squeeze(0).unsqueeze(1)
            if self.args.backbone == "convnet" and self.args.setting == "transductive" and self.args.shot == 1:
                weight_data_query_i = rerank_data[indices[i, 0:self.args.rerank], :] * i_weight.squeeze(0).unsqueeze(1)
            else:
                weight_data_query_i = data_shot_prooject_v[indices[i, 0:self.args.rerank], :] * i_weight.squeeze(0).unsqueeze(1)
                weight_data_query_i = self.fc_new(weight_data_query_i)
                weight_data_query_i = self.layer_norm(weight_data_query_i)

            mean_i = (mean_proto_category[i] * (1 - old_new_weight) * (i_weight.shape[1] + 1)).unsqueeze(0)
            test_proto[i] = torch.cat([weight_data_query_i, mean_i], dim=0).mean(dim=0)

        new_dis_mat = torch.zeros(self.args.way, data_query.shape[0], device=device)
        for i in range(self.args.way):
            new_dis_mat[i] = self._cos_dist_to_set(test_proto[i].unsqueeze(0), data_query)

        logits = -new_dis_mat.t()

        if self.training:
            return logits
        else:
            support_feats = data_shot.reshape(-1, data_shot.size(-1))
            query_feats = data_query.reshape(-1, data_query.size(-1))
            c_dummy = torch.tensor(0.0, device=device)
            return logits, support_feats, query_feats, test_proto, c_dummy
            


# class Curvature_generation_Visformer(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         model_name = args.backbone

#         if model_name == 'visformer':
#             self.encoder = visformer_tiny(num_classes=args.num_classes)
#         else:
#             raise ValueError("Model not found")

#         self.e2l = ToLorentz(c=0.01, clip_r=args.clip_r)


#     def forward(self, shot, query, semantic_prompt, ori_shot=None, fusion=False):
#         _, data_shot = self.encoder.fusion(shot, semantic_prompt, self.args)
#         _, data_query = self.encoder(query)
        
#         # data_shot = data_shot.view(self.args.way, self.args.shot, -1).mean(dim=1)
#         # logits = F.normalize(data_query, dim=-1) @ F.normalize(data_shot, dim=-1).t()
#         proto = self.e2l(data_shot)

#         proto = proto.reshape(self.args.way, self.args.shot, -1)

#         proto = lorentz_mean_centroid(proto, dim=1, c=self.e2l.c)

#         data_query = self.e2l(data_query)

#         query_expanded = data_query.unsqueeze(1)    # [B, 1, dim+1]
#         proto_expanded = proto.unsqueeze(0)    # [1, way, dim+1]

#         if isinstance(self.e2l.c, torch.Tensor):
#             k_val = 1.0 / self.e2l.c
#         else:
#             k_val = torch.tensor(1.0 / self.e2l.c, device=data_query.device)

#         if not fusion:
#             logits = (
#                 -dist(query_expanded, proto_expanded, k=k_val)
#             )
#             return logits
#         else:
#             if ori_shot is not None:
#                 _, ori_data_shot = self.encoder(ori_shot)
#                 ori_proto = self.e2l(ori_data_shot)
#                 ori_proto = ori_proto.reshape(self.args.way, self.args.shot, -1)
#                 ori_proto = lorentz_mean_centroid(ori_proto, dim=1, c=self.e2l.c)
#                 ori_proto_expanded = ori_proto.unsqueeze(0)    # [1, way, dim+1]
#                 return proto_expanded, ori_proto_expanded, self.e2l.c, query_expanded
#             else:
#                 raise ValueError("ori_shot must be provided when fusion is True")


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
