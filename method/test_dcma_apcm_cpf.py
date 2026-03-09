import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', type=str, default='vgsr-dcma-apcm-cpf-ucm-')
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--dataset', type=str, default='UCM')
parser.add_argument('--image_size', type=int,default=224)
parser.add_argument('--episode', type=int, default=2000)
parser.add_argument('--feat-size', type=int, default=384)
parser.add_argument('--semantic-size', type=int, default=512)
parser.add_argument('--backbone', type=str, default='visformer')
parser.add_argument('--aug_support', type=int, default=1)
parser.add_argument('--stage', type=float, default=3)
parser.add_argument('--num_workers', type=int, default=8, choices=[16,8,4,2,1])
parser.add_argument('--gpu', default='5')
# parser.add_argument('--c', type=float, default=0.01, help='Curvature')
# parser.add_argument('--clip_r', type=float, default=None)
parser.add_argument('--rerank', type=int, default=4, help='Rerank')
parser.add_argument('--setting', type=str, default='inductive', choices=['inductive', 'transductive'], help='Exp Setting')
# parser.add_argument('--multihead', type=int, default=20, help='multi_head')
parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='dropout probability')
parser.add_argument('--sft_factor', type=float, default=1.0, help='softmax temperature')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset,DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from data.dataloader import EpisodeSampler
from model.visformer_dcma_apcm_cpf import Visformer_DCMA_APCM_CPF
from model.dual_cross_modal_attention import DAB
import utils

def transductive_rectification_cosine(
    support_feats, 
    query_feats, 
    init_proto, 
    temperature=10.0
):
    """
    通用直推式原型校准函数 (Cosine Version) - 适配 Flatten 输入
    
    Args:
        support_feats (Tensor): [Way * Shot, Dim] 展平的支持集特征。
        query_feats (Tensor): [Query_Num, Dim] 查询集特征。
        init_proto (Tensor): [Way, Dim] 初始原型 (EAN/CAN 输出)。
        temperature (float): Softmax 温度系数。
    
    Returns:
        refined_proto (Tensor): [Way, Dim] 校准后的原型。
        uncertainty (Tensor): [Way] 分布不确定性。
    """
    # 1. 维度恢复与预处理
    n_way = args.way
    n_shot = args.shot
    dim = support_feats.shape[-1]
    device = support_feats.device
    
    # [Way * Shot, Dim] -> [Way, Shot, Dim]
    # 确保 contiguous() 以避免 view 操作报错
    support_feats = support_feats.view(n_way, n_shot, dim)
    
    # 归一化 (Cosine Metric 前提)
    support_feats = F.normalize(support_feats, dim=-1)
    query_feats = F.normalize(query_feats, dim=-1)
    init_proto = F.normalize(init_proto, dim=-1)
    
    # 2. 计算 Query 的软权重 (Soft Assignment)
    # 利用 init_proto 计算 Query 属于各类别的概率
    # query: [Q, D], proto: [N, D] -> sim: [Q, N]
    sims = torch.mm(query_feats, init_proto.t())
    logits = sims * temperature
    probs = F.softmax(logits, dim=1) # [Q, N]
    
    refined_protos_list = []
    uncertainty_list = []
    
    for k in range(n_way):
        # --- A. 准备权重 ---
        # Support 样本权重设为 1.0
        w_support = torch.ones(n_shot).to(device)
        # Query 样本权重 (取第 k 列)
        w_query = probs[:, k] # [Q]
        
        # 拼接特征: Support [Shot, D] + Query [Q, D]
        all_feats = torch.cat([support_feats[k], query_feats], dim=0) 
        # 拼接权重
        all_weights = torch.cat([w_support, w_query], dim=0) # [Shot + Q]
        sum_w = all_weights.sum() + 1e-8
        
        # --- B. 计算修正后的原型 (Refined Prototype) ---
        # Weighted Mean
        # all_feats: [S+Q, D], all_weights: [S+Q] -> broadcast -> sum -> [D]
        mu_k = (all_feats * all_weights.unsqueeze(1)).sum(0) / sum_w
        mu_k = F.normalize(mu_k, dim=0) # 投影回球面
        
        refined_protos_list.append(mu_k)
        
        # --- C. 计算不确定性 (Uncertainty / Dispersion) ---
        # 计算簇内样本到新原型的余弦距离 (1 - CosSim)
        sim_to_refined = torch.matmul(all_feats, mu_k)
        dist_to_refined = 1.0 - sim_to_refined
        
        # 加权平均距离 -> sigma_k
        sigma_k = (dist_to_refined * all_weights).sum() / sum_w
        uncertainty_list.append(sigma_k)
        
    refined_proto = torch.stack(refined_protos_list) # [Way, Dim]
    uncertainty = torch.stack(uncertainty_list)      # [Way]
    
    return refined_proto, uncertainty

def adaptive_dual_fusion_cosine(
    proto_vis, sigma_vis, 
    proto_sem, sigma_sem, 
):
    """
    自适应融合两个修正后的原型
    
    Args:
        proto_vis: [N_way, Dim] 修正后的均值原型
        sigma_vis: [N_way] 均值原型的离散度
        proto_sem: [N_way, Dim] 修正后的语义增强原型
        sigma_sem: [N_way] 语义增强原型的离散度
    """
    # N_way = proto_vis.shape[0]
    
    # 计算相对置信度权重
    # 我们希望: sigma 越小 -> weight 越大
    # 方法: 使用 Softmax( -sigma * scale ) 或者简单的比率
    
    # 这里使用一种简单直观的归一化权重:
    # weight_vis = sigma_sem / (sigma_vis + sigma_sem)
    # 如果 sigma_vis 很小 (非常确定)，则 weight_vis 接近 1
    
    # 为了增加数值稳定性，加一个 epsilon
    sum_sigma = sigma_vis + sigma_sem + 1e-8
    
    # 动态权重 (Dynamic Weight)
    lambda_vis = sigma_sem / sum_sigma  # 语义越不确定，视觉权重越大
    lambda_sem = sigma_vis / sum_sigma  # 视觉越不确定，语义权重越大
    
    # 扩展维度 [N_way, 1]
    lambda_vis = lambda_vis.unsqueeze(1)
    lambda_sem = lambda_sem.unsqueeze(1)
    
    # 融合
    # 可以引入 beta 作为先验偏好 (例如稍微偏向视觉)
    # final = beta * (lambda_vis * vis) + (1-beta) * (lambda_sem * sem)
    # 或者直接使用计算出的动态权重:
    
    final_proto = lambda_vis * proto_vis + lambda_sem * proto_sem
    
    # 再次归一化
    final_proto = F.normalize(final_proto, dim=-1)
    
    return final_proto

def get_grouped_few_shot_images(val_dataset, target_classes, shot):
    all_labels = torch.tensor(val_dataset.targets)
    class_to_indices = defaultdict(list)

    for idx, label in enumerate(all_labels):
        if label.item() in target_classes.tolist():
            class_to_indices[label.item()].append(idx)

    for c in target_classes:
        assert len(class_to_indices[c.item()]) >= shot, f"class {c.item()} lack {shot} samples"

    sampled_per_class = {}
    for c in target_classes:
        inds = np.random.choice(class_to_indices[c.item()], size=shot, replace=False)
        sampled_per_class[c.item()] = list(inds)

    grouped_indices = []
    grouped_labels = []
    for c in target_classes:
        grouped_indices.extend(sampled_per_class[c.item()])
        grouped_labels.extend([c.item()] * shot)

    final_dataset = Subset(val_dataset, grouped_indices)
    final_loader = DataLoader(final_dataset, batch_size=len(grouped_indices), shuffle=False)
    images, _ = next(iter(final_loader))
    labels = torch.tensor(grouped_labels)

    return images, labels


def main(config):
    # seed
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    args.root = '/mnt/sdb/pzy/Awesome-Projects/VT-FSL-Ours'
    svname = args.name+'{}-shot'.format(args.shot) 
    args.work_dir = os.path.join(f'{args.root}/save', args.dataset, svname)
    utils.set_log_path(args.work_dir)

    # Dataset
    if args.dataset in ['miniImageNet', 'tieredImageNet']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CIFAR-FS', 'FC100']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224_cifar)
    elif args.dataset in ['FG-CUB', 'FG-Cars','FG-Dogs']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CD-Cars','CropDiseases', 'EuroSAT', 'ISIC', 'ChestX','Places', 'Plantae']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/novel'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['NWPU']:
        args.train = '/mnt/sdb/pzy/rssc_dataset/NWPU-RESISC45/train'
        args.val = '/mnt/sdb/pzy/rssc_dataset/NWPU-RESISC45/test'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['AID', 'UCM']:
        if args.dataset == 'AID':
            args.train = f'/mnt/sdb/pzy/rssc_dataset/AID_split1/train'
            args.val = f'/mnt/sdb/pzy/rssc_dataset/AID_split1/test'
        else:
            args.train = f'/mnt/sdb/pzy/rssc_dataset/{args.dataset}/train'
            args.val = f'/mnt/sdb/pzy/rssc_dataset/{args.dataset}/test'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    # load cache
    # generated_cache = torch.load('{}/my_data/testing/test_{}_{}shot.pt'.format(args.root, args.dataset,str(args.shot)))  # class_id -> [shot, 3, 224, 224]

    if args.aug_support == 1:
        utils.log('fs dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset),len(val_dataset.classes)),'test')
    else:
        utils.log('fs dataset: {} (x{}), {}'.format(val_dataset[0][0][0].shape, len(val_dataset),len(val_dataset.classes)),'test')

    # Dataloader
    val_sampler = EpisodeSampler(val_dataset.targets, args.episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,num_workers=args.num_workers, pin_memory=True)

    args.num_classes = len(train_dataset.classes)

    val_idx_to_class = val_dataset.class_to_idx
    idx_to_class = train_dataset.class_to_idx
    if args.dataset == 'FG-CUB':
        val_idx_to_class = {k: v.split(".", 1)[-1] for v,k in val_idx_to_class.items()}
        idx_to_class = {k: v.split(".", 1)[-1] for v,k in idx_to_class.items()}
    elif args.dataset == 'FG-Dogs':
        val_idx_to_class = {k: v.split("-", 1)[-1] for v,k in val_idx_to_class.items()}
        idx_to_class = {k: v.split("-", 1)[-1] for v,k in idx_to_class.items()}
    else:
        val_idx_to_class = {k: v for v, k in val_idx_to_class.items()}
        idx_to_class = {k: v for v, k in idx_to_class.items()}

    semantic = torch.load('{}/my_data/semantic/{}_semantic_clip_cot_with_color_acg.pth'.format(args.root, args.dataset))['semantic_feature']
    # semantic = torch.load('{}/nwpu_attr.pth'.format(args.root))['semantic_feature']
    # semantic = torch.load('{}/evo_nwpu_attr.pth'.format(args.root))['semantic_feature']
    semantic = {k: v.float() for k, v in semantic.items()}

    # backbone
    if args.backbone == 'visformer':
        args.dim = args.feat_size
        model = Visformer_DCMA_APCM_CPF(args=args)
    else:
        raise ValueError(f'unknown model: {args.model}')
    
    # load
    if args.backbone == 'visformer':
        text_dim = args.semantic_size
        feature_dim = args.feat_size
        args.patch_len = 49
        if 2 <= args.stage < 3:
            feature_dim = 192
        model.encoder.t2i = torch.nn.Linear(text_dim, feature_dim, bias=False)
        model.encoder.t2i2 = torch.nn.Linear(text_dim, feature_dim, bias=False)
        # model.encoder.se_block = torch.nn.Sequential(torch.nn.Linear(feature_dim*2, feature_dim, bias=True),
        #                                                 torch.nn.Sigmoid(),
        #                                                 torch.nn.Linear(feature_dim, feature_dim),
        #                                                 torch.nn.Sigmoid(),)
        model.encoder.dab_block = DAB(args)
        model.encoder.gate = torch.nn.Parameter(torch.tensor(0.))
    init = args.work_dir+'/'+'best.pth'
    # init = f'{args.root}/checkpoint/{args.dataset}-{args.shot}-shot.pth'
    checkpoint = torch.load(init,map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    utils.log('num params: {}'.format(utils.compute_n_params(model)),'test')
    model = model.to(device)

    # test 
    model.eval()
    va_lst = []
    A_acc = []
    # f = float(checkpoint['k'])
    
    with torch.no_grad():
        for episode in tqdm(val_loader,desc='fs-' + str(args.shot), leave=False):
            if args.aug_support ==1:
                image = episode[0].to(device)
                glabels = episode[1].to(device)
                labels = torch.arange(args.way).unsqueeze(-1).repeat(1, args.query).view(-1).to(device)

                # generated_labels = glabels.view(args.way, args.shot+15)[:, :1]
                # generated_labels = generated_labels.contiguous().view(-1)
                # support_list = [generated_cache[int(class_id)] for class_id in generated_labels]
                # generated_support = torch.cat(support_list, dim=0).to(device)

                image = image.view(args.way, args.shot+args.query, *image.shape[1:])
                sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
                sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

                glabels = glabels.view(args.way, args.shot+15)[:, :args.shot]
                glabels = glabels.contiguous().view(-1)
                text_features = torch.stack([semantic[val_idx_to_class[l.item()]] for l in glabels]).to(device)

                sup_reshaped = sup.view(args.way, args.shot, 3, 224, 224)
                # gen_reshaped = generated_support.view(args.way, args.shot, 3, 224, 224)
                # merged = torch.cat([sup_reshaped, gen_reshaped], dim=1)
                # sup1 = merged.view(args.way * 2 * args.shot, 3, 224, 224)
                sup1 = sup_reshaped.view(args.way * args.shot, 3, 224, 224)
                # _, support = model(sup1)
                # support = support.view(args.way, args.shot, -1).mean(dim=1)

                # _, gen_support = model.fusion(sup, text_features, args)
                # gen_support = gen_support.view(args.way, args.shot, -1).mean(dim=1)
                # _, query = model(que)

                # logits = F.normalize(query, dim=-1) @ F.normalize(gen_support, dim=-1).t()
                # support_targets = torch.arange(args.way).unsqueeze(1).repeat(1, args.shot).view(-1).to(device)

                (ori_logits, support_proto), support_feats, query_feats = model(sup1, que)
                (gen_logits, gen_support_proto), gen_support_feats, query_feats = model.fusion(sup, que, text_features)
                acc = utils.compute_acc(gen_logits, labels)
                va_lst.append(acc)

                mu1, var1 = transductive_rectification_cosine(support_feats, query_feats, support_proto)
                mu2, var2 = transductive_rectification_cosine(gen_support_feats, query_feats, gen_support_proto)

                fused_proto = adaptive_dual_fusion_cosine(mu1, var1, mu2, var2)

                logits = F.normalize(query_feats, dim=-1) @ F.normalize(fused_proto, dim=-1).t()
                acc = utils.compute_acc(logits, labels)
                A_acc.append(acc)

                # mu_v, var_v = estimate_gaussian_parameters(support, query, support_targets, args.way)
                # mu_t, var_t = estimate_gaussian_parameters(gen_support, query, support_targets, args.way)

                # fused_proto = gaussian_fusion(mu_v, var_v, mu_t, var_t)

                # logits = F.normalize(query, dim=-1) @ F.normalize(fused_proto, dim=-1).t()
                # acc = utils.compute_acc(logits, labels)
                # A_acc.append(acc)

                # proto_expanded, ori_proto_expanded, c, query_expanded = model(sup, que, text_features, ori_shot=sup1, fusion=True)

                # if isinstance(c, torch.Tensor):
                #     k_val = 1.0 / c
                # else:
                #     k_val = torch.tensor(1.0 / c, device=query_expanded.device)

                # # com_proto = f * support + (1 - f) * gen_support
                # logits = f * ori_logits + (1 - f) * gen_logits
                # # com_proto = lorentz_linear_fusion(ori_proto_expanded, proto_expanded, f, c)
                # # logits = -dist(query_expanded, com_proto, k=k_val)
                # acc = utils.compute_acc(logits, labels)
                # A_acc.append(acc)

    
    va_lst = utils.count_95acc(np.array(va_lst))
    A_acc = utils.count_95acc(np.array(A_acc))
    log_str = 'test epoch : acc = {:.2f} +- {:.2f} (%)'.format(va_lst[0] * 100, va_lst[1] * 100)
    log_str += ' | {:.2f} +- {:.2f}'.format(A_acc[0] * 100, A_acc[1] * 100)
    utils.log(log_str,'test')

if __name__ == '__main__':

    main(args)

