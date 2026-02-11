import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', type=str, default='acg-han-hl-evo-lgf-')
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--dataset', type=str, default='AID')
parser.add_argument('--image_size', type=int,default=224)
parser.add_argument('--episode', type=int, default=2000)
parser.add_argument('--feat-size', type=int, default=384)
parser.add_argument('--semantic-size', type=int, default=512)
parser.add_argument('--backbone', type=str, default='visformer')
parser.add_argument('--aug_support', type=int, default=1)
parser.add_argument('--stage', type=float, default=3)
parser.add_argument('--num_workers', type=int, default=8, choices=[16,8,4,2,1])
parser.add_argument('--gpu', default='5')
parser.add_argument('--c', type=float, default=0.01, help='Curvature')
parser.add_argument('--clip_r', type=float, default=None)
parser.add_argument('--rerank', type=int, default=5, help='Rerank')
parser.add_argument('--setting', type=str, default='inductive', choices=['inductive', 'transductive'], help='Exp Setting')
parser.add_argument('--multihead', type=int, default=20, help='multi_head')
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
from model.visformer_han_hl_evo_lgf import Curvature_generation_Visformer, ProtoNet
from model.model_components import HLFormerBlock
from onmt.nn import lorentz_linear_fusion
from onmt.lmath import dist
from model.lorentz_gaussian_fusion_han import LorentzGaussianFusion
import utils

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

    args.root = '/mnt/sdb/pzy/Awesome-Projects/VTEvo-FSL'
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
        model = Curvature_generation_Visformer(args=args)
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
        model.encoder.hlformer_block = HLFormerBlock(args)
        model.encoder.gate = torch.nn.Parameter(torch.tensor(0.))
    init = args.work_dir+'/'+'best.pth'
    # init = f'{args.root}/checkpoint/{args.dataset}-{args.shot}-shot.pth'
    checkpoint = torch.load(init,map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    utils.log('num params: {}'.format(utils.compute_n_params(model)),'test')
    model = model.to(device)

    lgf = LorentzGaussianFusion(curvature=args.c)

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
                logits, gen_support_feats, query_feats, gen_support_protos, c = model.fusion(sup, que, text_features)
                acc = utils.compute_acc(logits, labels)
                va_lst.append(acc)

                support_targets = torch.arange(args.way).unsqueeze(1).repeat(1, args.shot).view(-1).to(device)

                _, support_feats, _, support_protos, _ = model(sup1, que)
                query_expanded = query_feats.unsqueeze(1)    # [way*query, 1, dim+1]

                mu1, var1 = lgf.estimate_parameters(support_feats, query_feats, support_protos, support_targets, args.way)
                mu2, var2 = lgf.estimate_parameters(gen_support_feats, query_feats, gen_support_protos, support_targets, args.way)

                # tangent space gaussian fusion
                # mu_fused = lgf.tangent_space_fusion(mu1, var1, mu2, var2)
                # proto_expanded = mu_fused.unsqueeze(0)    # [1, way, dim+1]

                # geodesic gaussian fusion
                mu_fused = lgf.geodesic_fusion(mu1, var1, mu2, var2)
                proto_expanded = mu_fused.unsqueeze(0)    # [1, way, dim+1]

                if isinstance(c, torch.Tensor):
                    k_val = 1.0 / c
                else:
                    k_val = torch.tensor(1.0 / c, device=query_expanded.device)

                logits = -dist(query_expanded, proto_expanded, k=k_val)
                acc = utils.compute_acc(logits, labels)
                A_acc.append(acc)

                # proto_expanded, ori_proto_expanded, c, query_expanded = model(sup, que, text_features, ori_shot=sup1, fusion=True)

                # if isinstance(c, torch.Tensor):
                #     k_val = 1.0 / c
                # else:
                #     k_val = torch.tensor(1.0 / c, device=query_expanded.device)

                # # com_proto = f * support + (1 - f) * gen_support
                # # logits = F.normalize(query, dim=-1) @ F.normalize(com_proto, dim=-1).t()
                # com_proto = lorentz_linear_fusion(ori_proto_expanded, proto_expanded, f, c)
                # logits = -dist(query_expanded, com_proto, k=k_val)
                # acc = utils.compute_acc(logits, labels)
                # A_acc.append(acc)

    
    va_lst = utils.count_95acc(np.array(va_lst))
    A_acc = utils.count_95acc(np.array(A_acc))
    log_str = 'test epoch : acc = {:.2f} +- {:.2f} (%)'.format(va_lst[0] * 100, va_lst[1] * 100)
    log_str += ' | {:.2f} +- {:.2f}'.format(A_acc[0] * 100, A_acc[1] * 100)
    utils.log(log_str,'test')

if __name__ == '__main__':

    main(args)

