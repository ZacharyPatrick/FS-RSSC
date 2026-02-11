import argparse
import sys
import os
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--name', type=str, default='acg-han-hl-evo-lgf-test-check-')
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--query', type=int, default=15)
parser.add_argument('--dataset', type=str, default='UCM')
parser.add_argument('--image_size', type=int,default=224)
parser.add_argument('--episode', type=int, default=600)
parser.add_argument('--feat-size', type=int, default=384)
parser.add_argument('--semantic-size', type=int, default=512)
parser.add_argument('--backbone', type=str, default='visformer')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr1', type=float, default=1e-6)
parser.add_argument('--c_lr', type=float, default=1e-3)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--epoch', type=int,default=100)
parser.add_argument('--stage', type=float, default=3)
parser.add_argument('--num_workers', type=int, default=8, choices=[16,8,4,2,1])
parser.add_argument('--t', type=float, default=1.0)
parser.add_argument('--gpu', default='2')
parser.add_argument('--warmup_epochs', type=int, default=None, 
                   help='Warmup epochs (default: epoch//20)')
parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                   help='Min learning rate ratio (default: 0.01)')
parser.add_argument('--scheduler', type=str, default='none', choices=['cosine', 'none'],
                   help='Learning rate scheduler type')
# parser.add_argument('--hyperbolic', action='store_true', default=True, help='Use hyperbolic space')
parser.add_argument('--c', type=float, default=0.01, help='Curvature')
# parser.add_argument('--train_c', action='store_true', default=False, help='Train c')
# parser.add_argument('--train_x', action='store_true', default=False, help='Train x')
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
from torch.utils.data import DataLoader
import numpy as np
from timm.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from data.dataloader import EpisodeSampler, CategoriesSampler
from model.visformer_han_hl_evo_lgf import Curvature_generation_Visformer, ProtoNet
from model.model_components import HLFormerBlock
from onmt.nn import lorentz_linear_fusion
from onmt.lmath import dist
from model.lorentz_gaussian_fusion_han import LorentzGaussianFusion
import utils

def kernel_fn(x, y, type='rbf', sigma=1.0, degree=3, coef0=1.0):
    if type == 'linear':
        return x @ y.T
    elif type == 'poly':
        return (x @ y.T + coef0) ** degree
    elif type == 'rbf':
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (B1, 1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (B2, 1)
        dist = x_norm - 2 * x @ y.T + y_norm.T      # (B1, B2)
        return torch.exp(-dist / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel type: {type}")

def volume_computation_with_kernel(anchor, *inputs, kernel_type='rbf', sigma=1.0, degree=3, coef0=1.0):
    """
    Volume computation in kernel-induced feature space.

    Args:
    - anchor: Tensor of shape (B1, D)
    - *inputs: list of tensors of shape (B2, D)
    - kernel_type: 'linear', 'poly', or 'rbf'
    Returns:
    - volume matrix: (B1, B2)
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]
    n_modalities = 1 + len(inputs)

    # Compute kernel matrices
    aa = kernel_fn(anchor, anchor, type=kernel_type, sigma=sigma, degree=degree, coef0=coef0)
    aa_diag = torch.diagonal(aa, dim1=0, dim2=1).unsqueeze(1).expand(-1, batch_size2)

    l_inputs = [kernel_fn(anchor, input, type=kernel_type, sigma=sigma, degree=degree, coef0=coef0)
                for input in inputs]

    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot = kernel_fn(input1, input2, type=kernel_type, sigma=sigma, degree=degree, coef0=coef0)
            dot_diag = torch.diagonal(dot, dim1=0, dim2=1).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_diag)
        input_dot_products.append(row)

    # Construct Gram matrix for each anchor pair
    G = torch.stack(
        [torch.stack([aa_diag] + l_inputs, dim=-1)] +
        [torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1)
         for i in range(len(inputs))],
        dim=-2
    )  # (B1, B2, n_modalities, n_modalities)

    # Compute determinant for each Gram matrix
    gram_det = torch.det(G.float())
    res = torch.sqrt(torch.clamp(gram_det, min=1e-8))  # numerical stability
    return res

def create_cosine_scheduler(optimizer, total_epochs, warmup_epochs=0, min_lr=0):
    """
    使用PyTorch内置调度器创建cosine学习率调度器
    
    Args:
        optimizer: 优化器
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        min_lr: 最小学习率
    
    Returns:
        scheduler: 学习率调度器
    """
    if warmup_epochs > 0:
        # 带预热的cosine调度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
            ],
            milestones=[warmup_epochs]
        )
    else:
        # 纯cosine调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    
    return scheduler

def main(config):
    # seed
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    args.root = '/mnt/sdb/pzy/Awesome-Projects/VTEvo-FSL'
    svname = args.name+'{}-shot'.format(args.shot) 
    args.work_dir = os.path.join(f'{args.root}/save', args.dataset, svname)
    utils.ensure_path(args.work_dir) 
    utils.set_log_path(args.work_dir)
    utils.log(vars(args))
    
    writer = SummaryWriter(os.path.join(args.work_dir, 'tensorboard'))  

    # Dataset
    if args.dataset in ['miniImageNet', 'tieredImageNet']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CIFAR-FS', 'FC100']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224_cifar)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224_cifar)
    elif args.dataset in ['FG-CUB', 'FG-Cars','FG-Dogs']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['CD-CUB','Places', 'Plantae']:
        args.train = f'{args.root}/dataset/{args.dataset}/base'
        args.val = f'{args.root}/dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['NWPU']:
        args.train = '/mnt/sdb/pzy/rssc_dataset/NWPU-RESISC45/train'
        args.val = '/mnt/sdb/pzy/rssc_dataset/NWPU-RESISC45/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)
    elif args.dataset in ['AID', 'UCM']:
        if args.dataset == 'AID':
            args.train = f'/mnt/sdb/pzy/rssc_dataset/AID_split1/train'
            args.val = f'/mnt/sdb/pzy/rssc_dataset/AID_split1/test'
        else:
            args.train = f'/mnt/sdb/pzy/rssc_dataset/{args.dataset}/train'
            args.val = f'/mnt/sdb/pzy/rssc_dataset/{args.dataset}/val'
        train_dataset = ImageFolder(args.train, transform=utils.transform_train_224)
        val_dataset = ImageFolder(args.val, transform=utils.transform_val_224)

    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset),len(train_dataset.classes)))
    utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset),len(val_dataset.classes)))

    # val_generated_cache = torch.load('{}/data/valid/val_{}_{}shot.pt'.format(args.root, args.dataset, str(args.shot)))  # class_id -> [shot, 3, 224, 224]
    
    # Dataloader
    n_episodes = int(len(train_dataset) / (args.way * (args.shot + 15)))
    # n_episodes = int(len(train_dataset) / (args.way * (args.shot + 15))) if args.dataset in ['NWPU'] else 200
    episode_sampler = EpisodeSampler(train_dataset.targets,n_episodes, args.way, args.shot + 15, fix_seed=False)
    train_loader = DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=args.num_workers, pin_memory=True)
    val_sampler = EpisodeSampler(val_dataset.targets, args.episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,num_workers=args.num_workers, pin_memory=True)

    args.num_classes = len(train_dataset.classes)

     # text
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

    #image
    # generated_cache = torch.load('{}/data/training/train_{}_{}shot.pt'.format(args.root, args.dataset, str(args.shot)))

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
    
    lgf = LorentzGaussianFusion(curvature=args.c)

    # resume
    if args.resume:
        init = args.resume
    else:
        # init = '{}/checkpoint/visformer-{}.pth'.format(args.root, 'miniImageNet')
        # init = '{}/checkpoint/{}/best-{}shot.pth'.format(args.root, args.dataset, args.shot)
        # checkpoint = torch.load(init,map_location=device)
        # model.encoder.load_state_dict(checkpoint['state_dict'], strict=False)
        init = '{}/checkpoint/{}/best-{}shot.pth'.format(args.root, 'AID', args.shot)
        # init = '{}/checkpoint/miniImageNet-{}-shot.pth'.format(args.root, args.shot)visformer-miniImageNet.pth
        # init = '{}/checkpoint/visformer-miniImageNet.pth'.format(args.root)
        checkpoint = torch.load(init, map_location=device)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.encoder.state_dict()

        # 过滤掉形状不匹配的键
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                print(f"[WARNING] Skip loading {k}: shape mismatch "
                      f"pretrained{v.shape} vs model{model_dict[k].shape if k in model_dict else 'N/A'}")

        model.encoder.load_state_dict(filtered_dict, strict=False)
    
    model = model.to(device)
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # optimizer
    # 1) 中等学习率组：args.lr
    mid_lr_params_id = []
    mid_lr_params_id += [id(param) for param in model.encoder.t2i.parameters()]
    mid_lr_params_id += [id(param) for param in model.encoder.t2i2.parameters()]
    mid_lr_params_id += [id(model.encoder.gate)]
    mid_lr_params_id += [id(model.encoder.log_tau)]

    # 2) 高学习率组：args.c_lr
    high_lr_params_id = []
    high_lr_params_id += [id(param) for param in model.support_controller.parameters()]
    high_lr_params_id += [id(param) for param in model.rerank_controller.parameters()]
    high_lr_params_id += [id(param) for param in model.proj_k.parameters()]
    high_lr_params_id += [id(param) for param in model.proj_q.parameters()]
    high_lr_params_id += [id(param) for param in model.proj_v.parameters()]
    high_lr_params_id += [id(param) for param in model.fc_new.parameters()]
    high_lr_params_id += [id(param) for param in model.layer_norm.parameters()]
    high_lr_params_id += [id(param) for param in model.layer_norm2.parameters()]

    # 3) 低学习率组：args.lr1（其余参数）
    other_params_id = []
    for param in model.parameters():
        pid = id(param)
        if pid not in high_lr_params_id and pid not in mid_lr_params_id:
            other_params_id.append(pid)

    high_lr_params = [p for p in model.parameters() if id(p) in high_lr_params_id]
    mid_lr_params  = [p for p in model.parameters() if id(p) in mid_lr_params_id]
    other_params   = [p for p in model.parameters() if id(p) in other_params_id]

    optimizer = AdamW([
        {'params': mid_lr_params, 'lr': args.lr,   'weight_decay': 1e-4},
        {'params': high_lr_params,  'lr': args.c_lr, 'weight_decay': 1e-4},
        {'params': other_params,   'lr': args.lr1,  'weight_decay': 1e-4}
    ])
    # optimizer = AdamW([{'params': optim_params, 'lr':args.lr, 'weight_decay': 1e-4}])

    grad_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if grad_param_count >= 1e6:
        grad_param_count = '{:.1f}M'.format(grad_param_count / 1e6)
    else:
        grad_param_count = '{:.1f}K'.format(grad_param_count / 1e3)
    utils.log('num trainable params: ' + grad_param_count)

    # === 使用PyTorch内置的Cosine调度器 ===
    if args.scheduler == 'cosine':
        warmup_epochs = max(1, args.epoch // 20) if args.warmup_epochs is None else args.warmup_epochs
        min_lr = args.lr * args.min_lr_ratio
        
        lr_scheduler = create_cosine_scheduler(
            optimizer, 
            total_epochs=args.epoch,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr
        )
        
        utils.log(f'Learning rate scheduler: Cosine Annealing with {warmup_epochs} warmup epochs')
        utils.log(f'Initial lr: {args.lr}, Min lr: {min_lr}')
    else:
        lr_scheduler = None
        utils.log('No learning rate scheduler used')

    # train
    save_epoch = 25
    max_va = 0.
    ef_epoch = 1
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    
    if args.resume:
        checkpoint = torch.load(args.root + '/' + args.resume,map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'scheduler' in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'load checkpoint at epoch {start_epoch}')
    else:
         start_epoch = 1
    
    for epoch in range(start_epoch, args.epoch + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va', 'va_pure_vis']
        aves = {k: utils.Averager() for k in aves_keys}
        
        # === 获取当前学习率并记录 ===
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)

        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch} [Train]")
        for episode in train_pbar:
            image = episode[0].to(device)
            glabels = episode[1].to(device)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).to(device)

            # generated_labels = glabels.view(args.way, args.shot+15)[:, :1]
            # generated_labels = generated_labels.contiguous().view(-1)
            # support_list = [generated_cache[int(class_id)] for class_id in generated_labels]
            # synthetic_support = torch.cat(support_list, dim=0).to(device)
            # synthetic_support = synthetic_support.view(args.way * args.shot, 3, 224, 224)

            image = image.view(args.way, args.shot+15, *image.shape[1:])
            sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
            sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

            glabels = glabels.view(args.way, args.shot+15)[:, :args.shot]
            glabels = glabels.contiguous().view(-1)

            text_features = torch.stack([semantic[idx_to_class[l.item()]] for l in glabels]).to(device)
            # _, gen_support = model.fusion(sup, text_features, args)
            # _, syn_support = model(synthetic_support)
    
            # text_features = model.t2i(text_features)
            # temp = model.log_tau.exp()
            # text_features_norm = F.normalize(text_features, dim=-1)
            # gen_support_norm = F.normalize(gen_support, dim=-1)
            # syn_support_norm = F.normalize(syn_support, dim=-1)
            
            #  kernelized volume
            # volume = volume_computation_with_kernel(
            #     text_features_norm,
            #     gen_support_norm,
            #     syn_support_norm,
            #     kernel_type='rbf',  
            #     sigma=0.5          
            # )
            # volume = volume / temp
            # volumeT = volume.T / temp
            # targets = torch.arange(volume.shape[0]).to(device)
            # align_loss = (
            #     F.cross_entropy(-volume, targets, label_smoothing=0.2) +
            #     F.cross_entropy(-volumeT, targets, label_smoothing=0.2)
            # ) / 2   

            # gen_support = gen_support.view(args.way, args.shot, -1).mean(dim=1)
            # _, query = model(que)
            # logits = F.normalize(query, dim=-1) @ F.normalize(gen_support, dim=-1).t()
            logits = model.fusion(sup, que, text_features)
            loss_cls = F.cross_entropy(logits/ args.t, labels)

            # loss = loss_cls + align_loss
            loss = loss_cls

            acc = utils.compute_acc(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_value_(model.encoder.parameters(), 15)

            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.4f}"})
        
        # eval
        # max_acc = {'k': 0, 'acc': 0}
        if epoch % ef_epoch == 0 or epoch== 1:
            # ks = np.arange(0, 101) * 0.01
            # P_acc = {}
            model.eval()
            with torch.no_grad():
                eval_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Eval {args.shot}-shot]")
                for episode in eval_pbar:
                    image = episode[0].to(device)
                    glabels = episode[1].to(device)
                    labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).to(device)

                    # generated_labels = glabels.view(args.way, args.shot+15)[:, :1]
                    # generated_labels = generated_labels.contiguous().view(-1)
                    # support_list = [val_generated_cache[int(class_id)] for class_id in generated_labels]
                    # generated_support = torch.cat(support_list, dim=0).to(device)

                    image = image.view(args.way, args.shot+15, *image.shape[1:])
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
                    acc_pure = utils.compute_acc(logits, labels)
                    aves['va_pure_vis'].add(acc_pure)

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
                    aves['va'].add(acc)

                    eval_pbar.set_postfix({"Acc_pure_vis": f"{acc_pure:.4f}", "Acc": f"{acc:.4f}"})

                    # proto_expanded, ori_proto_expanded, c, query_expanded = model(sup, que, text_features, ori_shot=sup1, fusion=True)

                    # if isinstance(c, torch.Tensor):
                    #     k_val = 1.0 / c
                    # else:
                    #     k_val = torch.tensor(1.0 / c, device=query_expanded.device)

                    # for f in ks:
                    #     # com_proto = f * support + (1 - f) * gen_support
                    #     # logits = F.normalize(query, dim=-1) @ F.normalize(com_proto, dim=-1).t()
                    #     com_proto = lorentz_linear_fusion(ori_proto_expanded, proto_expanded, f, c)
                    #     logits = -dist(query_expanded, com_proto, k=k_val)
                    #     acc = utils.compute_acc(logits, labels)
                    #     if str(f) in P_acc:
                    #         P_acc[str(f)].append(acc)
                    #     else:
                    #         P_acc[str(f)] = []
                    #         P_acc[str(f)].append(acc)

        # post #
        if lr_scheduler is not None:
            lr_scheduler.step(epoch-1)

        # key's value to item()
        for k, v in aves.items():
            aves[k] = v.item()
        
        # time of a epoch ,sum epochs and max epochs
        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * args.epoch)

        # log train loss and acc  
        log_str = 'epoch {}, train {:.4f}|{:.2f}%'.format(epoch, aves['tl'], aves['ta'] * 100)
        log_str += ', lr: {:.6f}'.format(current_lr)
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)
        
        # log val acc 
        if epoch % ef_epoch == 0 or epoch== 1:
            log_str += ', val {}-shot, pure vis: {:.4f}, fused: {:.4f}'.format(args.shot, aves['va_pure_vis'], aves['va'])
            writer.add_scalars('acc', {'val': aves['va'], 'val_pure_vis': aves['va_pure_vis']}, epoch)
        
        # save checkpoint
        checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }
        
        if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(checkpoint,os.path.join(args.work_dir, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(checkpoint, os.path.join(args.work_dir, 'best.pth'))
        
        log_str += '| MAX: {:.4f}'.format(max_va)
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        writer.flush()

if __name__ == '__main__':
    main(args)
