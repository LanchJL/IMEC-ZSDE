################
import pickle
import os
import clip
import torch
import network
from network.models import _VAE
import argparse
from torch.utils import data
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from utils.scheduler import PolyLR
from Prompts.load_prompts import load_prompts
from utils.utils import templates,compose_text_with_templates,calc_mean_std,get_dataset
from tqdm import tqdm

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','gta5'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default = 'RN50',
                        help= "backbone name" )
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False,
                        help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # target domain description
    parser.add_argument("--ckpt", default='pretrain/CS_source.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--freeze_BB", action='store_true', default=True,
                        help="Freeze the backbone when training")
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")
    parser.add_argument("--target_domain",type =str,default='night',
                        help="target domain name")
    #VAE parameters
    parser.add_argument("--resSize_low", type=int, default=256,
                        help="(default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="(default: 256)")
    parser.add_argument("--train", action='store_true',default=False,
                        help="mix statistics")
    parser.add_argument("--DKI_save_dir",type=str,help="mix statistics")
    parser.add_argument("--lambda_s", type = float, default=1.0, help="mix statistics")
    parser.add_argument("--DKI_checkpoints", type=str, default='',
                        help="path to dataset")
    # IMEC parameters
    parser.add_argument("--Style_dir", type=str,help= "path for learnt parameters saving")
    parser.add_argument("--p", type = int, default =200,
                        help= "total number of optimization iterations")
    parser.add_argument("--l", type = int, default =20,
                        help= "total number of optimization iterations")
    return parser

def save_style(opts,learnt_mu_f1,learnt_std_f1,img_id,index):
    for k in range(learnt_mu_f1.shape[0]):
        learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
        learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())
        stats = {}
        stats['mu_f1'] = learnt_mu_f1_
        stats['std_f1'] = learnt_std_f1_

        img_path = opts.Style_dir + '/' + img_id[k].split('/')[-1]
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        with open(img_path + '/' + str(index) + '.pkl', 'wb') as f:
            pickle.dump(stats, f)
def main():
    '''=================== Init Setting ====================='''
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    '''=================== Datasets ====================='''
    train_dst, val_dst = get_dataset(opts.dataset, opts.data_root, opts.crop_size, data_aug=False)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
                                   drop_last=False)  # drop_last=True to ignore single-image batches.
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    '''=================== Build Model ====================='''
    model = network.modeling.__dict__[opts.model](num_classes=19,
                                                  BB= opts.BB,replace_stride_with_dilation=[False,False,False]).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    VAE = _VAE(opts,model.backbone).to(device)

    '''=================== Load Text Description ====================='''
    decs_path = os.path.join('../Prompts', opts.target_domain + '.txt')
    Decs = load_prompts(decs_path)
    textEmbedding = torch.zeros((len(Decs), 1024))
    for idx, name in enumerate(Decs):
        target = compose_text_with_templates(name, templates())
        tokens = clip.tokenize(target).to(device)
        text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
        text_target /= text_target.norm(dim=-1, keepdim=True)
        textEmbedding[idx] = text_target
    textEmb_Loc = textEmbedding.type(torch.float32).to(device)

    '''=================== Load DKI ====================='''
    VAE.load_state_dict(torch.load(opts.DKI_checkpoints))
    VAE.eval()
    print('-----------Load DKI from ', opts.DKI_checkpoints,'-----------')


    if not os.path.isdir(opts.Style_dir):
        os.mkdir(opts.Style_dir)

    loss_sty = 0.0
    loss_rec = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i,(img_id, tar_id, images, labels) in pbar:
        pbar.set_description(f"- lossSty {loss_sty:.2f}- lossRec {loss_rec:.2f}")

        '''=================== ELP ====================='''
        Content = VAE.backbone(images.to(device), trunc1=False, trunc2=False,
                               trunc3=False, trunc4=False, get1=True, get2=False, get3=False, get4=False)
        for i in range(1, textEmb_Loc.shape[0]):
            sampling = None
            _, _, Content_mean = calc_mean_std(Content)
            n0style, n0 = VAE.Generate_Style(Content_mean, sampling, noise=False)

            learnt_mu_f1 = n0style[:,:256]
            learnt_std_f1 = n0style[:,256:]
            save_style(opts,learnt_mu_f1,learnt_std_f1,img_id,0)

            txt = textEmb_Loc[i].repeat(opts.batch_size, 1)
            if  i == len(train_loader) - 1 and images.shape[0] < opts.batch_size:
                txt = txt[:images.shape[0]]
            iters = 0
            while iters < opts.l:
                iters+=1
                style, _ = VAE.Generate_Style(Content_mean, n0, noise=False)
                n0.required_grad = True

                Feat = VAE.AdaIN(Content, style)
                Feat_t1 = VAE.avgpool(Feat)
                cls = VAE.backbone(Feat_t1, trunc1=True, trunc2=False,
                                   trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=False)
                loss_style = (1 - torch.cosine_similarity(cls, txt, dim=-1)).mean()
                n0.retain_grad()
                loss_style.backward(retain_graph=True)
                n0 = n0 - (20.0 / loss_style.item()) * n0.grad.data

            style, _ = VAE.Generate_Style(Content_mean, n0, noise=False)
            learnt_mu_f1 = style[:,:256]
            learnt_std_f1 = style[:,256:]

            save_style(opts,learnt_mu_f1,learnt_std_f1,img_id,i)
        pbar.update(1)



'''=================== Run ====================='''
if __name__ == '__main__':
    main()

