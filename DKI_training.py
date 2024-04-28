################
#script to augment features with CLIP
import pickle
import os
import clip
import torch
import network
from network.models import _VAE
import argparse
from main import get_dataset
from torch.utils import data
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from utils.scheduler import PolyLR
from Prompts.load_prompts import load_prompts
from utils.utils import templates,compose_text_with_templates,calc_mean_std
from tqdm import tqdm

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to dataset")
    parser.add_argument("--save_dir", type=str,
                        help= "path for learnt parameters saving")
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
    parser.add_argument("--total_it", type = int, default =100,
                        help= "total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False,
                        help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # target domain description
    parser.add_argument("--domain_desc", type=str , default = "driving at night.",
                        help = "description of the target domain")
    parser.add_argument("--ckpt", default='pretrain/CS_source.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--freeze_BB", action='store_true', default=True,
                        help="Freeze the backbone when training")
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")
    parser.add_argument("--target_domain",type =str,default='night',
                        help="target domain name")
    parser.add_argument("--training_epochs", type=int, default=300,
                        help="(default: 50)")
    parser.add_argument("--source_domain", type=str, default='daylight',
                        help="source domain name")
    #VAE parameters
    parser.add_argument("--attSize", type=int, default=1024,
                        help="(default: 1024)")
    parser.add_argument("--nz", type=int, default=256,
                        help="(default: 256)")
    parser.add_argument("--ndh", type=int, default=1024,
                        help="(default: 4096)")
    parser.add_argument("--ngh", type=int, default=4096,
                        help="(default: 4096)")
    parser.add_argument("--resSize_low", type=int, default=256,
                        help="(default: 256)")
    parser.add_argument("--resSize_high", type=int, default=1024,
                        help="(default: 1024)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="(default: 256)")
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument("--train", action='store_true',default=False,
                        help="mix statistics")
    parser.add_argument("--DKI_save_dir",type=str,help="mix statistics")
    parser.add_argument("--lambda_s", type = float, default=1.0, help="mix statistics")
    parser.add_argument("--Encoder_checkpoints", type=str, default='',
                        help="path to dataset")
    parser.add_argument("--Decoder_checkpoints", type=str, default='',
                        help="path to dataset")
    return parser


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
    optimE = torch.optim.Adam(VAE.encoder.parameters(), lr=opts.lr)
    optimD = torch.optim.Adam(VAE.decoder.parameters(), lr=opts.lr)
    schedulerE = PolyLR(optimE, (opts.training_epochs + 1) * len(train_loader), power=0.9)
    schedulerD = PolyLR(optimD, (opts.training_epochs + 1) * len(train_loader), power=0.9)

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
    if opts.train:
        textEmb_Mean = textEmbedding.mean(dim=0).repeat(opts.batch_size, 1).type(torch.float32).to(device)

    if opts.train == False:
        VAE.encoder.load_state_dict(torch.load(opts.Encoder_checkpoints))
        VAE.decoder.load_state_dict(torch.load(opts.Decoder_checkpoints))
        VAE.eval()
        print('--Load VAE MODEL--Encoder--', opts.Encoder_checkpoints)
        print('--Load VAE MODEL--Decoder--', opts.Decoder_checkpoints)

        '''=================== Other Settings ====================='''
        if not os.path.isdir(opts.save_dir):
            os.mkdir(opts.save_dir)

    '''=================== Training Model ====================='''
    for epoch in range(opts.training_epochs):
        loss = {}
        loss['VAE'] = 0.0
        num_batches = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        VAE_Loss = 0.0
        for i,(img_id, tar_id, images, labels) in pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{opts.training_epochs}- lossVAE {VAE_Loss:.2f}")
            if opts.train:
                '''=================== Training VAE ====================='''
                optimD.zero_grad()
                optimE.zero_grad()
                VAE_Loss = VAE.VAR_train(images.to(device),textEmb_Mean)

                VAE_Loss.backward(retain_graph=True)
                optimE.step()
                optimD.step()

                loss['VAE'] += VAE_Loss.item()

                num_batches += 1
                pbar.update(1)
                schedulerE.step()
                schedulerD.step()

        if opts.train:
            l1 = loss['VAE'] / num_batches
            print(f"Epoch {epoch + 1} "
                  f"- Average Loss_VAE: {l1:.4f}")

        '''=================== Saving Data ====================='''
        if opts.train:
            if (epoch) % 20 == 0:
                if not os.path.isdir(opts.DKI_save_dir):
                    os.mkdir(opts.DKI_save_dir)
                    print('========== Make Dir {} =========='.format(opts.DKI_save_dir))
                save_path_e = os.path.join(opts.DKI_save_dir,'Encoder{}.pth'.format(epoch))
                save_path_d = os.path.join(opts.DKI_save_dir,'Decoder{}.pth'.format(epoch))
                torch.save(VAE.encoder.state_dict(),save_path_e)
                torch.save(VAE.decoder.state_dict(),save_path_d)

'''=================== Run ====================='''
if __name__ == '__main__':
    main()

