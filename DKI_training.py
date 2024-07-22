################
import pickle
import os
import clip
import torch
import network
from network.models import _VAE
import argparse
from utils.utils import get_dataset
from torch.utils import data
import numpy as np
import random
from utils.scheduler import PolyLR
from Prompts.load_prompts import load_prompts
from utils.utils import templates,compose_text_with_templates,calc_mean_std
from tqdm import tqdm

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0,
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
    parser.add_argument("--total_it", type = int, default =100,
                        help= "total number of optimization iterations")
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
    parser.add_argument("--training_epochs", type=int, default=50,
                        help="(default: 50)")
    #VAE parameters
    parser.add_argument("--resSize_low", type=int, default=256,
                        help="(default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="(default: 256)")
    parser.add_argument("--train", action='store_true',default=False,
                        help="mix statistics")
    parser.add_argument("--DKI_save_dir",type=str,help="mix statistics")
    parser.add_argument("--lambda_s", type = float, default=1.0, help="mix statistics")
    parser.add_argument("--Encoder_checkpoints", type=str, default='',
                        help="path to dataset")
    parser.add_argument("--Decoder_checkpoints", type=str, default='',
                        help="path to dataset")
    parser.add_argument("--weights_path", type=str, default = None,
                        help="path to VAE_weights")
    return parser


def main():
    '''=================== Init Setting ====================='''
    opts = get_argparser().parse_args()
    device = torch.device('cuda:{}'.format(opts.gpu_id) if torch.cuda.is_available() else 'cpu')
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

    if opts.weights_path and opts.train:
        VAE.load_state_dict(torch.load(opts.weights_path))
        print('load weights from', opts.weights_path)
    '''=================== Load Text Description ====================='''
    decs_path = os.path.join('../Prompts', opts.target_domain + '.txt')
    print('using prompts from', decs_path)
    Decs = load_prompts(decs_path)
    textEmbedding = torch.zeros((len(Decs), 1024))
    for idx, name in enumerate(Decs):
        target = compose_text_with_templates(name, templates())
        tokens = clip.tokenize(target).to(device)
        text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
        text_target /= text_target.norm(dim=-1, keepdim=True)
        textEmbedding[idx] = text_target
    textEmb_Mean = textEmbedding.mean(dim=0).repeat(opts.batch_size, 1).type(torch.float32).to(device)

    '''=================== Training Model ====================='''
    for epoch in range(opts.training_epochs):
        loss_sty = 0.0
        loss_rec = 0.0
        Loss = 0.0
        num_batches = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i,(img_id, tar_id, images, labels) in pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{opts.training_epochs}- lossSty {loss_sty:.2f}- lossRec {loss_rec:.2f}")
            if opts.train:
                '''=================== Training VAE ====================='''
                if i == len(train_loader) - 1 and images.shape[0] < opts.batch_size:
                    txt = textEmb_Mean[:images.shape[0]]
                else:
                    txt = textEmb_Mean

                optimD.zero_grad()
                optimE.zero_grad()
                loss = VAE.DKI_train(images.to(device),txt)
                Loss = loss['kl'] + 10 * loss['rec'] + opts.lambda_s * loss['sty']
                loss_sty = loss['sty'].item()
                loss_rec = loss['rec'].item()
                Loss.backward()
                optimE.step()
                optimD.step()
                Loss += Loss.item()
                num_batches += 1
                pbar.update(1)
                schedulerE.step()
                schedulerD.step()

        if opts.train:
            l1 = Loss / num_batches
            print(f"Epoch {epoch + 1} "
                  f"- Average Loss: {l1:.4f}")

        '''=================== Saving Data ====================='''
        if opts.train:
            if (epoch + 1) % 10 == 0:
                if not os.path.isdir(opts.DKI_save_dir):
                    os.mkdir(opts.DKI_save_dir)
                    print('-----------Make Dir {} -----------'.format(opts.DKI_save_dir))
                save_path = os.path.join(opts.DKI_save_dir,'DKI_{}.pth'.format(epoch))
                torch.save(VAE.state_dict(),save_path)
                print('-----------Model Saved as', save_path, '-----------')
'''=================== Run ====================='''
if __name__ == '__main__':
    main()

