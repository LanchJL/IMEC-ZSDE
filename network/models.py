import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import calc_mean_std
from torch.nn import init
from .backbone import resnet_clip
import os

class _Segmentation(nn.Module):
    def __init__(self, backbone, classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def transfer_forward(self, low, input_shape, activation=None):
        features = {}
        features['low_level'] = activation(low)
        features['out'] = self.backbone(features['low_level'], trunc1=True, trunc2=False,
                                        trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=True)
        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features


    def forward(self, x, transfer=False,activation=None, text_embedding = None):
        input_shape = x.shape[-2:]
        features = {}
        features['low_level'] = self.backbone(x, trunc1=False, trunc2=False,
                                              trunc3=False, trunc4=False, get1=True, get2=False, get3=False, get4=False)
        if transfer:
            sampling = None
            for i in range(2):
                features['low_level'], sampling = self.VAE_Iter(features['low_level'], self.text, sampling)
            features['low_level'] = activation(features['low_level'])

        features['out'] = self.backbone(features['low_level'], trunc1=True, trunc2=False,
                                        trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=True)
        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class _VAE(nn.Module):
    def __init__(self, opts, backbone = None):
        super(_VAE, self).__init__()
        input_dim = 2*opts.resSize_low
        latent_dim = 2*opts.resSize_low
        d_dim = opts.resSize_low
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, input_dim)
        )
        self.encoder.apply(weights_init_kaiming)
        self.decoder.apply(weights_init_kaiming)

        self.backbone = backbone
        self.opts = opts

        self.rec_loss = nn.MSELoss()

        self.avgpool = nn.AdaptiveAvgPool2d((56, 56))

        self.Sampling = None

    def _built_backbone(self, replace_stride_with_dilation = [False,False,False]):
        model_url = "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
        model_path = resnet_clip._download(model_url, os.path.expanduser("~/.cache/clip"))
        with open(model_path, 'rb') as opened_file:
            backbone = torch.jit.load(opened_file, map_location="cpu").eval()
            self.backbone = resnet_clip.build_model(backbone.state_dict(),
                                               replace_stride_with_dilation=replace_stride_with_dilation)

    def calc_feat_mean_std(self, input, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = input.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = input.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std, torch.cat([feat_mean, feat_std], dim = 1)

    def AdaIN(self,content_feat, style_feat):
        size = content_feat.size()
        N, C = size[:2]
        style_mean = style_feat[:, :self.opts.resSize_low].view(N, C, 1, 1)
        style_std = style_feat[:, self.opts.resSize_low:].view(N, C, 1, 1)
        content_mean, content_std, _ = calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def latent_loss(self, z_mean, z_stddev, eps=1e-5):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq + eps) - 1)

    def VAE_Iter(self, images, text, sampling = None):
        Feat, sampling = self.Generate_T(images, Sampling = sampling)
        Feat_t1 = self.avgpool(Feat)
        cls = self.backbone(Feat_t1, trunc1=True, trunc2=False,
                      trunc3=False, trunc4=False, get1=False, get2=False, get3=False, get4=False)
        loss = (1-torch.cosine_similarity(cls,text,dim=-1)).mean()
        sampling.retain_grad()
        loss.backward(retain_graph=True)
        sampling = sampling - (20 / loss.item()) * sampling.grad.data
        style = self.decoder(sampling)
        return style, sampling

    def Generate(self, Content, alpha = 0.3, Sampling = None):
        Content = self.backbone(Content,trunc1=False,trunc2=False,
        trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
        if Sampling is None:
            _, _, MeanStd = calc_mean_std(Content)
            MeanStd = self.encoder(MeanStd.squeeze(-1).squeeze(-1))
            Mean = MeanStd[:, :self.opts.resSize_low]
            Std = MeanStd[:, self.opts.resSize_low:]
            noise = torch.randn_like(Mean)
            Sampling = Mean + noise * Std
        Sampling.required_grad = True
        Style = self.decoder(Sampling)
        Output_Feat = self.AdaIN(content_feat=Content, style_feat=Style)
        Output_Feat = alpha * Output_Feat + (1-alpha) * Content
        return Output_Feat, Sampling

    def Generate_Style(self, Content, Sampling, noise=False):
        if Sampling is None:
            MeanStd = self.encoder(Content.squeeze(-1).squeeze(-1))
            Mean = MeanStd[:, :self.opts.resSize_low]
            Std = MeanStd[:, self.opts.resSize_low:]
            if noise:
                noise = torch.randn_like(Mean)
            else:
                noise = torch.ones_like(Mean)
            Sampling = Mean + noise * Std
            Sampling.require_grad = True
        Style = self.decoder(Sampling)
        return Style, Sampling

    def Get_Layers_Feature(self, Feat):
        Feats = []
        Feats.append(self.avgpool(Feat))
        Feats.append(self.backbone(Feats[0], trunc1=True, trunc2=False,
                      trunc3=False, trunc4=False, get1=False, get2=True, get3=False, get4=False))
        Feats.append(self.backbone(Feats[1], trunc1=True, trunc2=True,
                      trunc3=False, trunc4=False, get1=False, get2=False, get3=True, get4=False))
        Feats.append(self.backbone(Feats[2], trunc1=True, trunc2=True,
                      trunc3=True, trunc4=False, get1=False, get2=False, get3=False, get4=True))
        Feats.append(self.backbone(Feats[3], trunc1=True, trunc2=True,
                      trunc3=True, trunc4=True, get1=False, get2=False, get3=False, get4=False, attnpool = True))
        return Feats

    def VAR_train(self, images, txt):
        feature = {}
        feature['low_level'] = self.backbone(images,trunc1=False,trunc2=False,
        trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
        _, _, feature['cat'] = calc_mean_std(feature['low_level'])

        feature['cat'] = self.encoder(feature['cat'].squeeze(-1).squeeze(-1))
        feature['latent_mean'] = feature['cat'][:, :self.opts.resSize_low]
        feature['latent_std'] = feature['cat'][:, self.opts.resSize_low:]

        noise = torch.randn_like(feature['latent_mean'])
        feature['noise_latent'] = feature['latent_mean'] + noise * feature['latent_std']
        feature['rec'] = self.decoder(feature['noise_latent'])

        feature['AdaIN'] = self.AdaIN(feature['low_level'], feature['rec'])

        loss_l = self.latent_loss(feature['latent_mean'], feature['latent_std'])
        loss_r = self.rec_loss(feature['rec'], feature['cat'].detach())

        feature['AdaIN'] = self.avgpool(feature['AdaIN'])
        feature['CLIP'] = self.backbone(feature['AdaIN'],trunc1=True,trunc2=False,
        trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
        loss_s = (1-torch.cosine_similarity(feature['CLIP'],txt)).mean()
        return loss_l + 5 * loss_r + self.opts.lambda_s * loss_s
