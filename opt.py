import argparse
import network

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','ACDC','gta5'], help='Name of dataset')
    parser.add_argument("--ACDC_sub", type=str, default="night",
                        help = "specify which subset of ACDC  to use")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=200e3,
                        help="epoch number (default: 200k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--train_aug",action='store_true',default=False,
                        help="train on augmented features using CLIP")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=10,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--forward_pass",action='store_true',default=False,
                        help="forward pass to update BN statistics")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--freeze_BB", action='store_true',default=False,
                        help="Freeze the backbone when training")
    parser.add_argument("--ckpts_path", type = str ,
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    #validation
    parser.add_argument("--val_results_dir", type=str,help="Folder name for validation results saving")

    parser.add_argument("--val_data_root", type=str, default='/media/yujiaguo/29719e14-8bb8-4829-90a4-727c2e661fc4/JCY/dataset/ACDC/',
                        help="path to Dataset")
    parser.add_argument("--val_dataset", type=str, default='ACDC',
                        choices=['cityscapes','ACDC','gta5'], help='Name of dataset')

    #Augmented features
    parser.add_argument("--path_mu_sig", type=str)
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")

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
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')

    return parser