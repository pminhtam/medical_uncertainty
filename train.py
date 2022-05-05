import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import torch.nn as nn
import argparse
import json
# from pytorch_model.train import *
# from tf_model.train import *
def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection")
    parser.add_argument('--train_set', default="/hdd/tam/kaggle/xray", help='path to train data ')
    parser.add_argument('--val_set', default="", help='path to test data ')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--workers', type=int, default=4, help='number wokers for dataloader ')
    parser.add_argument('--checkpoint',default = "checkpoint",required=False, help='path to checkpoint ')
    parser.add_argument('--gpu_id',type=int, default = 0, help='GPU id ')
    parser.add_argument('--resume',type=str, default = '', help='Resume from checkpoint ')
    parser.add_argument('--print_every',type=int, default = 5000, help='Print evaluate info every step train')
    parser.add_argument('--loss',type=str, default = "bce", help='Loss function use')

    subparsers = parser.add_subparsers(dest="model", help='Choose 1 of the model from: capsule,drn,resnext50, resnext ,gan,meso,xception')

    ## torch
    parser_xception = subparsers.add_parser('xception', help='Xceptionnet')
    parser_xception = subparsers.add_parser('normal', help='normal')
    parser_xception = subparsers.add_parser('normal_branchy', help='normal_branchy')

    ## adjust image
    parser.add_argument('--adj_brightness',type=float, default = 1, help='adj_brightness')
    parser.add_argument('--adj_contrast',type=float, default = 1, help='adj_contrast')

    return parser.parse_args()

def get_criterion_torch(arg_loss):
    criterion = None
    if arg_loss == "bce":
        criterion = nn.BCELoss()
    elif arg_loss == "focal":
        from pytorch_model.focal_loss import FocalLoss
        criterion = FocalLoss(gamma=2)
    return criterion
if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 0 if int(args.gpu_id) >=0 else -1
    adj_brightness = float(args.adj_brightness)
    adj_contrast = float(args.adj_contrast)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    with open(os.path.join(args.checkpoint, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if model== "capsule":
        from pytorch_model.train_torch import train_capsule
        train_capsule(train_set = args.train_set,val_set = args.val_set,gpu_id=gpu_id,manualSeed=args.seed,resume=args.resume,beta1=args.beta1, \
                      dropout=0.05,image_size=args.image_size,batch_size=args.batch_size,lr=args.lr, \
                      num_workers=args.workers,checkpoint=args.checkpoint,epochs=args.niter,\
                      adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
    elif model == "xception":
        from models.train_torch import train_cnn
        from models.xception import xception2
        from losses.mse_outline import mse_loss
        model = xception2(pretrained=True)
        # criterion = get_criterion_torch(args.loss)
        criterion = mse_loss
        print("xception")
        train_cnn(model,criterion=criterion,train_set = args.train_set,val_set = args.val_set,image_size=args.image_size,resume=args.resume, \
                  batch_size=args.batch_size,lr=args.lr,num_workers=args.workers,checkpoint=args.checkpoint,\
                  epochs=args.niter,print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass


    elif model == "normal":
        from models.wavelet_model.model_normal import NormalModel
        from models.train_torch import train_cnn
        from models.xception import xception2
        from losses.mse_outline import mse_loss

        model = NormalModel(in_channel=1)
        # criterion = get_criterion_torch(args.loss)
        criterion = mse_loss
        print("normal")
        train_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass

    elif model == "normal_branchy":
        from models.wavelet_model.model_normal import NormalModel
        from models.train_torch import train_branchy_cnn
        from models.xception import xception2
        from losses.mse_outline import mse_loss

        model = NormalModel(in_channel=1)
        # criterion = get_criterion_torch(args.loss)
        criterion = mse_loss
        print("normal")
        train_branchy_cnn(model, criterion=criterion, train_set=args.train_set, val_set=args.val_set,
                  image_size=args.image_size, resume=args.resume, \
                  batch_size=args.batch_size, lr=args.lr, num_workers=args.workers, checkpoint=args.checkpoint, \
                  epochs=args.niter, print_every=args.print_every,adj_brightness=adj_brightness,adj_contrast=adj_contrast)
        pass
# ---------------------------------------------------------------------------------------------
