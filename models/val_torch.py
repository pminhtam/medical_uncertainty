import torch
import random
import os
import sys
# import torchvision.transforms as transforms
from torch.optim import Adam
# import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

from torch import optim
import torch.nn as nn
from tqdm import tqdm
import torchsummary
from utils.logging import Logger
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from dataloader.xray_loader import MyDataset
import numpy as np
from utils.utils import calc_uncertainly

def eval_cnn(model,criterion,train_set = '../../extract_raw_img',val_set ="" ,image_size=256,\
              batch_size=16,resume = '',lr=0.003,num_workers=8,checkpoint="checkpoint",epochs=20,print_every=1000, \
              adj_brightness=1.0, adj_contrast=1.0):
    log = Logger(os.path.join(checkpoint,"logs"))
    best_accuracy = 0.0
    # from pytorch_model.focal_loss import FocalLoss
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed_all(0)
        cudnn.benchmark = True
    model = model.to(device)
    # criterion = nn.BCELoss().to(device)
    # criterion = FocalLoss(gamma=2).to(device)
    # criterion = criterion.to(device)

    INPUT_PATH = train_set

    df_train = pd.read_csv(Path(INPUT_PATH, "train_df.csv"))
    df_train['Target'] = df_train['Target'].apply(lambda x: x.strip())
    ## Xoa anh co 2 class
    df_train = df_train.drop(df_train[df_train.Target.str.count(' ') > 0].index)
    df_train['Target'] = df_train['Target'].apply(lambda x: int(x))
    df_train_15 = df_train.drop(df_train[df_train.Target > 15].index)
    df_train_outline = df_train.drop(df_train[df_train.Target <= 15].index)
    df_train_data = df_train_15[:1200]
    df_val_data = df_train_15[1200:]
    dataset_train = MyDataset(df_train_data, INPUT_PATH=INPUT_PATH)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False,
                                                   num_workers=1)
    dataset_val = MyDataset(df_val_data, INPUT_PATH=INPUT_PATH)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False,
                                                   num_workers=1)
    dataset_outline = MyDataset(df_train_outline, INPUT_PATH=INPUT_PATH, num_class=30)
    dataloader_outline = torch.utils.data.DataLoader(dataset_outline, batch_size=1, shuffle=False,
                                                     num_workers=1)
    if resume != '':
        model.load_state_dict(torch.load( os.path.join(checkpoint, resume)))
    else:
        print("ko co model")
        exit(1)
    model.eval()

    running_loss = 0
    steps = 0
    for inputs, labels in dataloader_val:
    # for inputs, labels in dataloader_outline:
        #     for inputs, labels in tqdm(testloader):
        steps += 1
        #         labels = np.array([labels])
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        out, _,_ = model.forward(inputs)
        # print(labels)
        # print(logps)
        # logps = logps.squeeze()
        print(calc_uncertainly(out))

        # loss = criterion(out, labels)

        # running_loss += loss.item()
    return

if __name__ == "__main__":
    from models.xception import xception

    model = xception(pretrained=False)
    criterion = nn.BCELoss()
    # eval_capsule(val_set ='../../../extract_raw_img_test',checkpoint="../../../model/capsule/",resume=6)

