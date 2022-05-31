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
es_patience = 100  # Early Stopping patience - for how many epochs with no improvements to wait
best_accuracy = 0.0

def eval_train(model ,dataloader_val,device,criterion,text_writer=None,adj_brightness=1.0, adj_contrast=1.0 ):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            # inputs = transforms.functional.adjust_brightness(inputs,adj_brightness)
            # inputs = transforms.functional.adjust_contrast(inputs,adj_contrast)
            logps,_,_ = model.forward(inputs)
            # logps = logps.squeeze()
            batch_loss = criterion(logps, labels)
            #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
            test_loss += batch_loss.item()
            #                     print("labels : ",labels)
            #                     print("logps  : ",logps)
            # equals = labels == (logps > 0.5)
            equals = np.argmax(labels.detach().cpu().numpy(), 1) == np.argmax(logps.detach().cpu().numpy(), 1)
            accuracy += np.mean(equals)
            #                     print("equals   ",equals)
            # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    #                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))
    print(
          f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloader_val):.3f}")
    if text_writer != None:
        text_writer.write('Test loss %.4f, Test accuracy  %.4f \n' % (
            test_loss / len(dataloader_val), accuracy / len(dataloader_val)))
        text_writer.flush()
    model.train()
    return accuracy
def train_xray_cnn(model,criterion,train_set = '../../extract_raw_img',val_set ="" ,image_size=256,\
              batch_size=16,resume = '',lr=0.003,num_workers=8,checkpoint="checkpoint",epochs=20,print_every=1000, \
              adj_brightness=1.0, adj_contrast=1.0):
    patience = es_patience
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
    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3, 6, 9, 12, 15, 18], gamma = 0.8)

    INPUT_PATH = train_set

    df_train = pd.read_csv(Path(INPUT_PATH, "train_df.csv"))
    df_train['Target'] = df_train['Target'].apply(lambda x: x.strip())
    ## Xoa anh co 2 class
    df_train = df_train.drop(df_train[df_train.Target.str.count(' ') > 0].index)
    df_train['Target'] = df_train['Target'].apply(lambda x: int(x))
    df_train_15 = df_train.drop(df_train[df_train.Target > 15].index)
    df_train_outline = df_train.drop(df_train[df_train.Target < 16].index)
    df_train_data = df_train_15[:1200]
    df_val_data = df_train_15[1200:]
    dataset_train = MyDataset(df_train_data, INPUT_PATH=INPUT_PATH)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)
    dataset_val = MyDataset(df_val_data, INPUT_PATH=INPUT_PATH)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False,
                                                   num_workers=4)
    dataset_outline = MyDataset(df_train_outline, INPUT_PATH=INPUT_PATH, num_class=30)
    dataloader_outline = torch.utils.data.DataLoader(dataset_outline, batch_size=1, shuffle=False,
                                                     num_workers=1)
    if resume != '':
        model.load_state_dict(torch.load( os.path.join(checkpoint, resume)))

    # train_losses, test_losses = [], []
    # import time
    text_writer = open(os.path.join(checkpoint, 'train.csv'), 'w')
    sys.stdout = open(os.path.join(checkpoint, 'model.txt'), "w")
    torchsummary.summary(model, (1, image_size, image_size))
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    model.train()

    running_loss = 0
    steps = 0
    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader_train):
            #     for inputs, labels in tqdm(testloader):
            steps += 1
            #         labels = np.array([labels])
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            # print(inputs.size())
            #         inputs, labels = inputs.to(device), labels[1].float().to(device)
            # inputs = transforms.functional.adjust_brightness(inputs,adj_brightness)
            # inputs = transforms.functional.adjust_contrast(inputs,adj_contrast)
            optimizer.zero_grad()
            out, _,_ = model.forward(inputs)
            # print(labels)
            # print(logps)
            # logps = logps.squeeze()
            loss = criterion(out, labels)
            #         loss = F.binary_cross_entropy_with_logits(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # time.sleep(0.05)
            # if steps % print_every == 0:
            if False:
                test_loss = 0
                train_loss = 0
                accuracy = 0
                accuracy_train = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader_val:
                        inputs, labels = inputs.float().to(device), labels.float().to(device)
                        # inputs = transforms.functional.adjust_brightness(inputs, adj_brightness)
                        # inputs = transforms.functional.adjust_contrast(inputs, adj_contrast)
                        logps = model.forward(inputs)
                        logps = logps.squeeze()
                        batch_loss = criterion(logps, labels)
                        #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
                        test_loss += batch_loss.item()
                        #                     print("labels : ",labels)
                        #                     print("logps  : ",logps)
                        equals = labels == (logps > 0.5)
                        #                     print("equals   ",equals)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #                 train_losses.append(running_loss/len(trainloader))
                #             test_losses.append(test_loss/len(testloader))
                    ################################################################################
                    for inputs, labels in dataloader_train:
                        inputs, labels = inputs.float().to(device), labels.float().to(device)
                        # inputs = transforms.functional.adjust_brightness(inputs, adj_brightness)
                        # inputs = transforms.functional.adjust_contrast(inputs, adj_contrast)
                        logps = model.forward(inputs)
                        logps = logps.squeeze()
                        batch_loss = criterion(logps, labels)
                        #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
                        train_loss += batch_loss.item()
                        #                     print("labels : ",labels)
                        #                     print("logps  : ",logps)
                        equals = labels == (logps > 0.5)
                        #                     print("equals   ",equals)
                        accuracy_train += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloader_val):.3f}")
                text_writer.write('Epoch %d, Train loss %.4f, Test loss %.4f, Test accuracy  %.4f \n' % (
                epoch, running_loss / print_every, test_loss / len(dataloader_val), accuracy / len(dataloader_val)))
                text_writer.flush()
                scalar_dict = {"Loss/train": train_loss/len(dataloader_train),"Loss/test": test_loss/len(dataloader_val),"Acc/train": accuracy_train/len(dataloader_train),"Acc/test": accuracy/len(dataloader_val)}
                log.write_scalar(scalar_dict=scalar_dict, global_step=steps)
                # running_loss = 0
                # steps = 0
                model.train()
            scalar_dict = {"Loss/train_step": loss.item()}
            log.write_scalar(scalar_dict=scalar_dict, global_step=steps)
        scheduler.step()
        print("Epoch  ", epoch, " running loss : ",
              running_loss / len(dataloader_train))
        text_writer.write('Epoch %.4f, running loss  %.4f \n' % (
            epoch, running_loss / len(dataloader_train)))
        running_loss = 0
        accuracy_score__= eval_train(model ,dataloader_val,device,criterion,text_writer,adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        torch.save(model.state_dict(), os.path.join(checkpoint, 'model_pytorch_%d.pt' % epoch))

        if accuracy_score__ >= best_accuracy:
            best_accuracy = accuracy_score__
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            print("best : at epoch  ",epoch, 'with accuracy ', best_accuracy)
            torch.save(model.state_dict(), os.path.join(checkpoint, 'model_best.pt'))
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_accuracy))
                break
        torch.save(model.state_dict(), os.path.join(checkpoint, 'model_last.pt'))
    return


def train_branchy_xray_cnn(model,criterion,train_set = '../../extract_raw_img',val_set ="" ,image_size=256,\
              batch_size=16,resume = '',lr=0.003,num_workers=8,checkpoint="checkpoint",epochs=20,print_every=1000, \
              adj_brightness=1.0, adj_contrast=1.0):
    patience = es_patience
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
    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3, 6, 9, 12, 15, 18], gamma = 0.8)

    INPUT_PATH = train_set

    df_train = pd.read_csv(Path(INPUT_PATH, "train_df.csv"))
    df_train['Target'] = df_train['Target'].apply(lambda x: x.strip())
    ## Xoa anh co 2 class
    df_train = df_train.drop(df_train[df_train.Target.str.count(' ') > 0].index)
    df_train['Target'] = df_train['Target'].apply(lambda x: int(x))
    df_train_15 = df_train.drop(df_train[df_train.Target > 15].index)
    df_train_outline = df_train.drop(df_train[df_train.Target < 16].index)
    df_train_data = df_train_15[:1200]
    df_val_data = df_train_15[1200:]
    dataset_train = MyDataset(df_train_data, INPUT_PATH=INPUT_PATH)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)
    dataset_val = MyDataset(df_val_data, INPUT_PATH=INPUT_PATH)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False,
                                                   num_workers=4)
    dataset_outline = MyDataset(df_train_outline, INPUT_PATH=INPUT_PATH, num_class=30)
    dataloader_outline = torch.utils.data.DataLoader(dataset_outline, batch_size=1, shuffle=False,
                                                     num_workers=1)
    if resume != '':
        model.load_state_dict(torch.load( os.path.join(checkpoint, resume)))

    # train_losses, test_losses = [], []
    # import time
    text_writer = open(os.path.join(checkpoint, 'train.csv'), 'w')
    sys.stdout = open(os.path.join(checkpoint, 'model.txt'), "w")
    torchsummary.summary(model, (1, image_size, image_size))
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    model.train()

    running_loss = 0
    steps = 0
    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader_train):
            #     for inputs, labels in tqdm(testloader):
            steps += 1
            #         labels = np.array([labels])
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            # print(inputs.size())
            #         inputs, labels = inputs.to(device), labels[1].float().to(device)
            # inputs = transforms.functional.adjust_brightness(inputs,adj_brightness)
            # inputs = transforms.functional.adjust_contrast(inputs,adj_contrast)
            optimizer.zero_grad()
            out, out_mid1,out_mid2 = model.forward(inputs)
            # print(labels)
            # print(logps)
            # out = out.squeeze()
            loss = criterion(out, labels) + criterion(out_mid1, labels)  + criterion(out_mid2, labels)
            #         loss = F.binary_cross_entropy_with_logits(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # time.sleep(0.05)
            # if steps % print_every == 0:
            if False:
                test_loss = 0
                train_loss = 0
                accuracy = 0
                accuracy_train = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader_val:
                        inputs, labels = inputs.float().to(device), labels.float().to(device)
                        # inputs = transforms.functional.adjust_brightness(inputs, adj_brightness)
                        # inputs = transforms.functional.adjust_contrast(inputs, adj_contrast)
                        logps = model.forward(inputs)
                        logps = logps.squeeze()
                        batch_loss = criterion(logps, labels)
                        #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
                        test_loss += batch_loss.item()
                        #                     print("labels : ",labels)
                        #                     print("logps  : ",logps)
                        equals = labels == (logps > 0.5)
                        #                     print("equals   ",equals)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                #                 train_losses.append(running_loss/len(trainloader))
                #             test_losses.append(test_loss/len(testloader))
                    ################################################################################
                    for inputs, labels in dataloader_train:
                        inputs, labels = inputs.float().to(device), labels.float().to(device)
                        # inputs = transforms.functional.adjust_brightness(inputs, adj_brightness)
                        # inputs = transforms.functional.adjust_contrast(inputs, adj_contrast)
                        logps = model.forward(inputs)
                        logps = logps.squeeze()
                        batch_loss = criterion(logps, labels)
                        #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
                        train_loss += batch_loss.item()
                        #                     print("labels : ",labels)
                        #                     print("logps  : ",logps)
                        equals = labels == (logps > 0.5)
                        #                     print("equals   ",equals)
                        accuracy_train += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloader_val):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloader_val):.3f}")
                text_writer.write('Epoch %d, Train loss %.4f, Test loss %.4f, Test accuracy  %.4f \n' % (
                epoch, running_loss / print_every, test_loss / len(dataloader_val), accuracy / len(dataloader_val)))
                text_writer.flush()
                scalar_dict = {"Loss/train": train_loss/len(dataloader_train),"Loss/test": test_loss/len(dataloader_val),"Acc/train": accuracy_train/len(dataloader_train),"Acc/test": accuracy/len(dataloader_val)}
                log.write_scalar(scalar_dict=scalar_dict, global_step=steps)
                # running_loss = 0
                # steps = 0
                model.train()
            scalar_dict = {"Loss/train_step": loss.item()}
            log.write_scalar(scalar_dict=scalar_dict, global_step=steps)
        scheduler.step()
        print("Epoch  ", epoch, " running loss : ",
              running_loss / len(dataloader_train))
        text_writer.write('Epoch %.4f, running loss  %.4f \n' % (
            epoch, running_loss / len(dataloader_train)))
        running_loss = 0
        accuracy_score__= eval_train(model ,dataloader_val,device,criterion,text_writer,adj_brightness=adj_brightness, adj_contrast=adj_brightness)
        torch.save(model.state_dict(), os.path.join(checkpoint, 'model_pytorch_%d.pt' % epoch))

        if accuracy_score__ >= best_accuracy:
            best_accuracy = accuracy_score__
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            print("best : at epoch  ",epoch, 'with accuracy ', best_accuracy)
            torch.save(model.state_dict(), os.path.join(checkpoint, 'model_best.pt'))
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_accuracy))
                break
        torch.save(model.state_dict(), os.path.join(checkpoint, 'model_last.pt'))
    return


if __name__ == "__main__":
    from models.xception import xception

    model = xception(pretrained=False)
    criterion = nn.BCELoss()
    train_xeay_cnn(model,criterion,train_set='../../../data/extract_raw_img_test', val_set='../../../data/extract_raw_img_test', checkpoint="../../../model/xception/",)
    # eval_capsule(val_set ='../../../extract_raw_img_test',checkpoint="../../../model/capsule/",resume=6)

