import MutuInfo
import Imgblock
import saliency
import resnet
import model
import dataset
from config import load_args

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader 

import os
from datetime import datetime

def save_checkpoint(model, optimizer, args, epoch):
    print('\nModel Saving...')
    model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.checkpoints, 'checkpoint_pretrain_model_' +str(epoch + 1) + '.pth'))


def main(args):
    args.batch_size = 14
    args.intervals = 100
    device = torch.device('cuda:0')
    trans = nn.Sequential(
        T.RandomResizedCrop((224, 224)),
        model.RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p = 0.3
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        model.RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p = 0.2
        ),
    )
    Myset = dataset.MyDataset("F:\\PythonProject\\testimg", transform = trans)
    trainData = DataLoader(Myset, batch_size= args.batch_size, shuffle = True, num_workers=args.num_workers)
    resnet_test = resnet.resnet50(pretrained = False)
    models = model.modifiedBYOL(args.proj_in, args.proj_out, args.M, resnet_test)
    #models = model.BYOL(resnet_test, args.proj_in, args.proj_out, args.M)
    models.train()
    models.to(device)
    optimizer = optim.SGD(models.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = torch.optim.Adam(models.parameters(), lr=args.lr, betas=(0.9,0.99))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    for epoch in range(args.epochs):
        losses, step = 0., 0.
        tag = datetime.now()
        for index, data in enumerate(trainData):     
            x1, x2, res_x, res_y, posx, posy = data
            x1, x2, res_x, res_y, posx, posy = x1.cuda(), x2.cuda(), res_x.long().cuda(), res_y.long().cuda(), posx.cuda(), posy.cuda()
            start = datetime.now()
            loss = models(x1, x2, res_x, res_y, posx, posy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses += loss.item()
            end = datetime.now()
            step += 1
            print('[Epoch: {0:3d}, step: [{1:d}/8] loss: {2:.3f}, lr: {3:.3f}'.format(int(epoch + 1), int(step), losses / step, lr_scheduler.get_last_lr()[0]))
            #print('[Epoch: {0:3d}, step: [{1:d}/8] loss: {2:.3f}'.format(int(epoch + 1), int(step), losses / step))
            print("modeltime: ", end - start, "loadtime: ", start - tag)
            tag = datetime.now()
        if epoch % args.intervals == 0:
            save_checkpoint(models, optimizer, args, epoch)


if __name__ == '__main__':
    args = load_args()
    main(args)