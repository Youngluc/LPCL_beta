import Imgblock
import saliency
import resnet
import model
import dataset
import MutuInfo
from config import load_args

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader 

import os
from datetime import datetime

from downutils import encode_labels, plot_history
from downtrain import train_model, test
import torchvision.datasets.voc as voc

class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``T.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)
        
    
    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)


import torchvision.models as models


class Resnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.load_state_dict(backbone.backbone.state_dict())
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_ftrs, 20)
        for name, param in self.backbone.named_parameters():
            if name == 'fc.weight' or name == 'fc.bias':
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)



def main(args): 
    args.batch_size = 14
    args.intervals = 100
    device = torch.device('cuda:0')
    resnet_test = resnet.resnet50(pretrained = False)
    Mymodel = model.modifiedBYOL(args.proj_in, args.proj_out, args.M, resnet_test)
    Mymodel.load_state_dict(torch.load("./checkpoints/checkpoint_pretrain_model_1.pth")['model_state_dict'])
    backbone = Mymodel.backbone

    testmodel = Resnet(Mymodel)


    net = Resnet(Mymodel)
    net.to(device)


    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    transformations = T.Compose([T.Resize((300, 300)),
#                                      T.RandomChoice([
#                                              T.CenterCrop(300),
#                                              T.RandomResizedCrop(300, scale=(0.80, 1.0)),
#                                              ]),                                      
                                      T.RandomChoice([
                                          T.ColorJitter(brightness=(0.80, 1.20)),
                                          T.RandomGrayscale(p = 0.25)
                                          ]),
                                      T.RandomHorizontalFlip(p = 0.25),
                                      T.RandomRotation(25),
                                      T.ToTensor(), 
                                      T.Normalize(mean = mean, std = std),
                                      ])
        
    transformations_valid = T.Compose([T.Resize(330), 
                                          T.CenterCrop(300), 
                                          T.ToTensor(), 
                                          T.Normalize(mean = mean, std = std),
                                          ])

    transformations_test = T.Compose([T.Resize(330), 
                                          T.FiveCrop(300),  
                                          T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                                          T.Lambda(lambda crops: torch.stack([T.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                          ])
    # from PIL import Image
    # t = Image.open('F:\\PythonProject\\Pytorch\\Info\\data\\VOCdevkit\\VOC2012\\JPEGImages\\2007_000027.jpg')
    # print(transformations_test(t))
    data_dir = './data/'

    dataset_train = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='train', 
                                      download=False, 
                                      transform=transformations, 
                                      target_transform=encode_labels)
    
    train_loader = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True)

    
    
    # Create validation dataloader
    dataset_valid = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=False, 
                                      transform=transformations_valid, 
                                      target_transform=encode_labels)
    
    valid_loader = DataLoader(dataset_valid, batch_size=16, num_workers=4)

    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    #print("optimized parameters",parameters)
    optimizer = optim.SGD([
            {'params': parameters, 'lr': 0.01, 'momentum': 0.9}
            ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)
    log_file = open('./log/log.txt', "w+")
    model_dir = './down_model/'
    trn_hist, val_hist = train_model(net, device, optimizer, scheduler, train_loader, valid_loader, model_dir, 1, 1, log_file)
    
    dataset_test = PascalVOC_Dataset(data_dir,
                                      year='2012', 
                                      image_set='val', 
                                      download=False, 
                                      transform=transformations_test, 
                                      target_transform=encode_labels)
    
    test_loader = DataLoader(dataset_test, batch_size=16, num_workers=0, shuffle=False)
    loss, ap= test(net, device, test_loader, returnAllScores=False)
    print(loss)
    print(ap)

if __name__ == '__main__':
    args = load_args()
    main(args)