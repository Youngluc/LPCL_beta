from PIL import Image
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader 
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim  
import model
import resnet
import torch
import random
import MutuInfo
import cv2
import numpy as np
import saliency
from datetime import datetime

import matplotlib.pyplot as plt


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0) 

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def Sortbysocre(x, M):
    temp = x.cpu()
    resized = tensor_to_np(temp)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #sal = saliency.LC(gray)
    sal = gray
    ha = []
    step = int(224 / M)
    for i in range(M * M):
        pos_a, pos_b = ((i // M) * step, (i % M) * step)
        patch_gray = sal[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        patch_rgb = resized[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step, 0:]
        #patch = gray[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        #ha.append((MutuInfo.Entropy(patch), np.sum(patch_gray), patch, i))
        ha.append((np.mean(MutuInfo.RGBEntropy(patch_rgb)), np.sum(patch_gray), patch_rgb, i))
        #ha.append((MutuInfo.Entropy(patch_gray), patch1, i))
    sorted_res = sorted(ha, key = lambda x : x[0], reverse = True)[0: 42]
    sorted_res.sort(key = lambda x : x[1], reverse = True)
    return np.array(sorted_res[0: 24], dtype = object)


class MyDataset(Dataset):
    def __init__(self, path, transform = None):
        fh = os.listdir(path)
        imgs = []
        for line in fh:
            # line = line.rstrip()
            words = line.split()
            imgs.append(path + "\\" + words[0])
            self.imgs = imgs 
            self.transform = transform

    def __getitem__(self, index):

        fn = self.imgs[index]
        #print(fn)
        t00 = datetime.now()
        img = Image.open(fn).convert('RGB')
        t0 = datetime.now()
        
        t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        img = t(img) 
        
        
        
        if self.transform is not None:
            img1 = self.transform(img) 
            img2 = self.transform(img)

        #print("augtime: ", datetime.now()-t0)
        time1 = datetime.now()
        xset = Sortbysocre(img1, 7)
        yset = Sortbysocre(img2, 7)
        time2 = datetime.now()

        # x, y, posx, posy = MutuInfo.matchPatch_list(xset, yset)
        # print("_x:", x)
        # print("_y:", y)
        #x, y, posx, posy = MutuInfo.matchPatch_mutual(xset, yset)
        # print("info_x:", x)
        # print("info_y:", y)
        #x, y, posx, posy = MutuInfo.matchPatch_dist(xset, yset) # matchpatch function is a version of Euclidean distance
        # print("dist_x:", x)
        # print("dist_y:", y)
        x, y, posx, posy = MutuInfo.matchPatch_dist(xset, yset)
        #x1, y1, posx1, posy1 = MutuInfo.matchPatch_dist(xset, yset)
        #y_0, x_0, posy_0, posx_0 = MutuInfo.matchPatch(yset, xset)
        # print("cosi_x:", x)
        # print("cosi_y:", y)
        # print("y_0:", y_0)
        # print("x_0:", x_0)
        time3 = datetime.now()


        # print(time3 - time2, time2 - time1, time3 - time1)
        # print("t0-t:", t0 - t00, " t1-t0", time1 - t0, time3 - t00)
        return img1, img2, torch.tensor(x), torch.tensor(y), torch.tensor(posx), torch.tensor(posy)

    def __len__(self):
        return len(self.imgs)
    

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


if __name__ == '__main__':
    #print('[Epoch: {0:4d}, step : [{1:d} / 4] loss: {2:.3f}'.format(1, 2, 3 / 5))
    
    
    device = torch.device('cuda:0')
    t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    trans = nn.Sequential(
        T.RandomResizedCrop((224, 224), scale= (0.75, 1.0)),
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p = 0.3
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p = 0.2
        ),
    )
    #trans = T.Compose([T.Resize((224, 224))])
    Myset = MyDataset("F:\\PythonProject\\testimg", transform = trans)
    Myset.__getitem__(1)
    trainData = DataLoader(Myset, batch_size= 112, shuffle = True)
    tag = datetime.now()
    for x1, x2 in enumerate(trainData):
        now = datetime.now()
        print("loadtime: ", now - tag)
        tag = datetime.now()
