import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader 

import numpy as np
import random
import copy
import cv2

import Imgblock
import saliency
import MutuInfo
import resnet
import dataset

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0) 


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def Sortbysocre(x, M):
    temp = x.cpu()
    resized = tensor_to_np(temp)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    sal = saliency.LC(gray)
    ha = []
    step = int(224 / M)
    for i in range(M * M):
        pos_a, pos_b = ((i // M) * step, (i % M) * step)
        patch_gray = sal[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step]
        patch_rgb = resized[int(pos_a) : int(pos_a) + step, int(pos_b) : int(pos_b) + step, 0:]
        ha.append((np.mean(MutuInfo.RGBEntropy(patch_rgb)), np.sum(patch_gray), patch_rgb, i))
        #ha.append((MutuInfo.Entropy(patch_gray), patch1, i))
    sorted_res = sorted(ha, key = lambda x : x[0], reverse = True)[0: (M**2) // 2]
    sorted_res.sort(key = lambda x : x[1], reverse = True)
    x = x.cuda()
    return sorted_res


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class Randompatch(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.blocknum_sqrt = M

    def forward(self, x):
        return Imgblock.TensorBlock(x, self.blocknum_sqrt)


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        # layers is a str list containing the names for each layer such as ['layer4', 'avgpool']
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self.features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        c = torch.empty(0)
        downsample = nn.AdaptiveAvgPool2d((7, 7))
        for name, output in self.features.items():
            #print(name, output.shape)
            output = downsample(output)
            if name == '4':
                c = output
            elif name == '8':
                pass
            else:
                c = torch.cat([c, output], dim = 1)
        return c, self.features
    

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.squeeze()
        return x


class BYOL(nn.Module):
    def __init__(self, backbone, proj_input_size, proj_output_size, M, hidden_size = 4096, img_size = 224):
        super().__init__()
        # online network
        self.backbone = backbone
        self.projector = MLP(proj_input_size, proj_output_size, hidden_size)
        self.sq = Reshape()
        self.online_encoder = nn.Sequential(
            self.backbone,
            self.sq,
            self.projector
        )

        # target network
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = MLP(proj_output_size, proj_output_size, hidden_size)

        # augmentation
        SimCLR_aug = nn.Sequential(
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
            T.RandomResizedCrop((img_size, img_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.Myaug = Randompatch(M)
        self.aug1 = SimCLR_aug
        self.aug2 = nn.Sequential(
            self.aug1
            #self.Myaug
        )
        self.Normlized = T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))

    def update_moving_average(self, moving_average_decay):
        tau = moving_average_decay
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data


    def forward(self, x1, x2):
        #x1, x2 = self.aug1(x), self.aug2(x)
        x1, x2 = self.Normlized(x1), self.Normlized(x2)
        
        z1, z2 = self.online_encoder(x1), self.online_encoder(x2)

        p1, p2 = self.predictor(z1), self.predictor(z2)
        
        with torch.no_grad():
            #x1, x2 = self.Myaug(x1), self.Myaug(x2)
            self.update_moving_average(0.996)
            zt_1, zt_2 = self.target_encoder(x1), self.target_encoder(x2)
            zt_1.detach_()
            zt_2.detach_()
            
        
        loss1 = loss_fn(p1, zt_2)
        loss2 = loss_fn(p2, zt_1)
        loss = loss1 + loss2
        
        return loss.mean()


class modifiedBYOL(nn.Module):
    def __init__(self, proj_input_size, proj_output_size, M, backbone, layers = ['4', '5', '6', '7', '8'], hidden_size = 4096, img_size = 224, pretrain = True):
        super().__init__()
        # online network 
        self.backbone = backbone
        self.projector = MLP(proj_input_size, proj_output_size, hidden_size)

        self.predictor = MLP(proj_output_size, proj_output_size, hidden_size)

        # target network
        self.target_encoder = copy.deepcopy(self.backbone)
        #self.target_encoder = FeatureExtractor(net, layers)
        self.target_projector = copy.deepcopy(self.projector)

        # local part(without global and pos)
        self.l_projector = MLP(proj_input_size, proj_output_size, hidden_size)
        self.l_predictor = MLP(proj_output_size, proj_output_size, hidden_size)
        self.target_l_projector = copy.deepcopy(self.l_projector)

        # local part(with global and pos)
        self.local_projector = MLP(proj_input_size * 2 + 512, proj_output_size, hidden_size)
        self.local_predictor = MLP(proj_output_size, proj_output_size, hidden_size)
        self.target_local_projector = copy.deepcopy(self.local_projector)

        # augmentation
        SimCLR_aug = nn.Sequential(
            T.RandomResizedCrop((img_size, img_size)),
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
            # T.ToTensor()
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.Normlized = T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))

        #self.Myaug = Randompatch(M)
        self.Myaug = nn.Sequential()
        self.aug1 = SimCLR_aug
        self.aug2 = nn.Sequential(self.aug1)

        self.M = M
        self.pretrain = pretrain

    def update_moving_average(self, moving_average_decay):
        tau = moving_average_decay
        for online, target in zip(self.backbone.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
        
        for online_proj, target_proj in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_proj.data = tau * target_proj.data + (1 - tau) * online_proj.data

        for l_proj, target_l_proj in zip(self.l_projector.parameters(), self.target_l_projector.parameters()):
            target_l_proj.data = tau * target_l_proj.data + (1 - tau) * l_proj.data

        for local_proj, target_local_proj in zip(self.local_projector.parameters(), self.target_local_projector.parameters()):
            target_local_proj.data = tau * target_local_proj.data + (1 - tau) * local_proj.data


    def forward(self, x1, x2, xres, yres, posx, posy, patch_num = 24):
        x1, x2 = self.Normlized(x1), self.Normlized(x2)

        features_global1, cat_feature_maps1 = self.backbone(x1) # backbone is resnet50 removed fc layers
        features_global2, cat_feature_maps2 = self.backbone(x2)

        y1, y2 = features_global1.squeeze(), features_global2.squeeze()
        z1, z2 = self.projector(y1), self.projector(y2)

        if self.pretrain == False: # eval, only use the backbone
            return z1
        
        p1, p2 = self.predictor(z1), self.predictor(z2)

        with torch.no_grad():
            self.update_moving_average(0.996)
            x1_0, x2_0 = self.Myaug(x1), self.Myaug(x2) # before sent to the target encoder, apply randompatch aug to x1 and x2
            target_features_global1, target_cat_feature_maps1 = self.target_encoder(x1_0)
            target_features_global2, target_cat_feature_maps2 = self.target_encoder(x2_0)
            yt_1, yt_2 = target_features_global1.squeeze(), target_features_global2.squeeze()
            zt_1, zt_2 = self.target_projector(yt_1), self.target_projector(yt_2)
            zt_1.detach_()
            zt_2.detach_()
            
        global_loss1 = loss_fn(p1, zt_2)
        global_loss2 = loss_fn(p2, zt_1)
        global_total_loss = global_loss1 +global_loss2

        xres.detach_() # index and pos tensor should not take part in backward
        yres.detach_()
        # posx.detach_() # pos is a function of res, so also not take part in backward
        # posy.detach_()

        batchsize = x1.shape[0]
        # now we will process the local part, firstly prepare the related tensor  (bs, 2048, 49)
        cat_feature_maps1 = cat_feature_maps1.reshape(cat_feature_maps1.shape[0], cat_feature_maps1.shape[1], -1)
        cat_feature_maps2 = cat_feature_maps2.reshape(cat_feature_maps2.shape[0], cat_feature_maps2.shape[1], -1)
        #print(cat_feature_maps1.shape)
        target_cat_feature_maps1 = target_cat_feature_maps1.reshape(
            target_cat_feature_maps1.shape[0],
            target_cat_feature_maps1.shape[1],
            -1
        )
        target_cat_feature_maps2 = target_cat_feature_maps2.reshape(
            target_cat_feature_maps2.shape[0],
            target_cat_feature_maps2.shape[1],
            -1
        )


        patch_feat_x1 = torch.stack(
            [cat_feature_maps1[i, :, xres[i]] for i in range(batchsize)]
        )

        patch_feat_x2 = torch.stack(
            [cat_feature_maps2[i, :, yres[i]] for i in range(batchsize)]
        )

        target_patch_feat_x1 = torch.stack(
            [target_cat_feature_maps1[i, :, xres[i]] for i in range(batchsize)]
        )

        target_patch_feat_x2 = torch.stack(
            [target_cat_feature_maps2[i, :, yres[i]] for i in range(batchsize)]
        )


        lz_1 = self.l_projector(patch_feat_x1.permute(0, 2, 1).reshape(batchsize * patch_num, -1))  
        lz_2 = self.l_projector(patch_feat_x2.permute(0, 2, 1).reshape(batchsize * patch_num, -1)) 
        lp_1, lp_2 = self.l_predictor(lz_1), self.l_predictor(lz_2)

        with torch.no_grad():
            lzt_1 = self.target_l_projector(target_patch_feat_x1.permute(0, 2, 1).reshape(batchsize * patch_num, -1)) 
            lzt_2 = self.target_l_projector(target_patch_feat_x2.permute(0, 2, 1).reshape(batchsize * patch_num, -1))
            lzt_1.detach_()
            lzt_2.detach_()

        l_loss1 = loss_fn(lp_1.reshape(batchsize, patch_num, -1), lzt_2.reshape(batchsize, patch_num, -1)).mean(dim = 1) # for each images, mean their patch's loss
        l_loss2 = loss_fn(lp_2.reshape(batchsize, patch_num, -1), lzt_1.reshape(batchsize, patch_num, -1)).mean(dim = 1) # shape(128,24) after mean == shape(128)
        l_total_loss = l_loss1 + l_loss2

        # all of feature maps has been reshaped to the local patch feature
        # (batchsize, 2048, 24) for patch feature shape
        # pos_v1 = torch.zeros(batchsize, 512, 24)
        # pos_v2 = torch.zeros(batchsize, 512, 24)
        # patchfeat_x1 = torch.empty(batchsize, 2048, 24)
        # for i in range(batchsize):
        Vcat1 = torch.cat(
            [patch_feat_x1, y1.unsqueeze(2).expand(-1, -1, patch_num), posx], 
            dim = 1).permute(0, 2, 1).float().reshape(batchsize * patch_num, -1) # (128, 24, 2048 + 2048 + 512); if don't add .float(), will be automatically float64(double)
        Vcat2 = torch.cat(
            [patch_feat_x2, y2.unsqueeze(2).expand(-1, -1, patch_num), posy], 
            dim = 1).permute(0, 2, 1).float().reshape(batchsize * patch_num, -1)
        target_Vcat1 = torch.cat(
            [target_patch_feat_x1, yt_1.unsqueeze(2).expand(-1, -1, patch_num), posx],
            dim = 1).permute(0, 2, 1).float().reshape(batchsize * patch_num, -1)
        target_Vcat2 = torch.cat(
            [target_patch_feat_x2, yt_2.unsqueeze(2).expand(-1, -1, patch_num), posy], 
            dim = 1).permute(0, 2, 1).float().reshape(batchsize * patch_num, -1)

        # target_Vcat1.detach_()
        # target_Vcat2.detach_()
        vz_1, vz_2 = self.local_projector(Vcat1), self.local_projector(Vcat2)
        vp_1, vp_2 = self.local_predictor(vz_1), self.local_predictor(vz_2)

        with torch.no_grad():
            vzt_1, vzt_2 = self.target_local_projector(target_Vcat1), self.target_local_projector(target_Vcat2)
            vzt_1.detach_()
            vzt_2.detach_()

        local_loss1 = loss_fn(vp_1.reshape(batchsize, patch_num, -1), vzt_2.reshape(batchsize, patch_num, -1)).mean(dim = 1) # for each images, mean their patch's loss
        local_loss2 = loss_fn(vp_2.reshape(batchsize, patch_num, -1), vzt_1.reshape(batchsize, patch_num, -1)).mean(dim = 1) # shape(128,24) after mean == shape(128)
        local_total_loss = local_loss1 + local_loss2


        # lamb = 0.5
        # loss = lamb * global_total_loss + (1 - lamb) * local_total_loss

        alpha, beta, gamma = 1/3, 1/3, 1/3
        loss = alpha * global_total_loss + beta * l_total_loss + gamma * local_total_loss
        
        return loss.mean()

from datetime import datetime

if __name__ == '__main__':

    device = torch.device('cuda:0')
    trans = nn.Sequential(
        T.RandomResizedCrop((224, 224)),
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
    Myset = dataset.MyDataset("F:\\PythonProject\\testimg", transform = trans)
    trainData = DataLoader(Myset, batch_size= 14, shuffle = True, num_workers = 4)
    resnet_test = resnet.resnet50(pretrained = False)
    models = modifiedBYOL(2048, 2048, 7, resnet_test).float()
    #models = BYOL(resnet_test, 2048, 2048, 7)
    models.train()
    models.to(device)
    #optimizer = optim.SGD(models.parameters(), lr=0.01, momentum=0.9)
    optimizer=torch.optim.Adam(models.parameters(),lr=0.01,betas=(0.9,0.99))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for epoch in range(100):
        losses, step = 0., 0.
        tag = datetime.now()
        for index, data in enumerate(trainData):
            
            x1, x2, res_x, res_y, posx, posy = data
            x1, x2, res_x, res_y, posx, posy = x1.cuda(), x2.cuda(), res_x.long().cuda(), res_y.long().cuda(), posx.cuda(), posy.cuda()
            start = datetime.now()
            #print(start)
            loss = models(x1, x2, res_x, res_y, posx, posy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses += loss.item()
            end = datetime.now()
            #print(end)
            step += 1
            print('[Epoch: {0:3d}, step: [{1:d}/8] loss: {2:.3f}, lr: {3:.3f}'.format(int(epoch + 1), int(step), losses / step, lr_scheduler.get_last_lr()[0]))
            #print('[Epoch: {0:3d}, step: [{1:d}/8] loss: {2:.3f}'.format(int(epoch + 1), int(step), losses / step))
            print("modeltime: ", end - start, "loadtime: ", start - tag)
            tag = datetime.now()