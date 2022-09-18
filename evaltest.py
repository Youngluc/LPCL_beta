import torch
import torch.nn as nn
import torchvision.models as models
import model
import resnet

class Resnet_Detection(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.load_state_dict(backbone.backbone.state_dict())
        for name, param in self.backbone.named_parameters():
            if name == 'fc.weight' or name == 'fc.bias':
                param.requires_grad = False
            else:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

# obj = torch.load('./moco_v1_200ep_pretrain.pth.tar')
# obj = obj['state_dict']

resnet_test = resnet.resnet50(pretrained = False)
Mymodel = model.modifiedBYOL(2048, 2048, 7, resnet_test)
Mymodel.load_state_dict({k.replace('module.',''):v for k, v in torch.load("./checkpoints/gpu4_checkpoint_pretrain_model_100.pth", map_location=torch.device('cpu'))['model_state_dict'].items()})
res = Resnet_Detection(Mymodel)
torch.save({'state_dict': res.state_dict()}, './detection/resnet.pth.tar')

# for k, v in obj.items():
#     print(k)

# for k in res.state_dict():
#     print(k)