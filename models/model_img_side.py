import torch
import math,copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class VggNet(nn.Module):
    def __init__(self, config):
        super(VggNet,self).__init__()
        self.device = config['device']
        self.class_num = config['class_num']
        self.model = vgg16(pretrained = True)
        num_fc = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_fc,self.class_num)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True
    def forward(self, batch):
        x = batch['source_img_batch']
        tag = batch['tag_batch']
        tag = F.one_hot(tag,self.class_num).float()
        
        if self.device:
            x = x.cuda(self.device)
            tag = tag.cuda(self.device)
    
        x = x.float()
        x = x.permute(0,3,1,2)
        x = self.model(x)

        loss = nn.CrossEntropyLoss()
        #print(output.shape)
        #print(tag)
        loss(x,tag)
        loss_dict = {
            "result":x,
            "loss":loss(x,tag)
        }
        return loss_dict
    
    def model_test(self,batch,dataloader,is_display=False):
        
        x = batch['source_img_batch']
        if self.device:
            x = x.cuda(self.device)
        x = x.float()
        x = x.permute(0,3,1,2)
        x = self.model(x)
        return x
    


