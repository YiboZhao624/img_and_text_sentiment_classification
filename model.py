import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class Self_Attention(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Self_Attention, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
    
    def forward(self, inputs,mask):
        input_tensor, mask_tensor = inputs
        if input_tensor.dim() != 3:
            raise ValueError("The dim of inputs is required 3 but get {}".format(input_tensor.dim()))
        
        x = self.W_layer(x)
        x = torch.tanh(input_tensor)
        score = self.dropout_layer(score)

        mask = mask.unsqueeze(-1)
        padding = torch.ones_like(mask) * (-2**31+1)
        score = torch.where(mask == 0,padding,score)
        score = F.softmax(score,dim=1)

        output = torch.matmul(input_tensor, score.transpose(1,2))
        output /= input_tensor.size(-1)**0.5#归一化
        output = output.squeeze(-1)
        return output
    def build(self, input_shape):
        k = input_shape[0][-1]#dimension of word vector
        self.W_layer = nn.Linear(k,k)
        self.U_weight = nn.Parameter(torch.Tensor(k,1))
        nn.init.xavier_uniform(self.U_weight)

class Image_Text_Attention(nn.Module):
    '''
    inputs:image_emb,seq_emb,mask
    image_emb:[]
    '''
    def __init__(self, dropout_rate = 0.0):
        super(Image_Text_Attention, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
    
    def forward(self, inputs):
        image_emb, seq_emb, mask = inputs

        p = self.image_layer(image_emb)
        q = self.seq_layer(seq_emb)
        p = torch.tanh(p)
        q = torch.tanh(q)

        emb = torch.matmul(p,q.transpose(1,2))
        emb = emb + q.transpose(1,2)
        emb = torch.matmul(emb, self.V_weight)
        score = self.dropout_layer(emb)

        mask = mask.unsqueeze(1).repeat(1,score.size(1),1)
        padding = torch.ones_like(mask) * (-2**31+1)
        score = torch.where(mask == 0,padding,score)
        score = F.softmax(score,dim=-1)

        output = torch.matmul(score,seq_emb)
        output /= seq_emb.size(-1)**0.5
        return output

    def build(self, input_shape):
        self.l = input_shape[1][1]
        self.k = input_shape[1][-1]
        self.image_layer = nn.Linear(self.l,1)
        self.seq_layer = nn.Linear(self.k,1)
        self.V_weight = nn.Parameter(torch.Tensor(self.l,self.l))
        nn.init.xavier_uniform_(self.V_weight)

class VggNet(nn.Module):
    def __init__(self, out_dim=1000):
        super(VggNet,self).__init__()
        self.out_dim = out_dim
        self.model = vgg16(pretrained = True)
        num_fc = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_fc,out_dim)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True
    
    def get_cnn_block(self,out_channel,layer_num):
        module_list = []
        for i in range(layer_num):
            module_list.append(nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding=1))
            module_list.append(nn.ReLU(inplace=True))
        module_list.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*module_list)

    def get_out_block(self,hidden_units,outdim,dropout_rate):
        module_list = []
        for i in range (len(hidden_units)-1):
            module_list.append(nn.Linear(hidden_units[i],hidden_units[i+1]))
            module_list.append(nn.ReLU(inplace=True))
            module_list.append(nn.Dropout(dropout_rate))
        module_list.append(nn.Linear(hidden_units[-1],outdim))
        module_list.append(nn.SoftMarginLoss(dim=1))
        return nn.Sequential(*module_list)
    
    def forward(self,inputs):
        if len(inputs.shape) != 4:
            raise ValueError("The dim of inputs is required 4 but get {}".format(len(inputs.shape)))
        
        x = inputs
        cnn_block_list = [self.cnn_block1,self.cnn_block2,self.cnn_block3,self.cnn_block4,self.cnn_block5]

        for cnn_block in cnn_block_list:
            x = cnn_block(x)
        x = self.flatten(x)

        output = self.out_block(x)
        return output

class VistaNet(nn.Module):
    def __init__(self, embedding_dim, vocab_size, out_dim=4096,attention_dropout=0.0,gru_units=[64,128],class_num=3):
        super(VistaNet,self).__init__()
        self.vgg16 = VggNet(out_dim=out_dim)
        self.word_embedding = nn.Embedding(vocab_size,embedding_dim)
        self.word_self_attention = Self_Attention(attention_dropout)
        self.img_seq_attention = Image_Text_Attention(attention_dropout)
        self.doc_self_attention = Self_Attention(attention_dropout)
        self.BiGRU_layer1 = nn.GRU(input_size=embedding_dim,
                                   hidden_size=gru_units[0],
                                   batch_first=True,
                                   bidirectional=True)
        self.BiGRU_layer2 = nn.GRU(input_size=2*gru_units[0],
                                   hidden_size=gru_units[1],
                                   batch_first=True,
                                   bidirectional=True)
        self.output_layer = nn.Linear(gru_units[1]*2,class_num)
    
    def forward(self,inputs):
        image_inputs,text_inputs,mask=inputs

        image_emb =self.vgg16(image_inputs)
        word_emb = self.word_embedding(text_inputs)
        word_emb,_ = self.BiGRU_layer1(word_emb)
        seq_emb = self.word_self_attention(word_emb,mask)

        seq_emb,_ = self.BiGRU_layer2(seq_emb)
        image_emb = image_emb.unsqueeze(0)
        
        mask = torch.argmax(mask,dim=1)
        mask = mask.unsqueeze(0)

        doc_emb = self.img_seq_attention(image_emb,seq_emb,mask)
        D_emb = self.doc_self_attention(doc_emb)

        output = self.output_layer(D_emb)
        return output
    