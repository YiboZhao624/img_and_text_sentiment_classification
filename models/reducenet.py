import torch
import math,copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from specialtokens import SpecialTokens

class CNN(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.layer4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.layer5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.pooling =  nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = x.float()
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) 
        x = F.relu(self.layer4(x)) 
        x = F.relu(self.layer5(x))
        x = self.pooling(x) 
        x = x.reshape(x.size(0), -1)
        feature = self.fc(x)
        return feature
    
#class 1: word Embedding
class Embeddings(nn.Module):
    '''
    to embedding the word to the vector,d_model is the hyperparameter,default is 512.
    vocab is the lenth of vocabulary, it is 1299 in this lab.
    it needs to be trained
    '''
    def __init__(self,d_model:int,vocab:int):
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab ,d_model)
        self.d_model = d_model
        self.vocab = vocab
    def forward(self,x):
        #print(x)
        #print(self.d_model)
        #print(self.vocab)
        x = x.int()
        return self.lut(x)*math.sqrt(self.d_model)

#class 2: Positional Embedding
#IMPORTANT: NOTHING to UPDATE in this class
class Positional_Encoding(nn.Module):
    '''
    as the report, it uses the formula to calculate the Positional_Embedding
    '''
    def __init__(self, d_model:int, dropout:float, max_len=5000):
        super(Positional_Encoding,self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        positional_embedding = torch.zeros(max_len,d_model)#initialize the tensor
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        positional_embedding[:,0::2] = torch.sin(position * div_term)#even index
        positional_embedding[:,1::2] = torch.cos(position * div_term)#odd index
        positional_embedding = positional_embedding.unsqueeze(0)
        #[max_len,d_model] to [1,max_len,d_model],for batch size
        self.register_buffer('pe',positional_embedding)

    def forward(self,x):
        '''
        x is the vector after embedding. it is [seq_length,d_model]
        if in batch.it is [batch_size,seq_length,d_model]
        '''
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)
    
#Part 3: multi-head attention
def attention(query, key, value, mask = None, dropout = None):
    '''
    query,key,value are the tensors. Q and K has the same size
    V: [batch_size, num_head, seq_length_1, d_K] = K.size
    Q: [batch_size, num_head, seq_length_2, d_K]
    d_K is a new hyperparameter. to set the dimension of words.
    it is a little different from my report.
    '''
    d_k = query.size(-1)
    #print("qkv:",query.shape,key.shape)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    #print("score:",scores.shape)
    #print("mask:",mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #after softmax, masked place will set as 0
    p_attention = F.softmax(scores,dim=-1)
    #[batch_size, num_head, seq_length_2, seq_length_1]
    if dropout is not None:
        p_attention = dropout(p_attention)
    return torch.matmul(p_attention,value),p_attention

#before class 3,we need a clones function
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#class 3: multi-head attention
class Multi_Head_Attention(nn.Module):
    def __init__(self, num_head, d_model, dropout = 0.1):
        super(Multi_Head_Attention,self).__init__()
        assert d_model % num_head == 0 # why we got this assert? without it, what will change?
        self.d_K = d_model // num_head 
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attention = None
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask = None):
        '''
        query:[batch_size,seq_len_2,d_model]
        key:[batch_size,seq_len_1,d_model]  = value
        '''
        #print('multi head attention forward')
        batch_size = query.size(0)
        query,key,value = [l(x).view(batch_size,-1,self.num_head,self.d_K)
                           .transpose(1,2) for l,x in 
                           zip(self.linears,(query,key,value))]
        x,self.attention = attention(query,key,value,mask=mask,dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.num_head*self.d_K)
        return self.linears[-1](x)

#class 4: sublayerconnection
#before that we need the class 7: Layer_Norm
class Layer_Norm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(Layer_Norm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        #two parameters. vector. trainable. [features=d_model]
        self.eps = eps
    def forward(self,x):
        #print("layer norm forward")
        #x.size = [batch_size, seq_length, d_model]
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim = True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

#class 4: sublayerconnection
class Sub_Layer_Connection(nn.Module):
    '''
    residual connection and a layer norm.
    '''
    def __init__(self,size,dropout):
        super(Sub_Layer_Connection,self).__init__()
        self.norm = Layer_Norm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#class 5: Positionwise_Feed_Forward
class Positionwise_Feed_Forward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(Positionwise_Feed_Forward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        #first fcc, [d_model,d_ff]
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        #print("pff forward")
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

#class 6: Encoder_Layer
class Encoder_Layer(nn.Module):
    def __init__(self,size:int,self_attention,feed_forward,dropout):
        '''
        size is d_model
        self_attention is the object Multi_Head_Attention, is the first sublayer
        feed_forward is the object Postionwise_Feed_Forward, seconde sublayer
        '''
        super(Encoder_Layer,self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(Sub_Layer_Connection(size,dropout),2)
        self.size = size
    def forward(self,x,mask):
        x = self.sublayer[0](x,lambda x:self.self_attention(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

#class 7 is defined above
#class 8:Encoder
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = Layer_Norm(layer.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    #it won't change the size of x

class DocEmbedder(nn.Module):
    def __init__(self,config,dataset):
        super(DocEmbedder,self).__init__()

        self.N = config["N"]
        self.d_model = config["d_model"]
        self.d_ff = config["d_ff"]
        self.num_head = config["num_head"]
        self.dropout = config["dropout"]
        self.source_vocab = dataset.vocab_size
        c = copy.deepcopy#deep copy/clone
        attention  = Multi_Head_Attention(self.num_head,self.d_model)#object
        ff = Positionwise_Feed_Forward(self.d_model,self.d_ff,self.dropout)#object
        position = Positional_Encoding(self.d_model,self.dropout)
    
        
        self.encoder = Encoder(Encoder_Layer(self.d_model,c(attention),c(ff),self.dropout),self.N)
        # two objects
        self.source_embedder = nn.Sequential(Embeddings(self.d_model,self.source_vocab),c(position))
        # two embedded, include word embedding and positional embedding
        self.vocab = dataset
        self.device = config["device"]
        
    def forward(self,batch):
        
        text_index = batch['source_index_batch']
        mask = batch['source_mask_vector']
                                                      
        if self.device:
            text_index = text_index.cuda(self.device)
            mask = mask.cuda(self.device)
        
        doc_vec = self.encode(text_index,mask)             
        return doc_vec
                                                      
    def encode(self,source,source_mask):
        return self.encoder(self.source_embedder(source)
                            ,source_mask)

class ReduceNet(nn.Module):
    def __init__(self, config,dataloader):
        '''
            embedding size: the dimension of the word vector
            out_dim:
            vocab_size: how many vocabulary in the text.
            attention_dropout:
            gru_units:
            class_num: the number of sentiment. In this instance, it should be 3.
        '''
        super(ReduceNet,self).__init__()
        self.class_num = config['class_num']
        self.dataloader = dataloader
        self.vgg16 = CNN(in_channels=3,out_dim=config['out_dim']) #img to vector
        #print(config['attention_head'])
        self.doc_embedder = DocEmbedder(config,dataloader)
        self.output_layer1 = nn.Linear(config["out_dim"]+config['embedding_dim'],config['class_num'])
        self.device = config['device']
    def forward(self,batch):

        image_inputs = batch['source_img_batch']
        text_index = batch['source_index_batch']
        mask_pad = batch['source_mask_vector']
        mask_pad =mask_pad.unsqueeze(1).unsqueeze(2)
        len = mask_pad.size(-1)
        mask_forward = torch.triu(torch.ones(len, len))
        mask_forward = mask_forward.unsqueeze(0).unsqueeze(1)
        mask = torch.where(mask_pad == 1, float('-inf'), mask_forward)

        #text_vector = self.dataloader.convert_ids_to_vector(text_inputs['sequence'])
        tag = batch['tag_batch']
        tag = F.one_hot(tag,self.class_num).float()
        
        if self.device:
            image_inputs = image_inputs.cuda(self.device)
            mask = mask.cuda(self.device)
            text_index = text_index.cuda(self.device)
            tag = tag.cuda(self.device)

        image_emb =self.vgg16(image_inputs)
        text_vector = self.doc_embedder({"source_index_batch":text_index,"source_mask_vector":mask})
        # image_emb = image_emb.unsqueeze(1)#(batch_size,1,4096)
        #text_vector and image_vector, then concentate
        text_vector = torch.mean(text_vector,dim=1)

        doc_emb = torch.cat((text_vector,image_emb),dim=1)        
        output = self.output_layer1(doc_emb)
        output = output.squeeze(dim=1)
        
        loss = nn.CrossEntropyLoss()
        #print(output.shape)
        #print(tag)
        loss(output,tag)
        loss_dict = {
            "result":output,
            "loss":loss(output,tag)
        }
        return loss_dict
    
    def model_test(self,batch,dataloader,is_display=False):
        
        image_inputs = batch['source_img_batch']
        text_index = batch['source_index_batch']
        mask = batch['source_mask_vector']
        #text_vector = self.dataloader.convert_ids_to_vector(text_inputs['sequence'])
        
        if self.device:
            image_inputs = image_inputs.cuda(self.device)
            mask = mask.cuda(self.device)
            text_index = text_index.cuda(self.device)
        
        image_emb =self.vgg16(image_inputs)
        text_vector = self.doc_embedder({"source_index_batch":text_index,"source_mask_vector":mask})
        # image_emb = image_emb.unsqueeze(1)#(batch_size,1,4096)
        #text_vector and image_vector, then concentate
        text_vector = torch.mean(text_vector,dim=1)

        doc_emb = torch.cat((text_vector,image_emb),dim=1)        
        output = self.output_layer1(doc_emb)
        output = output.squeeze(dim=1)
        return output
