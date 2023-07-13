import torch
import math,copy
import torch.nn as nn
import torch.nn.functional as F

def attention(query, key, value, mask = None, dropout = None):
    '''
    query,key,value are the tensors. Q and K has the same size
    V: [batch_size, num_head, seq_length_1, d_K] = K.size
    Q: [batch_size, num_head, seq_length_2, d_K]
    d_K is a new hyperparameter. to set the dimension of words.
    it is a little different from my report.
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    #it is the W_Q \times W_K^T
    #now it is [batch_size, num_head, seq_length_2, seq_length_1]
    #"./math.sqrt(d_K)" is to normalization, set the Variance to 1 as report
    if mask is not None:
        #print(scores)
        #print(mask)
        #print('------attention------')
        #print("scores:",scores.shape)
        #print("mask:",mask.shape)#mask is the source_mask
        scores = scores.masked_fill(mask == 0, -1e9)
        #after softmax, masked place will set as 0
    p_attention = F.softmax(scores,dim=-1)
    #[batch_size, num_head, seq_length_2, seq_length_1]
    if dropout is not None:
        p_attention = dropout(p_attention)
    return torch.matmul(p_attention,value),p_attention
    #after matmul ,it is [batch_size, num_head, seq_length_2, d_k]
    #p_attention is not necessary to return.

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
        #define 4 linear network. [d_model,d_model],in a list
        #two type of parameters to train
        #weights.size =[d_model,d_model]
        #bias.size = [d_model]
        self.attention = None
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask = None):
        '''
        query:[batch_size,seq_len_2,d_model]
        key:[batch_size,seq_len_1,d_model]  = value
        '''
        #print('\n\n\nmulti head attention forward')
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query,key,value = [l(x).view(batch_size,-1,self.num_head,self.d_K)
                           .transpose(1,2) for l,x in 
                           zip(self.linears,(query,key,value))]
        #query:[batch_size, seq_len_2,d_model] -->linear network -->
        #[batch_size, seq_len_2,d_model]-->view-->[batch_size,seq_len_2,num_head,d_K]
        #that's why we assert that
        #transpose(1,2)-->[batch_size,num_head,seq_len_2,d_K]
        #then apply attention on all projected vectors in batch
        #but in practice, the input q,k,v is the same x(which is source in fact)
        #thus it depends on the linears to change and differ it.
        #and the shape is same. which is [bs,seqlen,d_model] as linear is [d_model,d_model]
        #it won't change. then it view and transpose
        #we get the [bs,seqlen,num_head,d_K].
        
        x,self.attention = attention(query,key,value,
                                     mask=mask,dropout=self.dropout)
        #x.size is [batch_size, num_head, seq_len_1, d_K]
        #then apply final linear
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.num_head*self.d_K)
        #why it doesn't make it (batch_size,-1,d_model)
        #we have the assert.
        return self.linears[-1](x)

class TextNet(nn.Module):
    def __init__(self, config,dataloader):
        '''
            embedding size: the dimension of the word vector
            out_dim:
            vocab_size: how many vocabulary in the text.
            attention_dropout:
            gru_units:
            class_num: the number of sentiment. In this instance, it should be 3.
        '''
        super(TextNet,self).__init__()
        self.class_num = config['class_num']
        self.dataloader = dataloader
        #print(config['attention_head'])
        self.word_self_attention = Multi_Head_Attention(config['attention_head'],768) 
        #get the attention of the text.
        
        self.output_layer1 = nn.Linear(768,config['class_num'])
        self.device = config['device']
    def forward(self,batch):
        batchsize = len(batch['tag_batch'])
        text_vector = batch['source_text_vector']
        mask = batch['source_mask_vector']
        #text_vector = self.dataloader.convert_ids_to_vector(text_inputs['sequence'])
        tag = batch['tag_batch']
        tag = F.one_hot(tag,self.class_num).float()
        
        if self.device:
            mask = mask.cuda(self.device)
            text_vector = text_vector.cuda(self.device)
            tag = tag.cuda(self.device)
        text_vector = text_vector.transpose(1,2)
        seq_emb = self.word_self_attention(text_vector,text_vector,text_vector)
        D_emb = torch.mean(seq_emb,dim=1)
        D_emb.reshape(batchsize,-1)
        output = self.output_layer1(D_emb)
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
        
        batchsize = len(batch['source_text_vector'])
        text_vector = batch['source_text_vector']
        mask = batch['source_mask_vector']
        
        if self.device:
            mask = mask.cuda(self.device)
            text_vector = text_vector.cuda(self.device)
            
        text_vector = text_vector.transpose(1,2)
        seq_emb = self.word_self_attention(text_vector,text_vector,text_vector)
        D_emb = torch.mean(seq_emb,dim=1)
        D_emb.reshape(batchsize,-1)
        output = self.output_layer1(D_emb)
        output = output.squeeze(dim=1)
        
        return output
    

