import torch
import math,copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

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
    def forward(self, x):
        #print("img input shape is:",x.shape)
        x = x.float()
        x = x.permute(0,3,1,2)
        x = self.model(x)
        return x
    
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


class Image_Text_Attention(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(Image_Text_Attention, self).__init__()
        self.dropout_rate = dropout_rate

        self.l = None  # Initialize these attributes to be determined during runtime
        self.k = None
        self.img_layer = None
        self.seq_layer = None
        self.V_weight = None

        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def init_weights(self):
        initializer = nn.init.xavier_uniform_
        initializer(self.V_weight)

    def forward(self, image_emb, seq_emb, mask):
        #print("image embedding is:",image_emb)#(4,1,2048)
        print('\n\n\n image shape is:\n',image_emb.shape)
        #print("seq embedding is:",seq_emb)#(4,256)
        print("\nseq embedding shape is:\n",seq_emb.shape)
        if self.l is None or self.k is None:
            self.l = seq_emb.size(1)   # Number of sentences
            self.k = seq_emb.size(2)   # Sentence embedding dimension

            self.img_layer = nn.Linear(image_emb.size(2), 1)  # Mapping image_emb to 1 dimension
            self.seq_layer = nn.Linear(self.k, 1)              # Mapping seq_emb to 1 dimension

            self.V_weight = nn.Parameter(torch.Tensor(self.l, self.l))
            self.init_weights()

        # Linear mapping
        p = self.img_layer(image_emb)  # [1, batchsize, 1]
        q = self.seq_layer(seq_emb)    # [1, batchsize, 1]
        print(p.shape)
        print(q.shape)
        # Inner product + mapping (calculation of score)
        emb = torch.matmul(p, q.transpose(1,2))
           # [None, M, L]
        print('emb:',emb.shape)
        emb = emb + q.transpose(1, 2)              # [None, 1, L]
        emb = torch.matmul(emb, self.V_weight)     # [None, 1, L]
        score = self.dropout_layer(emb)            # Random dropout (optional)
        print("score:",score.shape)
        # Masking
        mask = mask.unsqueeze(1).expand(-1, score.size(1), -1)  # [None, 1, L], copying mask tensor to the same shape as the score tensor
        padding = torch.ones_like(mask) * (-2 ** 31 + 1)
        score = torch.where(mask == 0, padding, score)
        score = F.softmax(score, dim=-1)         # [None, M, L]

        # Weighted sum of vectors
        output = torch.matmul(score, seq_emb)   # [None, M, k]
        output /= self.k**0.5                    # Normalization
        return output



class VistaNet(nn.Module):
    def __init__(self, config,dataloader):
        '''
            embedding size: the dimension of the word vector
            out_dim:
            vocab_size: how many vocabulary in the text.
            attention_dropout:
            gru_units:
            class_num: the number of sentiment. In this instance, it should be 3.
        '''
        super(VistaNet,self).__init__()
        self.class_num = config['class_num']
        self.dataloader = dataloader
        self.vgg16 = VggNet(out_dim=config['out_dim']) #img to vector
        
        self.word_self_attention = Multi_Head_Attention(config['attention_head'],768) 
        #get the attention of the text. it also uses the BiGRU_layer1 
        self.img_seq_attention = Image_Text_Attention(0.1)
        #
        self.doc_self_attention = Multi_Head_Attention(config['attention_head'],768)
        
        self.output_layer = nn.Linear(config['embedding_dim'],config['class_num'])
    
    def forward(self,batch):
        image_inputs = batch['source_img_batch']
        text_inputs = self.dataloader.truncate_tensor(batch['source_index_batch'])#(batchsize,maxlen)
        text_vector = self.dataloader.convert_ids_to_vector(text_inputs['sequence'])
        tag = batch['tag_batch']
        tag = F.one_hot(tag,self.class_num).float()
        
        image_emb =self.vgg16(image_inputs)
        #一些博客认为数据形状是（224，224，3），但是在这里会报错，因此我加了一个permute来让其适配
        #正确的应该是3，224，224.
        #image_emb is (batch_size,out_dim)=(4,4096)
        text_vector = text_vector.transpose(1,2)
        seq_emb = self.word_self_attention(text_vector,text_vector,text_vector)
        #print("\nseq emb:\n",seq_emb.shape)
        #(4,54,768)
        image_emb = image_emb.unsqueeze(1)#(batch_size,1,4096)
        #print("\nimage emb\n",image_emb.shape)
        #(4,1,2048)
        mask = torch.argmax(text_inputs['mask'],dim=1)
        mask = mask.unsqueeze(1)
        print("\nmask:\n",mask.shape)
        doc_emb = self.img_seq_attention(image_emb,seq_emb,mask)
        D_emb = self.doc_self_attention(doc_emb,doc_emb,doc_emb)
        print("\nD_emb:\n",D_emb.shape)
        output = self.output_layer(D_emb)
        output = output.squeeze(dim=1)
        
        loss = nn.CrossEntropyLoss()
        loss_dict = {
            "result":output,
            "loss":loss(output,tag)
        }
        return loss_dict


if __name__ == "__main__":
    from dataset import Dataset
    from dataloader import Dataloader
    config = {
        "out_dim":2048,
        "embedding_dim":768,
        "attention_dropout":0.2,
        "max_input_len":50,
        "pretrained_model_name":"bert-base-cased",
        "model":"vistanet",
        "validset_divide":0.2,
        "device":"cuda:0",
        'resume_training':False,
        'resume':False,
        'train_batch_size':4,
        'test_batch_size':1,
        'max_len':50,
        "shuffle":True,
        "class_num":3,
        "attention_head":2,
        "epoch":50
    }
    
    dataset = Dataset(config)
    dataset._load_train_and_valid_dataset()
    dataset._load_test_dataset()
    dataloader = Dataloader(config,dataset)
    #print(dataloader.vocab_size)
    dataloader.load_data('train')
    batch = dataloader.load_next_batch("train")
    print("\n\noriginal batch:-----------------------------------------------------------------------------\n\n\n\n",batch['source_text_batch'])
    print(batch['source_index_batch'])
    print(batch['source_img_batch'].shape)
    print(batch['tag_batch'])
    print("\n\n--------------------------------------------------------------\n\n")
    #print(batch['source_img_batch'].shape)
    #dataloader.__build_batch()
    model = VistaNet(config,dataloader)
    loss = model(batch)
    print(loss)
    pass
