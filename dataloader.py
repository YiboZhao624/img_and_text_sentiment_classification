import math
import re
import torch
from typing import List
import numpy as np

from abstract_dataloader import AbstractDataLoader
from transformers import AutoTokenizer, BertTokenizer, BertModel
from dataset import Dataset
from specialtokens import SpecialTokens

class Dataloader(AbstractDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.max_input_len = config["max_input_len"]
        special_tokens = [SpecialTokens.__dict__[k] for k in SpecialTokens.__dict__ if not re.search('^\_', k)]
        special_tokens.sort()
        
        try:
            self.pretrained_tokenizer = BertTokenizer.from_pretrained("pretrained_model/vocab.txt")
            #self.pretrained_tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})
            self.embedder = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
            self.embedder.eval()
            #please check your settings and ensure your VPN is off
            #I've trapped in this fucking stupid bug for 20 min
        except:
            print("something error, we can't load the pretrained model. Check it please.")
        self.vocab_size = self.pretrained_tokenizer.vocab_size
        dataset = self.pretrained_tokenizer
        self.model = config['model']
        self.__init_batches()
    
    def __build_batch(self,batch_data):
        source_text_batch = []
        source_index_batch = []
        source_img_batch = []
        source_embedded_batch = []
        tag_batch = []
        for data in batch_data:
            #print(batch_data)
            text_data = [w for w in data['text'][0].strip('\n').strip().split(' ')]
            img_data = data['img']
            tag = data['tag']

            sequence = ""
            '''
            it is the data from the social media, 
            thus I replace the tag with <TAG> and the content, 
            replace the @somebody with special token <USER>
            '''
            for word in text_data:
                '''if word.startswith("@"):
                    word = "[USER]"
                if word.startswith("#"):
                    sequence += '[TAG]'+' '
                    word = word[1:]
                if word == "RT":
                    word = 'retweet'
                '''
                sequence += word + ' '
            #print(sequence)
            marked_text = "[CLS] " + sequence + " [SEP]"
            tokens = self.pretrained_tokenizer.tokenize(marked_text)
            text_index_list = self.pretrained_tokenizer.convert_tokens_to_ids(tokens)

            source_text_batch.append(tokens)
            source_index_batch.append(text_index_list)
            source_img_batch.append(img_data)
            tag_batch.append(tag)
        #print(source_text_batch)
        #print(source_index_batch[0])
        source_img_batch = np.array(source_img_batch)
        #print(source_img_batch)
        if tag_batch[0] != None:
            return {
                "source_text_batch":source_text_batch,
                "source_index_batch":source_index_batch,
                "source_img_batch":torch.from_numpy(source_img_batch),#.to(torch.uint8)
                "tag_batch":torch.tensor(tag_batch)
            }
        else:
            return {
                "source_text_batch":source_text_batch,
                "source_index_batch":source_index_batch,
                "source_img_batch":torch.from_numpy(source_img_batch).to(torch.uint8),
                "tag_batch":None
            }

    def __init_batches(self):
        #print("it worked")
        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []
        for set_type in ['train', 'valid', 'test']:
            if set_type == 'train':
                datas = self.dataset.trainset
                batch_size = self.train_batch_size
            elif set_type == 'valid':
                datas = self.dataset.validset
                batch_size = self.test_batch_size
            elif set_type == 'test':
                datas = self.dataset.testset
                batch_size = self.test_batch_size
            else:
                raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
            num_total = len(datas)
            batch_num = math.ceil(num_total / batch_size)

            for batch_i in range(batch_num):
                start_idx = batch_i * batch_size
                end_idx = (batch_i + 1) * batch_size
                if end_idx <= num_total:
                    batch_data = datas[start_idx:end_idx]
                else:
                    batch_data = datas[start_idx:num_total]
                built_batch = self.__build_batch(batch_data)
                if set_type == 'train':
                    self.trainset_batches.append(built_batch)
                elif set_type == 'valid':
                    self.validset_batches.append(built_batch)
                elif set_type == 'test':
                    self.testset_batches.append(built_batch)
                else:
                    raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = len(self.trainset_batches)
        self.validset_batch_nums = len(self.validset_batches)
        self.testset_batch_nums = len(self.testset_batches)

    def convert_ids_to_vector(self,tokens_tensor):
        #tokens_tensor = torch.tensor(indexed_tokens)
        #print(tokens_tensor)
        dim0,dim1 = tokens_tensor.shape
        batch_vector = []
        for i in range(dim0):
            with torch.no_grad():
                outputs = self.embedder(tokens_tensor[i].unsqueeze(0))
                hidden_state = outputs[2]
            #print(hidden_state[0].shape)
                last_four_layer = torch.stack(hidden_state[-4:])
            #print('last fout stack is',last_four_layer.shape)
                average = torch.mean(last_four_layer,dim=0)
                batch_vector.append(average)
        return torch.tensor([item.cpu().detach().numpy() for item in batch_vector])

    
    def truncate_tensor(self, sequence):
        max_len = 0
        for instance in sequence:
            max_len = max(len(instance), max_len)
        result_batch_tag_list = list()
        result_batch_mask_list = list()
        for instance in sequence:
            one_tag_list = []
            one_mask_list = []
            one_tag_list.extend(instance)
            len_diff = max_len - len(one_tag_list)
            for _ in range(len(one_tag_list)):
                one_mask_list.append(0)
            for _ in range(len_diff):
                one_tag_list.append(self.pretrained_tokenizer.convert_tokens_to_ids('[PAD]')) # for padding
                one_mask_list.append(1)
            result_batch_tag_list.append(one_tag_list)
            result_batch_mask_list.append(one_mask_list)
        result_batch_mask_matrix = np.array(result_batch_mask_list)
        result_batch_mask_matrix = torch.tensor(result_batch_mask_matrix)
        result_batch_tag_matrix = np.array(result_batch_tag_list)
        result_batch_tag_matrix = torch.tensor(result_batch_tag_matrix)

        return {"sequence":result_batch_tag_matrix,"mask":result_batch_mask_matrix}
    
if __name__ =="__main__":
    config = {
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
    }
    
    dataset = Dataset(config)
    dataset._load_train_and_valid_dataset()
    dataset._load_test_dataset()
    dataloader = Dataloader(config,dataset)
    print(dataloader.vocab_size)
    dataloader.load_data('train')
    batch = dataloader.load_next_batch("train")
    print(batch)
    print(batch['source_img_batch'].shape)
    #dataloader.__build_batch()
    
    pass
