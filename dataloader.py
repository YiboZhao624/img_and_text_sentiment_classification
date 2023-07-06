import math
import torch
from typing import List
import numpy as np

from abstract_dataloader import AbstractDataLoader
from transformers import AutoTokenizer


class Dataloader(AbstractDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.max_input_len = config["max_input_len"]
        
        try:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model_name'])
        except:
            print("something error, we can't load the pretrained model. Check it please.")
        
        dataset = self.pretrained_tokenizer
        self.model = config['model']
        self.__init_batches()
    
    def build_batch(self,batch_data):
        source_text_batch = []
        source_index_batch = []
        source_img_batch = []
        tag_batch = []
        for data in batch_data:
            text_data = [w for w in data['text'].strip()]
            img_data = data['img']
            tag = data['tag']

            text_index_list = list()
            for word in text_data:
                text_index_list.append(self.pretrained_tokenizer.convert_tokens_to_ids(word))
            
            source_text_batch.append(text_data)
            source_index_batch.append(text_index_list)
            source_img_batch.append(img_data)
            tag_batch.append(tag)
        
        return {
            "source_text_batch":source_text_batch,
            "source_index_batch":source_index_batch,
            "source_img_batch":source_img_batch,
            "tag_batch":tag_batch
        }
    
    def __init_batches(self):
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
    
    def truncate_tensor(self, sequence):
        max_len = 0
        for instance in sequence:
            max_len = max(len(instance), max_len)
        result_batch_tag_list = list()
        for instance in sequence:
            one_tag_list = []
            one_tag_list.extend(instance)
            len_diff = max_len - len(one_tag_list)
            for _ in range(len_diff):
                one_tag_list.append(self.pretrained_tokenizer.convert_tokens_to_ids('<-PAD->')) # for padding
            result_batch_tag_list.append(one_tag_list)

        result_batch_tag_matrix = np.array(result_batch_tag_list)
        result_batch_tag_matrix = torch.tensor(result_batch_tag_matrix)

        return result_batch_tag_matrix