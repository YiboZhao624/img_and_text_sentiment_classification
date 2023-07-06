# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/10 14:28
# @File: abstract_dataloader.py
from typing import List
import random
import math

class AbstractDataLoader(object):
    """abstract dataloader

    the base class of dataloader class
    """
    def __init__(self, config, dataset):
        """
        """
        super().__init__()
        self.model = config["model"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.share_vocab = config["share_vocab"]
        
        self.max_len = config["max_len"]
        self.device = config["device"]
        self.shuffle = config["shuffle"]

        self.dataset = dataset
        self.in_pad_token = None
        self.in_unk_token = None

        self.out_pad_token = None
        self.out_unk_token = None
        self.temp_unk_token = None
        self.temp_pad_token = None

        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []
        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = 0
        self.validset_batch_nums = 0
        self.testset_batch_nums = 0

    def _pad_input_batch(self, batch_seq, batch_seq_len):
        max_length = max(batch_seq_len)
        if self.max_len is not None:
            if self.max_len < max_length:
                max_length = self.max_len
        for idx, length in enumerate(batch_seq_len):
            if length < max_length:
                batch_seq[idx] += [self.in_pad_token for i in range(max_length - length)]
            else:
                if self.add_sos and self.add_eos:
                    batch_seq[idx] = [batch_seq[idx][0]] + batch_seq[idx][1:max_length-1] + [batch_seq[idx][-1]]
                else:
                    batch_seq[idx] = batch_seq[idx][:max_length]
        return batch_seq

    def _pad_output_batch(self, batch_target, batch_target_len):
        max_length = max(batch_target_len)
        if self.max_equ_len is not None:
            if self.max_equ_len < max_length:
                max_length = self.max_equ_len
        for idx, length in enumerate(batch_target_len):
            if length < max_length:
                batch_target[idx] += [self.out_pad_token for i in range(max_length - length)]
            else:
                batch_target[idx] = batch_target[idx][:max_length]
        return batch_target

    def _get_mask(self, batch_seq_len):
        max_length = max(batch_seq_len)
        batch_mask = []
        for idx, length in enumerate(batch_seq_len):
            batch_mask.append([1] * length + [0] * (max_length - length))
        return batch_mask
    
    def _get_input_mask(self, batch_seq_len):
        if self.max_len:
            max_length = self.max_len
        else:
            max_length = max(batch_seq_len)
        batch_mask = []
        for idx, length in enumerate(batch_seq_len):
            batch_mask.append([1] * length + [0] * (max_length - length))
        return batch_mask

    def _build_num_stack(self, equation, num_list):
        num_stack = []
        for word in equation:
            temp_num = []
            flag_not = True
            if word not in self.dataset.out_idx2symbol:
                flag_not = False
                if "NUM" in word:
                    temp_num.append(int(word[4:]))
                for i, j in enumerate(num_list):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(num_list))])
        num_stack.reverse()
        return num_stack

    def load_data(self,type:str):
        """
        Load batches, return every batch data in a generator object.

        :param type: [train | valid | test], data type.
        :return: Generator[dict], batches
        """
        if type == "train":
            self.__trainset_batch_idx=-1
            if self.shuffle:
                random.shuffle(self.trainset_batches)
            for batch in self.trainset_batches:
                self.__trainset_batch_idx = (self.__trainset_batch_idx + 1) % self.trainset_batch_nums
                yield batch
        elif type == "valid":
            self.__validset_batch_idx=-1
            for batch in self.validset_batches:
                self.__validset_batch_idx = (self.__validset_batch_idx + 1) % self.validset_batch_nums
                yield batch
        elif type == "test":
            self.__testset_batch_idx=-1
            for batch in self.testset_batches:
                self.__testset_batch_idx = (self.__testset_batch_idx + 1) % self.testset_batch_nums
                yield batch
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))


    def load_next_batch(self,type:str)->dict:
        """
        Return next batch data
        :param type: [train | valid | test], data type.
        :return: batch data
                """
        if type == "train":
            self.__trainset_batch_idx=(self.__trainset_batch_idx+1)%self.trainset_batch_nums
            return self.trainset_batches[self.__trainset_batch_idx]
        elif type == "valid":
            self.__validset_batch_idx = (self.__validset_batch_idx + 1) % self.validset_batch_nums
            return self.validset_batches[self.__validset_batch_idx]
        elif type == "test":
            self.__testset_batch_idx = (self.__testset_batch_idx + 1) % self.testset_batch_nums
            return self.testset_batches[self.__testset_batch_idx]
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))


    def init_batches(self):
        """
        Initialize batches of trainset, validset and testset.
        :return: None
                """
        self.__init_batches()

    def __init_batches(self):
        self.trainset_batches=[]
        self.validset_batches=[]
        self.testset_batches=[]
        for set_type in ['train','valid','test']:
            if set_type=='train':
                datas = self.dataset.trainset
                batch_size = self.train_batch_size
            elif set_type=='valid':
                datas = self.dataset.validset
                batch_size = self.test_batch_size
            elif set_type=='test':
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
        self.__trainset_batch_idx=-1
        self.__validset_batch_idx=-1
        self.__testset_batch_idx=-1
        self.trainset_batch_nums=len(self.trainset_batches)
        self.validset_batch_nums=len(self.validset_batches)
        self.testset_batch_nums=len(self.testset_batches)






