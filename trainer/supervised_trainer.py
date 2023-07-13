import os
import time
import math,json
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from trainer.abstract_trainer import AbstractTrainer
from specialtokens import SpecialTokens

def time_since(s):
    """compute time

    Args:
        s (float): the amount of time in seconds.

    Returns:
        (str) : formatting time.
    """
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


class SupervisedTrainer(AbstractTrainer):
    """supervised trainer, used to implement training, testing, parameter searching in supervised learning.
    
    example of instantiation:
        
        >>> trainer = SupervisedTrainer(config, model, dataloader, evaluator)

        for training:
            
            >>> trainer.fit()
        
        for testing:
            
            >>> trainer.test()
        
        for parameter searching:

            >>> trainer.param_search()
    """

    def __init__(self, config, model, dataloader, evaluator):
        """
        Args:
            config (config): An instance object of Config, used to record parameter information.
            model (Model): An object of deep-learning model. 
            dataloader (Dataloader): dataloader object.
            evaluator (Evaluator): evaluator object.
        """
        super().__init__(config, model, dataloader, evaluator)
        self._build_optimizer()
        self.display_train_step = config['display_train_step']
        self.test_step = config["test_step"]
        self.class_num = config["class_num"]
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "start_epoch": self.epoch_i,
            "best_valid_cor_f1": self.best_valid_cor_f1,
            "best_test_cor_f1": self.best_test_cor_f1,
        }
        checkpoint_dir = self.config['checkpoint_dir']
        if not os.path.abspath(checkpoint_dir):
            checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        save_dir = checkpoint_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_file = os.path.join(save_dir, 'trainer_checkpoint.pth')
        torch.save(check_pnt, model_file)
        np.save(save_dir+"/config.npy",self.config)

    def _load_checkpoint(self):
        load_dir = self.config['checkpoint_dir']
        model_file = os.path.join(load_dir, 'trainer_checkpoint.pth')
        check_pnt = torch.load(model_file, map_location=self.config["map_location"])#map_location is none
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["start_epoch"]
        self.best_valid_cor_f1 = check_pnt["best_valid_cor_f1"]
        self.best_test_cor_f1 = check_pnt["best_test_cor_f1"]
        
    def _save_predit(self, predict_out):
        save_dir = self.config['checkpoint_dir']
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        predict_file = os.path.join(save_dir, 'predicts.json')
        write_json_data(predict_out, predict_file)

    def _idx2word_2idx(self, batch_equation):
        batch_size, length = batch_equation.size()
        batch_equation_ = []
        for b in range(batch_size):
            equation = []
            for idx in range(length):
                equation.append(self.dataloader.dataset.out_symbol2idx[ \
                                    self.dataloader.dataset.in_idx2word[ \
                                        batch_equation[b, idx]]])
            batch_equation_.append(equation)
        batch_equation_ = torch.LongTensor(batch_equation_).to(self.config["device"])
        return batch_equation_

    def recover_num(self, rewrite_list, target_batch):
        new_rewrite_list = []
        source_unk_list = [w for w in target_batch if w not in self.dataloader.pretrained_tokenzier.vocab]
        unk_id = 0
        for w in rewrite_list:
            if w == self.dataloader.pretrained_tokenzier.unk_token and unk_id < len(source_unk_list):
                new_rewrite_list += [source_unk_list[unk_id]]
                unk_id += 1
            else:
                new_rewrite_list += [w]
        return new_rewrite_list

    def _train_batch(self, batch):
        batch_loss = self.model(batch)
        return batch_loss

    def _eval_batch(self, batch, is_display=False):

        gen_start = time.time()
        '''
        batch_loss = self.model(batch, self.dataloader)
        dec_f1s, cor_f1s = [batch_loss["loss"].item()], [batch_loss["loss"].item()]
        '''
        test_out = self.model.model_test(batch, self.dataloader, is_display)
        batch_size = len(test_out)
        loss = torch.nn.CrossEntropyLoss()
        cor_f1s = []
        predict_out = []
        losses = 0
        for idx in range(batch_size):
            cor_f1 = self.evaluator.measure(batch['tag_batch'][idx],
                                                    test_out[idx])
            tag = batch['tag_batch'][idx]
            tag = F.one_hot(tag,self.class_num).float().to("cpu")
            test = test_out[idx].to('cpu')
            loss_one = loss(test,tag)
            losses += loss_one
            cor_f1s.append(cor_f1)
            
            pred_out = {'id': ' '.join(batch['id'][idx]),
                        'tag': ' '.join(str(test_out[idx][0].item()))}
            predict_out.append(pred_out)

        return predict_out, cor_f1s,losses

    def _train_epoch(self):
        epoch_start_time = time.time()
        loss_total = defaultdict(list)
        self.model.train()
        for batch_idx, _ in enumerate(self.dataloader.load_data('train')):
            
            self.batch_idx = batch_idx + 1
            batch = self.dataloader.load_next_batch('train')
            self.optimizer.zero_grad()
            batch_loss = self._train_batch(batch)

            for k in batch_loss:
                if 'loss' in k:
                    loss_total[k] += [batch_loss[k].item()]

            batch_loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            #if batch_idx %50 == 0:
            #    print("50 batches finished")


        self.optimizer.zero_grad()
        self.model.zero_grad()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        """train model.
        """
        train_batch_size = self.config["train_batch_size"]
        epoch_nums = self.config["epoch"]

        self._save_checkpoint()
        
        self.train_batch_nums = math.ceil(self.dataloader.trainset_nums / train_batch_size)
        
        self.logger.info("start training...")
        #self.test()
        for epo in tqdm(range(self.start_epoch, epoch_nums), desc='train process '):
            self.epoch_i = epo + 1
            self.model.train()
            loss_total, train_time_cost = self._train_epoch()
            
            if epo % self.display_train_step == 0 or epo > epoch_nums - 5:
                logging_output = ""
                for l in loss_total:
                    logging_output += " " + l + " |"
                    logging_output += "[%2.3f]" %(np.sum(loss_total[l])*1./self.train_batch_nums)

                self.logger.info("epoch [%3d] train time %s | " %(self.epoch_i, train_time_cost) + logging_output)
                # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s" \
                #                  % (self.epoch_i, loss_total / self.train_batch_nums, train_time_cost))

            if epo >= 0 and (epo % self.test_step == 0) or (epo > epoch_nums - 5):
                torch.cuda.empty_cache()
                _, valid_cor_f1, valid_loss ,valid_total, valid_time_cost = self.evaluate('valid')
                torch.cuda.empty_cache()
                self.logger.info(
                    "---------- valid total [%d] | valid cor f1 [%2.3f] | valid loss [%2.3f] | valid time %s" \
                    % (valid_total, valid_cor_f1, valid_loss, valid_time_cost))

                self.test()
                torch.cuda.empty_cache()
                # self.logger.info(
                #     "---------- test total [%d] | test det f1 [%2.3f] | test cor f1 [%2.3f] | test time %s" \
                #     % (test_total, test_det_f1, test_cor_f1, test_time_cost))

                if valid_cor_f1 >= self.best_valid_cor_f1:
                    self.best_valid_cor_f1 = valid_cor_f1
                    self._save_output()

                    self._save_checkpoint()
                    
            # if epo % 5 == 0:
            #     self._save_checkpoint()
        self.logger.info('''training finished.
                            best valid result: correction F1 [%2.3f]''' \
                         % (self.best_valid_cor_f1))

    def evaluate(self, eval_set):
        """evaluate model.

        Args:
            eval_set (str): [valid | test], the dataset for evaluation.

        Returns:
            tuple(float,float,int,str):
            detection F1, correction F1, count of evaluated datas, formatted time string of evaluation time.
        """
        self.model.eval()
        self.model.zero_grad()
        cor_f1 = 0
        eval_total = 0
        loss = 0
        predict_out = []

        if eval_set == 'valid':
            batch_nums = self.dataloader.validset_batch_nums
        elif eval_set == 'test':
            batch_nums = self.dataloader.testset_batch_nums
        else:
            raise ValueError("{} type not in ['valid', 'test'].".format(eval_set))
        test_start_time = time.time()
        # for batch in self.dataloader.load_data(eval_set):
        for batch_idx in tqdm(range(batch_nums), desc='test {}set'.format(eval_set)):
            #if batch_idx > 300: continue

            batch = self.dataloader.load_next_batch(eval_set)
            pred_out, batch_cor_f1,batch_loss = self._eval_batch(batch, is_display = False)
            cor_f1 += np.sum(batch_cor_f1)
            eval_total += len(batch_cor_f1)
            loss += batch_loss
            predict_out += pred_out

        test_time_cost = time_since(time.time() - test_start_time)
        return predict_out, cor_f1 / eval_total, loss/eval_total ,eval_total, test_time_cost

    def test(self):
        """test model.
        """
        self.model.eval()
        predict_out = [["guid","tag"]]
        test_start_time = time.time()

        for batch in self.dataloader.load_data('test'):
            pred_out = self.model.model_test(batch,self.dataloader,is_display=False)
            id = batch['id']
            #print(pred_out)
            pred_out = pred_out.argmax(dim=-1)
            pred_out = pred_out.tolist()
            #print(pred_out)
            for i in range(len(id)):
                predict_out.append([id[i],pred_out[i]])

        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test time %s" \
                         % (test_time_cost))
        self._save_output()
        self._save_predit(predict_out)


