from specialtokens import SpecialTokens
import torch

class Evaluator(object):
    def __init__(self,tokenizer) -> None:
        specialtokens = [SpecialTokens.PAD_TOKEN,SpecialTokens.CLS_TOKEN,SpecialTokens.SEP_TOKEN]
        self.data_special_ids = [tokenizer.convert_tokens_to_ids(w) for w in specialtokens]
        self.pred_special_ids = [tokenizer.convert_tokens_to_ids(w) for w in [SpecialTokens.PAD_TOKEN]]

    def measure(self,label,pred):
        pred = torch.argmax(pred,-1)
        count = 0
        label = label.cuda("cuda:0")
        if torch.eq(pred,label):
            count = count + 1
        return count