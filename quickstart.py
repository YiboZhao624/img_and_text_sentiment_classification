from dataset import Dataset
from dataloader.dataloader import Dataloader
from models.vistanet import VistaNet
from models.reducenet import ReduceNet
import json,math,tqdm
from trainer.supervised_trainer import SupervisedTrainer
from evaluator import Evaluator
import logging
import warnings
import os,torch
from models.model_img_side import VggNet
from models.model_text_side import TextNet
 
warnings.filterwarnings("ignore")

def init_logger(config):

    logfilepath =config['log_file'] if config['log_file'] else config['log_path']

    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        handlers=[fh, sh]
    )


def quick_start(config):
    dataset = Dataset(config)
    dataset._load_train_and_valid_dataset(file=config["train_file"])
    dataset._load_test_dataset(file=config['test_file'])
    dataloader = Dataloader(config,dataset)
    if config['model'] == "reducenet":
        model = ReduceNet(config,dataloader)
    elif config['model'] == "vistanet":
        model = VistaNet(config,dataloader)
    elif config['model'] == "vgg":
        model = VggNet(config)
    elif config['model'] == 'textnet':
        model = TextNet(config,dataloader)
    if config['device']:
        model = model.to(config['device'])
    evaluator = Evaluator(dataloader.pretrained_tokenizer)
    trainer = SupervisedTrainer(config,model,dataloader,evaluator)
    if config['training_resume'] or config['resume']:
        trainer._load_checkpoint()
    init_logger(config)
    logger = logging.getLogger()
    logger.info(model)
    trainer.fit()



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    modelname = "textnet" #vistanet,vggnet,textnet,reducenet
    #vistanet is the multimodal model.
    #vggnet is img only and the textnet is text only.
    f = open('config_settings/config_'+modelname+".json")
    config = json.load(f)
    torch.cuda.empty_cache()
    quick_start(config)