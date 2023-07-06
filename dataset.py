#to split the word and turn to index
import cv2
import random

def read_one(name:str):
    text_path = "./data/" + name + ".txt"
    img_path = "./data/" + name + ".jpg"
    try:
        try:
            with open(text_path,encoding="utf-8") as f:
                text = f.readlines()
        except:
            with open(text_path,encoding='ANSI') as f:
                text = f.readlines()
    except:
        text = None
        img = None
        return text,img
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))
    return text,img


class Dataset(object):
    def __init__(self,config) -> None:
        super().__init__()
        self.model = config["model"]
        self.validset_divide = config["validset_divide"]
        self.device = config["device"]
        self.resume_training = config['resume_training'] if config['resume_training'] else config['resume']

        self.trainset = []
        self.validset = []
        self.testset = []

    def load_train_and_valid_dataset(self,file='train.txt'):
        dataset = []
        split = self.validset_divide
        try:
            with open(file,encoding='utf-8') as f:
                next(f)
                for line in f:
                    data = line
                    id = data.split(',')[0]
                    tag = data.split(',')[1]
                    textdata,imgdata = read_one(id)
                    one_piece = {'text':textdata,'img':imgdata,'tag':tag}
                    dataset.append(one_piece)
        except FileNotFoundError:
            print(f"File {file} not found.")
            return
        
        if split:
            val_length = int(len(dataset)*split)
            val_data = random.sample(dataset,val_length)
            train_data = [x for x in data if x not in val_data]
            self.validset = val_data
            self.trainset = train_data

    def load_test_dataset(self,file='test_without_label.txt'):
        testdata = []
        try:
            with open(file,encoding='utf-8') as f:
                next(f)
                for line in f:
                    data = line
                    id = data.split(',')[0]
                    textdata,imgdata = read_one(id)
                    one_piece = {'text':textdata,'img':imgdata,'tag':None}
                    testdata.append(one_piece)
        except FileNotFoundError:
            print(f"File {file} not found.")
            return
        self.testset = testdata

if __name__ == "__main__":
    pass