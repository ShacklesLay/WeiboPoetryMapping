import logging
import os
import json
import numpy.random
import torch.cuda
import random


def set_logger(log_dir, model_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        file_handler = logging.FileHandler(log_dir)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

def random_select_candidate(file_path,num=10):
    data = []
    with open(file_path,'r') as f:
        for line in f.readlines():
            text = json.loads(line)['ID']
            data.append(text)
    random.shuffle(data)
    return data[:num]

def train_dev_split(file_path):
    provinces = os.listdir(file_path)
    random.shuffle(provinces)
    return provinces[:25],provinces[25:]

if __name__=="__main__":
    print(random_select_candidate("./data/weibo/安徽/data.json"))
    