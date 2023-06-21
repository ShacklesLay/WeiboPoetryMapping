import torch
from torch.utils.data import Dataset
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import config
from utils import random_select_candidate
import gensim
import numpy as np

with open("./data/label.json",'r') as f:
    pro2label = json.loads(f.read())



class Image_Dataset(Dataset):
    def __init__(self,image_dir) -> None:
        self.image_dir = image_dir
        self.transform = transforms.Compose([
                                transforms.Resize((255, 255)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])
                            ])
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index):
        img_dir, province = self.image_dir[index]
        img = Image.open(img_dir).convert('L')
        img = self.transform(img)
        label = pro2label[province][config.ground_truth_type]
        return img,label

class Multi_Dataset(Dataset):
    def __init__(self,image_dir,text_dir) -> None:
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transforms.Compose([
                                transforms.Resize((255, 255)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])
                            ])
        self.dec2vec =  gensim.models.Doc2Vec.load(config.doc2vec_dir)
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, index):
        img_dir, province = self.image_dir[index]
        img = Image.open(img_dir).convert('L')
        img = self.transform(img)
        label = pro2label[province][config.ground_truth_type]
        text_path = os.path.join(self.text_dir,province,'data.json')
        textIDs = random_select_candidate(text_path,config.candidate_num)
        vectors = []
        for id in textIDs:
            vec = self.dec2vec.dv[id]
            vectors.append(vec)
        vectors = torch.tensor(np.array(vectors))
        return img,vectors,label
            
if __name__ == "__main__":
    import config
    dataset=Multi_Dataset(config.image_dir)
    img,label = dataset[0]
    img = torch.flatten(img)
    print(img.shape)
    print(label)
        