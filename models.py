import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class NL_model(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,3,padding=1)
        # 255x255
        self.conv2 = nn.Conv2d(1,1,5,padding=2)
        # 255x255
        self.mlp = nn.Sequential(
            nn.Linear(65025,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, x):
        # x.shape=(N,1,255,255)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x,1)
        res = self.mlp(x)
        return res

class Multi_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,3,padding=1)
        # 255x255
        self.conv2 = nn.Conv2d(1,1,5,padding=2)
        # 255x255
        self.mlp1 = nn.Sequential(
            nn.Linear(65025,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(config.candidate_num*config.dim_doc2vec,512),
            nn.ReLU(),
            nn.Linear(512,256),
        )
        self.head = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, x1, x2):
        # x1.shape=(N,1,255,255)
        # x2.shape=(N,config.candidate_num,config.dim_doc2vec)
        x = self.conv1(x1)
        x = self.conv2(x)
        x = torch.flatten(x,1)
        x = self.mlp1(x)
        
        y = torch.flatten(x2,1)
        y = self.mlp2(y)
        
        z = torch.cat((x,y),1)
        res = self.head(z)
        return res