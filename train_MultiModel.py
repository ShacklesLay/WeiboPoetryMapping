from utils import set_logger, set_seed
import config
import logging
from my_dataset import *
from models import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from utils import train_dev_split
import gensim

def pearson_r2(x,y):
    # 计算标准化后的张量
    x_norm = (x - x.mean()) / x.std()
    y_norm = (y - y.mean()) / y.std()

    # 计算皮尔森相关系数
    similarity = F.cosine_similarity(x_norm, y_norm, dim=0)
    correlation = similarity.item()
    return correlation**2

def evaluate(dev_loader, model):
    model.eval()
    with torch.no_grad():
        r2 = []
        for _, batch_sample in enumerate(tqdm(dev_loader)):
            batch_data,batch_vectors, batch_label = batch_sample
            batch_data = batch_data.to(config.device)
            batch_label = batch_label.to(config.device)
            batch_label = batch_label.float()
            batch_vectors = batch_vectors.to(config.device)
            outputs = model(batch_data,batch_vectors)
            label = batch_label.unsqueeze(1)
            
            p_r2 = pearson_r2(outputs, label)
            r2.append(p_r2)
    return sum(r2)/len(r2)
            

def train_epoch(train_loader, model, optimizer):
    model.train()
    train_loss = 0.0
    criterion  = nn.MSELoss()
    for _, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_vectors, batch_label = batch_sample
        batch_data = batch_data.to(config.device)
        batch_label = batch_label.to(config.device)
        batch_label = batch_label.float()
        batch_vectors = batch_vectors.to(config.device)
        outputs = model(batch_data,batch_vectors)
        
        label = batch_label.unsqueeze(1)
        loss = criterion(outputs, label)
        train_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = float(train_loss) / len(train_loader)
    return train_loss

def train(train_loader, dev_loader, model, optimizer):
    best_r2 = 0.0
    
    for epoch in range(1, config.epochs+1):
        train_loss = train_epoch(train_loader, model, optimizer)
        r2 = evaluate(dev_loader, model)
        logging.info('Epoch: {}, Train_loss: {}, R2: {}'.format(epoch, train_loss, r2))
        if r2 > best_r2 and epoch >= 5:
            torch.save(model.state_dict(),config.Multi_model_dir)
            logging.info('Model Saved!')
            logging.info('find the better recall score!')
            best_r2 = r2
    logging.info('Best recall: {}'.format(best_r2))
    logging.info('Training Finished!')

def run():
    set_logger(config.log_dir, config.model_dir)
    logging.info('device: {}'.format(config.device))
    set_seed(42)
    
    train_provinces,dev_provinces = train_dev_split(config.text_dir)
    train_dirs = []
    dev_dirs = []
    for file in os.listdir(config.image_dir):
        file_path = os.path.join(config.image_dir,file)
        if os.path.isdir(file_path):
            for name in os.listdir(file_path):
                province = name.split('-')[0]
                if province in dev_provinces:
                    dev_dirs.append([os.path.join(file_path,name),province])
                else:
                    train_dirs.append([os.path.join(file_path,name),province])
                    
    trainset = Multi_Dataset(train_dirs,config.text_dir)
    validset = Multi_Dataset(dev_dirs,config.text_dir)
    logging.info('Dataset Build!')
    
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(validset, batch_size=config.batch_size, shuffle=False)
    logging.info('Dataloader Build!')
    
    model = Multi_model()
    model.to(config.device)
    
    optimizer = Adam(model.parameters(), lr=config.lr)
    logging.info('Starting Training')
    train(train_loader, dev_loader, model, optimizer)
    

if __name__ == "__main__":
    run()