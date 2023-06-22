import torch
dim_doc2vec = 50

batch_size = 31
lr = 0.0001
epochs = 10

candidate_num = 10

ground_truth_type = 'GDP'   #Choice["Med","Engel","GDP"]
# device
device = torch.device('cuda')

log_dir = './log'
model_dir = "./results"
doc2vec_dir = model_dir+"/doc2vec"+"/doc2vec.model"

text_dir = './data/weibo'
image_dir = './data/nightlight_download/'
NL_model_dir = model_dir+"/NL_model"+"_lr_"+str(lr)+"_batch_size_"+str(batch_size)+"_epochs_"+str(epochs)+"_ground_truth_type_"+ground_truth_type                   
Multi_model_dir = model_dir+"/Mul_model"+"_lr_"+str(lr)+"_batch_size_"+str(batch_size)+"_epochs_"+str(epochs)+"_ground_truth_type_"+ground_truth_type                   