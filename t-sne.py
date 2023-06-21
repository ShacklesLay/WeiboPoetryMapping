import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import random_select_candidate
import config
import gensim
import os

#加载预训练的 Doc2Vec 模型
model_path = config.doc2vec_dir
model = gensim.models.Doc2Vec.load(model_path)

# 获取文本向量
vectors = []
ids = []
colors = np.linspace(0, 1, 31)
cc = []
for i in colors:
    cc.extend([i]*5)
cc = np.array(cc)
for name in os.listdir(config.text_dir):
    file_path = os.path.join(config.text_dir, name, 'data.json')
    ids.extend(random_select_candidate(file_path,num=5))
    
for i in ids:
    vectors.append(model.dv[i])
vectors = np.array(vectors)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2)
vectors_tsne = tsne.fit_transform(vectors)

# 可视化
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1],color=plt.get_cmap('Blues')(cc))
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
plt.savefig('doc2vec_tsne.png')  # 将绘制的图片保存成文件