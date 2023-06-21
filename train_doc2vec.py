import gensim
import jieba
import config
import os
import jsonlines

# 数据预处理
def preprocess(text):
    # 中文分词
    words = jieba.cut(text)
    return list(words)

def read_corpus():
    data = []
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/data/weibo'
    for name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, name)
        json_file_path = os.path.join(file_path, 'data.json')
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                data.append(gensim.models.doc2vec.TaggedDocument(preprocess(item['Content']), [item['ID']]))
    return data

def train():
    train_corpus = read_corpus()
    print("Data formatted")
    
    # Instantiate the model and build the vocabulary
    # DBOW, with word vectors as well
    model = gensim.models.doc2vec.Doc2Vec(dm=0,dbow_words=1,vector_size=config.dim_doc2vec, window=8, min_count=15, epochs=10)
    model.build_vocab(train_corpus)
    print("Vocabulary built")
    
    # Train the model
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Model trained")

    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.doc2vec_dir):
        os.mkdir(config.doc2vec_dir)
    # Save the model
    model.save(config.doc2vec_dir)
    print("Model saved")

if __name__ == "__main__":
    train()
    # import numpy as np
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt
    # # 加载预训练的 Doc2Vec 模型
    # model_path = config.doc2vec_dir
    # model = gensim.models.Doc2Vec.load(model_path)
    
    # # 获取文本向量
    # vectors = []
    # for i in range(len(model.dv)):
    #     vectors.append(model.dv[i])
    # vectors = np.array(vectors)

    # # 使用 t-SNE 进行降维
    # tsne = TSNE(n_components=2)
    # vectors_tsne = tsne.fit_transform(vectors)

    # # 可视化
    # plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
    # plt.savefig('doc2vec_tsne.png')  # 将绘制的图片保存成文件